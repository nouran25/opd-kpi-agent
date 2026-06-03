"""Data loading utilities for the OPD KPI agent."""

from difflib import SequenceMatcher
import re

import pandas as pd


class OPDDataLoader:
    """Load OPD KPI data and knowledge-base metadata."""

    def __init__(self, config):
        self.config = config
        self.df = pd.DataFrame()
        self.knowledge_base = {}
        self.kpi_catalog = {}
        self.kpi_alias_index = {}

    def load_all(self):
        """Load the KPI dataset and knowledge-base workbook."""
        self.df = pd.read_excel(
            self.config.dataset_path,
            sheet_name="OPD_KPI_Dataset",
        )
        self.df.columns = self.df.columns.str.strip()
        self._prepare_dataset()
        self._load_knowledge_base()
        self._build_kpi_catalog()
        return self

    def _prepare_dataset(self):
        if "Month" in self.df.columns:
            self.df["Date"] = pd.to_datetime(self.df["Month"])
            self.df["YearMonth"] = self.df["Date"].dt.to_period("M")
            self.df["Year"] = self.df["Date"].dt.year
            self.df["Month_Num"] = self.df["Date"].dt.month

        if {"Total Revenue", "Target Revenue"}.issubset(self.df.columns):
            self.df["Revenue_Achievement_%"] = (
                self.df["Total Revenue"] / self.df["Target Revenue"] * 100
            )

        if {"No. Cases", "Target No. cases"}.issubset(self.df.columns):
            self.df["Cases_Achievement_%"] = (
                self.df["No. Cases"] / self.df["Target No. cases"] * 100
            )

        if {"Total Revenue", "No. Cases"}.issubset(self.df.columns):
            self.df["Revenue_per_Case"] = (
                self.df["Total Revenue"] / self.df["No. Cases"]
            )

        if {"Total Leakage Revenue Losses", "Total Revenue"}.issubset(self.df.columns):
            self.df["Leakage_Impact_%"] = (
                self.df["Total Leakage Revenue Losses"] / self.df["Total Revenue"] * 100
            )

    def _load_knowledge_base(self):
        if not self.config.knowledge_path.exists():
            return

        try:
            self.knowledge_base = pd.read_excel(
                self.config.knowledge_path,
                sheet_name=None,
            )
        except Exception as exc:
            print(f"Warning: Could not load knowledge base: {exc}")
            self.knowledge_base = {}

    def _build_kpi_catalog(self):
        """Build a KPI catalog from the knowledge base and dataset columns."""
        names = set()
        for sheet in self.knowledge_base.values():
            if "KPI_Name" in sheet.columns:
                names.update(sheet["KPI_Name"].dropna().astype(str))
            if "KPI" in sheet.columns:
                names.update(sheet["KPI"].dropna().astype(str))
            if "Parent_KPI" in sheet.columns:
                names.update(sheet["Parent_KPI"].dropna().astype(str))
            if "Child_KPI" in sheet.columns:
                names.update(sheet["Child_KPI"].dropna().astype(str))

        names.update(str(column) for column in self.df.columns)

        data_columns_by_key = {
            self.normalize_lookup_text(column): column for column in self.df.columns
        }

        self.kpi_catalog = {}
        self.kpi_alias_index = {}
        for name in sorted(names):
            key = self.normalize_lookup_text(name)
            dataset_column = data_columns_by_key.get(key)
            aliases = self._generate_aliases(name)
            if dataset_column:
                aliases.update(self._generate_aliases(dataset_column))

            self.kpi_catalog[name] = {
                "name": name,
                "dataset_column": dataset_column,
                "aliases": sorted(aliases),
            }

            if dataset_column:
                for alias in aliases:
                    self.kpi_alias_index[alias] = dataset_column

    def _generate_aliases(self, value: str) -> set[str]:
        """Generate normalized lookup forms from KPI names, without hand-coded KPI mapping."""
        raw = str(value)
        normalized = self.normalize_lookup_text(raw)
        aliases = {normalized}

        no_percent = self.normalize_lookup_text(raw.replace("%", " percent"))
        aliases.add(no_percent)
        aliases.add(no_percent.replace(" percent", "").strip())

        no_number_prefix = re.sub(r"\bno\b", "number", normalized).strip()
        aliases.add(no_number_prefix)
        aliases.add(normalized.replace("number", "no").strip())

        compact = normalized.replace(" ", "")
        if compact:
            aliases.add(compact)

        aliases.update(self._synonym_aliases(aliases))

        acronym = "".join(part[0] for part in normalized.split() if part)
        if len(acronym) >= 3:
            aliases.add(acronym)

        words = normalized.split()
        if len(words) > 1 and words[0] in {
            "actual",
            "digital",
            "doctor",
            "patient",
            "target",
            "total",
        }:
            suffix = " ".join(words[1:])
            if suffix:
                aliases.add(suffix)
                aliases.add(suffix.replace(" ", ""))

        aliases.update(self._synonym_aliases(aliases))
        aliases.update(self._prefix_suffix_aliases(aliases))
        aliases.update(self._word_order_aliases(aliases))
        aliases.update(self._synonym_aliases(aliases))
        return {alias for alias in aliases if alias}

    @staticmethod
    def _prefix_suffix_aliases(aliases: set[str]) -> set[str]:
        expanded = set()
        removable_prefixes = {
            "actual",
            "digital",
            "doctor",
            "patient",
            "target",
            "total",
        }
        for alias in aliases:
            words = alias.split()
            if len(words) > 1 and words[0] in removable_prefixes:
                suffix = " ".join(words[1:])
                expanded.add(suffix)
                expanded.add(suffix.replace(" ", ""))
        return {item.strip() for item in expanded if item and item.strip()}

    def _synonym_aliases(self, aliases: set[str]) -> set[str]:
        """Expand common KPI wording variants such as percent/percentage and CR/conversion rate."""
        expanded = set()
        for alias in list(aliases):
            words = alias.split()

            if "percent" in words:
                expanded.add(alias.replace("percent", "percentage"))
                expanded.add(alias.replace("percent", "pct"))
                expanded.add(alias.replace("percent", "rate"))

            if "percentage" in words:
                expanded.add(alias.replace("percentage", "percent"))
                expanded.add(alias.replace("percentage", "pct"))
                expanded.add(alias.replace("percentage", "rate"))

            if "pct" in words:
                expanded.add(alias.replace("pct", "percent"))
                expanded.add(alias.replace("pct", "percentage"))
                expanded.add(alias.replace("pct", "rate"))

            if "cr" in words:
                expanded.add(alias.replace("cr", "conversion rate"))
                expanded.add(alias.replace("cr", "conversion"))

            if "conversion" in words and "rate" in words:
                conversion_alias = alias.replace("conversion rate", "cr")
                expanded.add(conversion_alias)
                expanded.add(conversion_alias.replace(" cr", "").strip())

            if "avg" in words:
                expanded.add(alias.replace("avg", "average"))

            if "average" in words:
                expanded.add(alias.replace("average", "avg"))

        return {item.strip() for item in expanded if item and item.strip()}

    @staticmethod
    def _word_order_aliases(aliases: set[str]) -> set[str]:
        expanded = set()
        metric_suffixes = {"percentage", "percent", "pct", "rate", "ratio", "score"}
        for alias in aliases:
            words = alias.split()
            if len(words) >= 3 and words[0] in metric_suffixes and words[1] == "of":
                expanded.add(" ".join(words[2:] + [words[0]]))
            if len(words) >= 3 and words[0] == "average" and words[1] == "of":
                expanded.add(" ".join(words[2:] + [words[0]]))
            if len(words) >= 3 and words[0] == "avg" and words[1] == "of":
                expanded.add(" ".join(words[2:] + ["average"]))
        return {item.strip() for item in expanded if item and item.strip()}

    def get_kpi_lookup_candidates(self, text: str) -> list[str]:
        """Return ordered KPI name/alias candidates for external lookups."""
        candidates = []

        def add(value: str | None):
            if value and str(value).strip() and str(value).strip() not in candidates:
                candidates.append(str(value).strip())

        add(text)
        for alias in sorted(self._generate_aliases(text), key=len, reverse=True):
            if len(alias) >= 3:
                add(self._readable_alias(alias))

        resolved_column = self.resolve_kpi(text)
        catalog_name = self.resolve_catalog_kpi(text)
        add(catalog_name)
        add(resolved_column)

        related_items = []
        for name, item in self.kpi_catalog.items():
            if name == catalog_name or item.get("dataset_column") == resolved_column:
                related_items.append((name, item))

        for name, item in related_items:
            add(name)
            add(item.get("dataset_column"))
            for alias in sorted(item.get("aliases", []), key=len, reverse=True):
                if len(alias) >= 3:
                    add(self._readable_alias(alias))

        return candidates

    @staticmethod
    def _readable_alias(alias: str) -> str:
        words = str(alias).replace("_", " ").split()
        if not words:
            return ""
        small = {"and", "by", "for", "of", "per", "to"}
        return " ".join(
            word.upper()
            if word in {"bu", "cr", "coe", "kpi", "pms"}
            else word
            if word in small
            else word.capitalize()
            for word in words
        )

    def resolve_kpi(self, text: str) -> str | None:
        """Resolve user wording to an actual dataset KPI column."""
        normalized = self.normalize_lookup_text(text)
        compact = normalized.replace(" ", "")

        for candidate in (normalized, compact):
            if candidate in self.kpi_alias_index:
                return self.kpi_alias_index[candidate]

        matches = []
        for alias, column in self.kpi_alias_index.items():
            alias_is_phrase = len(alias.split()) > 1
            if alias and len(alias) >= 3 and alias_is_phrase and alias in normalized:
                matches.append((len(alias), column))
            elif alias and len(alias) >= 3 and alias_is_phrase and alias in compact:
                matches.append((len(alias), column))

        if not matches:
            return self._best_similarity_match(
                normalized,
                [(alias, column) for alias, column in self.kpi_alias_index.items()],
            )

        matches.sort(reverse=True)
        return matches[0][1]

    def resolve_catalog_kpi(self, text: str) -> str | None:
        """Resolve user wording to any KPI in the catalog, including non-dataset KPIs."""
        normalized = self.normalize_lookup_text(text)
        compact = normalized.replace(" ", "")
        direct_matches = []
        fuzzy_matches = []

        for name, item in self.kpi_catalog.items():
            aliases = item.get("aliases", [])
            for alias in aliases:
                alias_compact = alias.replace(" ", "")
                if normalized == alias or compact == alias_compact:
                    direct_matches.append((len(alias), name))
                elif alias and len(alias) >= 3 and len(alias.split()) > 1 and (
                    alias in normalized or alias_compact in compact
                ):
                    fuzzy_matches.append((len(alias), name))

        matches = direct_matches or fuzzy_matches
        if not matches:
            return self._best_similarity_match(
                normalized,
                [
                    (alias, name)
                    for name, item in self.kpi_catalog.items()
                    for alias in item.get("aliases", [])
                ],
            )

        matches.sort(reverse=True)
        return matches[0][1]

    def _best_similarity_match(
        self,
        normalized_query: str,
        candidates: list[tuple[str, str]],
        min_score: float = 0.82,
    ) -> str | None:
        query_tokens = self._meaningful_lookup_tokens(normalized_query)
        query_compact = "".join(query_tokens) or normalized_query.replace(" ", "")
        best = (0.0, "")

        for alias, target in candidates:
            alias_tokens = self._meaningful_lookup_tokens(alias)
            if not alias_tokens:
                continue

            alias_compact = "".join(alias_tokens)
            sequence_score = SequenceMatcher(
                None,
                query_compact,
                alias_compact,
            ).ratio()
            overlap = len(set(query_tokens) & set(alias_tokens))
            overlap_score = overlap / max(len(set(alias_tokens)), 1)

            if len(alias_tokens) == 1 and len(query_tokens) > 1:
                sequence_score = max(
                    SequenceMatcher(None, token, alias_tokens[0]).ratio()
                    for token in query_tokens
                )
                overlap_score = 1.0 if alias_tokens[0] in query_tokens else 0.0

            score = max(sequence_score, overlap_score)
            if score > best[0]:
                best = (score, target)

        return best[1] if best[0] >= min_score else None

    @staticmethod
    def _meaningful_lookup_tokens(value: str) -> list[str]:
        stopwords = {
            "a",
            "an",
            "and",
            "by",
            "calculate",
            "compare",
            "doctor",
            "doctors",
            "each",
            "for",
            "formula",
            "give",
            "how",
            "in",
            "is",
            "kpi",
            "me",
            "of",
            "per",
            "show",
            "tell",
            "the",
            "to",
            "what",
            "which",
        }
        return [token for token in str(value).split() if token not in stopwords]

    def is_dataset_kpi(self, kpi_name: str) -> bool:
        item = self.kpi_catalog.get(kpi_name, {})
        return bool(item.get("dataset_column"))

    def resolve_bu(self, text: str) -> str | None:
        normalized = self.normalize_lookup_text(text)
        for bu in self.get_bu_list():
            if self.normalize_lookup_text(bu) == normalized:
                return bu
        return None

    def get_kpi_metadata(self, kpi_name: str) -> dict:
        """Return merged metadata for a KPI from knowledge-base sheets."""
        metadata = {"KPI_Name": kpi_name}

        for sheet in self.knowledge_base.values():
            if "KPI_Name" in sheet.columns:
                matches = sheet[sheet["KPI_Name"].astype(str) == kpi_name]
            elif "KPI" in sheet.columns:
                matches = sheet[sheet["KPI"].astype(str) == kpi_name]
            else:
                continue

            if not matches.empty:
                row = matches.iloc[0].dropna().to_dict()
                metadata.update(row)

        return metadata

    def get_kpi_relationships(self, kpi_name: str) -> pd.DataFrame:
        relationships = self.knowledge_base.get(
            "adx_kpi_relationship_map_x0009__x0009__x0009_",
            pd.DataFrame(),
        )
        if relationships.empty or "Parent_KPI" not in relationships.columns:
            return pd.DataFrame()

        return relationships[
            relationships["Parent_KPI"].astype(str) == str(kpi_name)
        ].copy()

    def get_playbook(self, kpi_name: str) -> pd.DataFrame:
        playbook = self.knowledge_base.get(
            "adx_kpi_investigation_playbook",
            pd.DataFrame(),
        )
        if playbook.empty or "KPI" not in playbook.columns:
            return pd.DataFrame()

        return playbook[playbook["KPI"].astype(str) == str(kpi_name)].copy()

    def get_doctor_list(self):
        """Get unique doctor names."""
        if "Doctor Name" not in self.df.columns:
            return []
        return self.df["Doctor Name"].dropna().unique().tolist()

    def get_doctor_bu_pairs(self):
        """Get unique doctor and BU combinations."""
        if not {"Doctor Name", "BU"}.issubset(self.df.columns):
            return []
        pairs = (
            self.df[["Doctor Name", "BU"]]
            .dropna()
            .drop_duplicates()
            .sort_values(["Doctor Name", "BU"])
        )
        return list(pairs.itertuples(index=False, name=None))

    def get_doctor_display_list(self):
        """Get display labels that treat the same doctor name in each BU separately."""
        return [
            f"{doctor} ({bu})"
            for doctor, bu in self.get_doctor_bu_pairs()
        ]

    def get_bus_for_doctor(self, doctor_name: str):
        """Get BUs where a doctor name appears."""
        if not {"Doctor Name", "BU"}.issubset(self.df.columns):
            return []
        normalized = self.normalize_lookup_text(doctor_name)
        matches = self.df[
            self.df["Doctor Name"].astype(str).map(self.normalize_lookup_text)
            == normalized
        ]
        return sorted(matches["BU"].dropna().unique().tolist())

    def get_bu_list(self):
        """Get unique business unit names."""
        if "BU" not in self.df.columns:
            return []
        return self.df["BU"].dropna().unique().tolist()

    @staticmethod
    def normalize_lookup_text(value: str) -> str:
        return re.sub(
            r"\s+",
            " ",
            re.sub(r"[^a-z0-9]+", " ", str(value).lower()),
        ).strip()
