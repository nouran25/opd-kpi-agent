"""Data loading utilities for the OPD KPI agent."""

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
            self.df["Revenue_per_Case"] = self.df["Total Revenue"] / self.df["No. Cases"]

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

        return {alias for alias in aliases if alias}

    def resolve_kpi(self, text: str) -> str | None:
        """Resolve user wording to an actual dataset KPI column."""
        normalized = self.normalize_lookup_text(text)
        compact = normalized.replace(" ", "")

        for candidate in (normalized, compact):
            if candidate in self.kpi_alias_index:
                return self.kpi_alias_index[candidate]

        matches = []
        for alias, column in self.kpi_alias_index.items():
            if alias and len(alias) >= 3 and alias in normalized:
                matches.append((len(alias), column))
            elif alias and len(alias) >= 3 and alias in compact:
                matches.append((len(alias), column))

        if not matches:
            return None

        matches.sort(reverse=True)
        return matches[0][1]

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
