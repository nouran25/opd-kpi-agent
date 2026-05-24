"""Main KPI agent with LangChain and metadata-driven analytics."""

from __future__ import annotations

import re
import sys
import json
import urllib.error
import urllib.request
from pathlib import Path

from langchain.agents import create_agent
from langchain.tools import tool
from langchain_core.messages import AIMessage, HumanMessage
from langchain_groq import ChatGroq

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.analytics.engine import AnalyticsEngine
from src.config import config
from src.data.loader import OPDDataLoader
from src.data.vector_store import KPIKnowledgeVectorStore


class OPDKpiAgent:
    """OPD KPI analytics agent.

    Dataset interpretation is driven by:
    - the actual dataframe columns in the OPD dataset workbook
    - the KPI knowledge-base workbook
    - the relationship map and investigation playbook
    """

    def __init__(self):
        self.config = config
        self.config.ensure_directories()

        print("Loading data...")
        self.data = OPDDataLoader(self.config).load_all()
        self.analytics = AnalyticsEngine(self.data)
        self.knowledge_store = self._init_knowledge_store()

        self.llm = None
        self.agent_executor = None
        self.chat_history = []

        self._init_llm()
        self._create_agent()

    def _init_llm(self):
        """Initialize hosted Groq LLM."""
        try:
            self.llm = ChatGroq(
                model=self.config.llm_model,
                temperature=self.config.temperature,
                max_tokens=self.config.llm_max_tokens,
                reasoning_effort=self.config.llm_reasoning_effort,
                timeout=self.config.llm_timeout,
                max_retries=self.config.llm_max_retries,
            )
            print(f"LLM initialized: {self.config.llm_model}")
        except Exception as exc:
            print(f"LLM not available: {exc}")
            print("Set GROQ_API_KEY in your environment and restart the agent.")
            self.llm = None

    def _init_knowledge_store(self) -> KPIKnowledgeVectorStore | None:
        """Initialize and populate the persistent Chroma knowledge store."""
        try:
            store = KPIKnowledgeVectorStore(self.config, self.data)
            indexed_count = store.sync()
            print(
                "Chroma knowledge store ready: "
                f"{indexed_count} documents at {self.config.vector_store_path}"
            )
            return store
        except Exception as exc:
            print(f"Chroma knowledge store not available: {exc}")
            return None

    def _create_agent(self):
        """Create the LangChain agent and expose safe analytical tools."""

        @tool
        def get_doctor_performance(doctor_name: str, bu_name: str = "") -> str:
            """Get performance summary for a doctor. If bu_name is omitted and the doctor works in multiple BUs, return each doctor-BU profile separately."""
            bu = self._resolve_optional_bu(bu_name)
            if bu_name and bu is None:
                return self._unknown_bu_message(bu_name)
            return self._format_doctor_performance(doctor_name, bu=bu)

        @tool
        def analyze_root_cause(kpi_name: str, bu_name: str = "") -> str:
            """Analyze root causes for a KPI using the knowledge-base relationship map. Optional bu_name can be ASH, SMH, or HJH."""
            metric = self.data.resolve_kpi(kpi_name)
            if metric is None:
                return self._unknown_metric_message(kpi_name)

            bu = self._resolve_optional_bu(bu_name)
            if bu_name and bu is None:
                return self._unknown_bu_message(bu_name)

            return self._format_root_cause(metric, bu=bu)

        @tool
        def compare_doctors(metric: str, bu_name: str = "") -> str:
            """Compare doctors by a KPI. Optional bu_name can be ASH, SMH, or HJH."""
            resolved_metric = self.data.resolve_kpi(metric)
            if resolved_metric is None:
                return self._unknown_metric_message(metric)

            bu = self._resolve_optional_bu(bu_name)
            if bu_name and bu is None:
                return self._unknown_bu_message(bu_name)

            return self._format_doctor_comparison(resolved_metric, bu=bu)

        @tool
        def compare_business_units(
            metric: str,
            bu_names: str = "",
            year: str = "",
            month: str = "",
        ) -> str:
            """Compare BUs by a KPI. bu_names can be comma-separated values such as ASH, SMH. Optional year is like 2025 and month is like March or 3."""
            resolved_metric = self.data.resolve_kpi(metric)
            if resolved_metric is None:
                return self._unknown_metric_message(metric)

            selected_bus = self._resolve_bu_list(bu_names)
            if bu_names and not selected_bus:
                return self._unknown_bu_message(bu_names)

            return self._format_bu_comparison(
                resolved_metric,
                bus=selected_bus,
                year=self._parse_year(year),
                month=self._parse_month(month),
            )

        @tool
        def get_kpi_trend(
            kpi_name: str,
            months: int = 12,
            bu_name: str = "",
            year: str = "",
        ) -> str:
            """Get monthly trend for a KPI. Optional bu_name can be ASH, SMH, or HJH. Optional year is like 2025. Leave bu_name empty for all BUs."""
            metric = self.data.resolve_kpi(kpi_name)
            if metric is None:
                return self._unknown_metric_message(kpi_name)

            bu = self._resolve_optional_bu(bu_name)
            if bu_name and bu is None:
                return self._unknown_bu_message(bu_name)

            return self._format_kpi_trend(
                metric,
                months=months,
                bu=bu,
                year=self._parse_year(year),
            )

        @tool
        def get_bu_summary(bu_name: str) -> str:
            """Get performance summary for a Business Unit."""
            bu = self.data.resolve_bu(bu_name)
            if bu is None:
                return self._unknown_bu_message(bu_name)
            return self._format_bu_summary(bu)

        @tool
        def search_kpi_knowledge(query: str) -> str:
            """Search the Chroma KPI knowledge base for definitions, formulas, relationships, investigation steps, and recommended actions. If a requested formula is not configured there, use the Dataverse formula fallback when configured."""
            return self._search_kpi_knowledge(query)

        @tool
        def lookup_dataverse_kpi_formula(kpi_name: str) -> str:
            """Look up a KPI formula in the optional Dataverse formula table when the loaded knowledge base does not contain the formula."""
            return self._format_dataverse_formula_lookup(kpi_name)

        tools = [
            get_doctor_performance,
            analyze_root_cause,
            compare_doctors,
            compare_business_units,
            get_kpi_trend,
            get_bu_summary,
            search_kpi_knowledge,
            lookup_dataverse_kpi_formula,
        ]

        if self.llm:
            self.agent_executor = create_agent(
                model=self.llm,
                tools=tools,
                system_prompt=self._system_prompt(),
            )
            print("Agent created successfully")

    def _system_prompt(self) -> str:
        kpis = ", ".join(self._available_kpi_columns()[:30])
        bus = ", ".join(self.data.get_bu_list())
        doctors = ", ".join(self.data.get_doctor_display_list()[:20])
        return (
            "You are a professional OPD KPI Analytics Agent for healthcare "
            "operations.\n\n"
            "Your role is to help users understand doctor performance, KPI trends, root causes, and recommended actions based on the OPD dataset and the KPI knowledge base.\n\n"
            "Rules:\n"
            "1. Use tools for all dataset questions. Never invent numbers.\n"
            "2. The tools resolve natural KPI wording through the loaded knowledge-base "
            "workbook and dataset columns.\n"
            "3. If the user asks to compare BUs such as ASH vs SMH, call compare_business_units, not compare_doctors.\n"
            "4. If the user mentions a BU, year, or month, pass it to the tool.\n"
            "5. Treat the same doctor name in different BUs as different doctor identities. If a doctor name is repeated across BUs and the user does not specify the BU, answer for each available doctor-BU profile separately.\n"
            "6. For root-cause questions, call analyze_root_cause.\n"
            "7. Do not assume a BU when the user does not provide one; leave bu_name empty so the tool uses all BUs.\n"
            "8. Use search_kpi_knowledge when the user asks about KPI definitions, "
            "formulas, playbooks, relationships, recommended actions, or a KPI "
            "whose wording does not resolve cleanly to a dataset column.\n"
            "9. If search_kpi_knowledge or the loaded knowledge base does not provide "
            "a configured formula, call lookup_dataverse_kpi_formula before saying the "
            "formula is unavailable. Clearly label Dataverse formulas as coming from "
            "the Dataverse KPI formula table.\n"
            "10. Explain what the numbers mean operationally. Include drivers, "
            "risks, and next actions.\n"
            "11. Keep answers concise but thorough: executive summary, evidence, "
            "interpretation, and recommendations.\n"
            "12. Stay within the 1024-token response limit and finish cleanly. Target "
            "350-500 words, or 250-400 words when using a table.\n"
            "13. Use tables when they improve clarity, especially for month-by-month "
            "trends, BU comparisons, doctor comparisons, or driver snapshots. Avoid "
            "tables for simple narrative explanations.\n"
            "14. Keep tables compact: use at most one table unless the user explicitly "
            "asks for detailed tables, include only the most useful columns, and avoid "
            "wide tables with long text inside cells.\n"
            "15. Format recommendations as a numbered action list with short bullets "
            "under each action when useful. Prefer 3 practical recommendations; use "
            "5 only when no table is included or the user asks for depth.\n"
            "16. If space is limited, prioritize the conclusion, the most important "
            "evidence, and actionable recommendations over extra explanation.\n\n"
            "17. Use 'revenue gap' only for the knowledge-base formula "
            "Target Revenue - Actual Revenue. For highest-vs-lowest BU comparisons, "
            "call the difference a BU revenue spread or BU difference.\n\n"
            f"Available BUs: {bus}.\n"
            f"Available doctors include: {doctors}.\n"
            f"Available KPI columns include: {kpis}."
            "If a doctor isn't found, list available doctors from the dataset.\n"
            "Available tools give you access to:\n"
            "- Doctor performance summaries\n"
            "- Root cause analysis with statistical variance\n"
            "- Doctor comparisons on any metric\n"
            "- KPI definitions and formulas\n"
            "- Semantic Chroma search over the KPI knowledge base\n"
            "- Trend analysis over time\n"
            "- BU-level summaries\n"
            "- Comprehensive reports\n"
            "Always respond in a professional, helpful manner."
        )

    def chat(self, user_input: str) -> str:
        """Process user message."""
        if self.agent_executor is None:
            return self._fallback_response()

        try:
            direct_response = self._direct_structured_response(user_input)
            if direct_response:
                self.chat_history.extend(
                    [HumanMessage(user_input), AIMessage(direct_response)]
                )
                return direct_response

            messages = self.chat_history + [HumanMessage(user_input)]
            response = self.agent_executor.invoke({"messages": messages})
            output = self._message_content_to_text(response["messages"][-1])
            self.chat_history.extend([HumanMessage(user_input), AIMessage(output)])
            return output
        except Exception as exc:
            return f"Error: {exc}"

    def _direct_structured_response(self, user_input: str) -> str | None:
        """Answer common KPI requests deterministically from the loaded dataset."""
        normalized = self.data.normalize_lookup_text(user_input)
        doctor = self._extract_doctor_from_text(user_input)
        bu = self._extract_bu_from_text(user_input)
        year = self._extract_year_from_text(user_input)
        month = self._extract_month_from_text(user_input)
        metric = self.data.resolve_kpi(user_input)
        catalog_metric = self.data.resolve_catalog_kpi(user_input)

        asks_for_justification = any(
            term in normalized
            for term in ["justify", "justification", "explain", "why", "performance"]
        )
        asks_for_root_cause = any(
            term in normalized
            for term in [
                "root cause",
                "root causes",
                "causing",
                "cause of",
                "why",
            ]
        )
        asks_for_knowledge = any(
            term in normalized
            for term in [
                "knowledge base",
                "knowledgebase",
                "definition",
                "formula",
                "owner",
                "business question",
                "investigation",
                "playbook",
                "recommended action",
                "recommendation",
                "relationship",
                "related",
            ]
        )
        asks_for_kpi_overview = any(
            term in normalized
            for term in [
                "tell me about",
                "overview",
                "in general",
                "what is",
                "describe",
            ]
        )

        if self._asks_total_revenue_diagnostic_checklist(normalized):
            return self._format_total_revenue_diagnostic_checklist(
                bu=bu,
                year=year,
                month=month,
            )

        if self._asks_doctor_pms_pricing_controls(normalized):
            return self._format_doctor_pms_pricing_controls()

        if self._asks_revenue_achievement_gap_analysis(normalized):
            return self._format_revenue_achievement_gap_analysis(
                year=year,
                bu=bu,
            )

        if metric and self._asks_patient_level_unavailable(normalized):
            return self._format_patient_level_unavailable_request(
                metric,
                bu=bu,
                year=year,
                month=month,
            )

        if (
            metric is None
            and catalog_metric
            and not self.data.is_dataset_kpi(catalog_metric)
            and not asks_for_knowledge
            and self._asks_for_missing_data_request(normalized)
        ):
            derived_response = self._format_dataverse_derived_kpi(
                catalog_metric,
                normalized=normalized,
                bu=bu,
                year=year,
                month=month,
                query=user_input,
            )
            if derived_response:
                return derived_response
            return self._format_generic_missing_kpi_request(
                catalog_metric,
                bu=bu,
                year=year,
                month=month,
                query=user_input,
            )

        if (
            metric is None
            and not asks_for_knowledge
            and self._asks_for_missing_data_request(normalized)
        ):
            requested_kpi = catalog_metric or self._extract_formula_lookup_kpi_name(
                user_input
            )
            derived_response = self._format_dataverse_derived_kpi(
                requested_kpi,
                normalized=normalized,
                bu=bu,
                year=year,
                month=month,
                query=user_input,
            )
            if derived_response:
                return derived_response
            return self._format_generic_missing_kpi_request(
                requested_kpi,
                bu=bu,
                year=year,
                month=month,
                query=user_input,
            )

        if metric is None and (bu or year or month) and asks_for_kpi_overview:
            lookup_kpi = self._extract_formula_lookup_kpi_name(user_input)
            dataverse_formula = self._lookup_dataverse_kpi_formula(lookup_kpi)
            if dataverse_formula.get("found"):
                return self._format_dataverse_formula_result(
                    lookup_kpi,
                    dataverse_formula,
                    scope=self._scope_text(bu=bu, year=year, month=month),
                    value_unavailable=True,
                )

        if asks_for_knowledge and metric is None:
            lookup_kpi = self._extract_formula_lookup_kpi_name(user_input)
            dataverse_formula = self._lookup_dataverse_kpi_formula(lookup_kpi)
            if dataverse_formula.get("found"):
                return self._format_dataverse_formula_result(
                    lookup_kpi,
                    dataverse_formula,
                )
            return self._unknown_knowledge_kpi_message(
                user_input,
                dataverse_status=dataverse_formula.get("status", ""),
            )

        if metric and asks_for_knowledge:
            lookup_kpi = self._extract_formula_lookup_kpi_name(user_input)
            if (
                lookup_kpi
                and self.data.normalize_lookup_text(lookup_kpi)
                != self.data.normalize_lookup_text(metric)
            ):
                dataverse_formula = self._lookup_dataverse_kpi_formula(lookup_kpi)
                if dataverse_formula.get("found"):
                    return self._format_dataverse_formula_result(
                        lookup_kpi,
                        dataverse_formula,
                    )
                if "formula" in normalized:
                    return self._unknown_knowledge_kpi_message(
                        lookup_kpi,
                        dataverse_status=dataverse_formula.get("status", ""),
                    )

        if metric and asks_for_root_cause and not normalized.startswith("search"):
            root_cause = self._format_root_cause(metric, bu=bu)
            if asks_for_knowledge:
                knowledge = self._format_kpi_knowledge_lookup(metric, user_input)
                return f"{root_cause}\n\n{knowledge}"
            return root_cause

        if metric and asks_for_knowledge:
            return self._format_kpi_knowledge_lookup(metric, user_input)

        if metric and self._asks_for_doctor_ranking(normalized):
            requested_kpi = self._extract_formula_lookup_kpi_name(user_input)
            if (
                requested_kpi
                and self.data.normalize_lookup_text(requested_kpi)
                != self.data.normalize_lookup_text(metric)
            ):
                derived_response = self._format_dataverse_derived_kpi(
                    requested_kpi,
                    normalized=normalized,
                    bu=bu,
                    year=year,
                    month=month,
                    query=user_input,
                )
                if derived_response:
                    return derived_response
            return self._format_doctor_comparison(
                metric,
                bu=bu,
                year=year,
                month=month,
            )

        if metric and asks_for_kpi_overview:
            requested_kpi = self._extract_formula_lookup_kpi_name(user_input)
            if (
                requested_kpi
                and self.data.normalize_lookup_text(requested_kpi)
                != self.data.normalize_lookup_text(metric)
            ):
                derived_response = self._format_dataverse_derived_kpi(
                    requested_kpi,
                    normalized=normalized,
                    bu=bu,
                    year=year,
                    month=month,
                    query=user_input,
                )
                if derived_response:
                    return derived_response
            if self._asks_for_kpi_value(normalized, bu=bu, year=year, month=month):
                return self._format_kpi_value(metric, bu=bu, year=year, month=month)
            return self._format_kpi_knowledge_lookup(metric, user_input)

        if doctor and asks_for_justification:
            if not bu and self._doctor_has_multiple_bus(doctor):
                if metric:
                    return self._format_all_doctor_bu_kpi_justifications(
                        doctor,
                        metric,
                    )
                return self._format_all_doctor_bu_profiles(
                    doctor,
                    year=year,
                    month=month,
                )
            if metric:
                return self._format_doctor_kpi_justification(doctor, metric, bu=bu)
            return self._format_doctor_kpi_profile(
                doctor,
                bu=bu,
                year=year,
                month=month,
            )

        return None

    @staticmethod
    def _asks_for_kpi_value(
        normalized: str,
        bu: str | None = None,
        year: int | None = None,
        month: int | None = None,
    ) -> bool:
        if bu or year or month:
            return True
        return any(
            term in normalized
            for term in [
                "value",
                "current",
                "latest",
                "actual",
                "percentage",
                "percent",
                "rate",
                "score",
                "how much",
                "how many",
            ]
        )

    @staticmethod
    def _asks_total_revenue_diagnostic_checklist(normalized: str) -> bool:
        return (
            "total revenue" in normalized
            and any(term in normalized for term in ["diagnostic", "checklist", "diagnosis"])
        )

    @staticmethod
    def _asks_doctor_pms_pricing_controls(normalized: str) -> bool:
        mentions_pms = "doctor pms" in normalized or re.search(r"\bpms\b", normalized)
        mentions_pricing = any(
            term in normalized
            for term in ["pricing", "discount", "charge per case", "price controls"]
        )
        mentions_relationship = any(
            term in normalized
            for term in ["related", "relationship", "ties", "tie", "linked", "how is"]
        )
        return bool(mentions_pms and mentions_pricing and mentions_relationship)

    @staticmethod
    def _asks_patient_level_unavailable(normalized: str) -> bool:
        asks_patient_level = any(
            term in normalized
            for term in [
                "patient level",
                "patient-level",
                "patient details",
                "patient detail",
                "individual patient",
                "appointment id",
                "patient id",
                "patient name",
                "raw patient",
            ]
        )
        asks_detail = any(
            term in normalized
            for term in ["details", "detail", "list", "show", "give me", "raw"]
        )
        return asks_patient_level or ("patient" in normalized and asks_detail)

    @staticmethod
    def _asks_for_missing_data_request(normalized: str) -> bool:
        return any(
            term in normalized
            for term in [
                "calculate",
                "what is",
                "which",
                "who",
                "show",
                "compare",
                "rank",
                "highest",
                "lowest",
                "trend",
                "by doctor",
                "each doctor",
                "for each doctor",
                "by bu",
                "by payer",
            ]
        )

    @staticmethod
    def _asks_for_doctor_ranking(normalized: str) -> bool:
        mentions_doctor = "doctor" in normalized or "physician" in normalized
        asks_ranking = any(
            term in normalized
            for term in [
                "which",
                "who",
                "highest",
                "lowest",
                "top",
                "rank",
                "ranking",
                "compare",
                "by doctor",
                "each doctor",
                "for each doctor",
            ]
        )
        return mentions_doctor and asks_ranking

    @staticmethod
    def _asks_revenue_achievement_gap_analysis(normalized: str) -> bool:
        return (
            "revenue achievement" in normalized
            and "target" in normalized
            and any(term in normalized for term in ["gap", "driver", "drivers", "recommendation", "analyze"])
        )

    def _format_total_revenue_diagnostic_checklist(
        self,
        bu: str | None = None,
        year: int | None = None,
        month: int | None = None,
    ) -> str:
        df = self._scoped_df(bu=bu, year=year, month=month)
        if df.empty:
            return f"No Total Revenue diagnostic data{self._scope_text(bu=bu, year=year, month=month)}."

        current, previous = self.analytics._current_previous_periods(df)
        current_label = self._period_label(current)
        previous_label = self._period_label(previous)
        total_revenue = self._sum_metric(current, "Total Revenue")
        previous_revenue = self._sum_metric(previous, "Total Revenue")
        revenue_change = self._pct_change(total_revenue, previous_revenue)

        def flag_line(label: str, value: str, status: str, action: str) -> str:
            return f"- {label}: {value} -> {status}. {action}"

        rows = []

        cases = self._sum_metric(current, "No. Cases")
        prev_cases = self._sum_metric(previous, "No. Cases")
        case_change = self._pct_change(cases, prev_cases)
        rows.append(
            flag_line(
                "Case volume",
                f"{cases:,.0f} cases ({case_change:+.1f}% vs {previous_label})",
                "FLAG" if case_change <= 0 else "OK",
                "Drill into BU and doctor case counts if flat or declining.",
            )
        )

        charge = self._mean_metric(current, "Charge per case")
        prev_charge = self._mean_metric(previous, "Charge per case")
        charge_change = self._pct_change(charge, prev_charge)
        rows.append(
            flag_line(
                "Average yield",
                f"{self._format_metric_value('Charge per case', charge)} ({charge_change:+.1f}% MoM)",
                "FLAG" if charge_change < -5 else "OK",
                "Check case mix, payer mix, pricing, and discount approvals.",
            )
        )

        leakage = self._sum_metric(current, "Total Leakage Revenue Losses")
        leakage_pct = (leakage / total_revenue * 100) if total_revenue else 0
        rows.append(
            flag_line(
                "Leakage",
                f"{self._format_metric_value('Total Leakage Revenue Losses', leakage)} ({leakage_pct:.1f}% of revenue)",
                "FLAG" if leakage_pct > 5 else "OK",
                "Reconcile services rendered against invoices and missed coding.",
            )
        )

        cancelled = self._sum_metric(current, "No. Cancelled Clinics")
        planned_slots = self._sum_metric(current, "No. Planned booking Slots")
        cancelled_pct = (cancelled / planned_slots * 100) if planned_slots else 0
        rows.append(
            flag_line(
                "Cancelled clinics",
                f"{cancelled:,.0f} ({cancelled_pct:.1f}% of planned slots)",
                "FLAG" if cancelled_pct > 3 else "OK",
                "Review scheduling policy and cancellation notice discipline.",
            )
        )

        no_show = self._mean_metric(current, "No-Show %")
        no_show_pct = no_show * 100 if abs(no_show) <= 1.5 else no_show
        rows.append(
            flag_line(
                "No-show rate",
                f"{no_show_pct:.1f}%",
                "FLAG" if no_show_pct > 10 else "OK",
                "Strengthen reminders and same-day slot-fill workflows.",
            )
        )

        cash = self._sum_metric(current, "Cash Revenue")
        credit = self._sum_metric(current, "Credit Revenue")
        prev_cash = self._sum_metric(previous, "Cash Revenue")
        prev_credit = self._sum_metric(previous, "Credit Revenue")
        credit_cash = (credit / cash) if cash else 0
        prev_credit_cash = (prev_credit / prev_cash) if prev_cash else 0
        mix_change = self._pct_change(credit_cash, prev_credit_cash)
        rows.append(
            flag_line(
                "Revenue mix",
                f"credit/cash {credit_cash:.2f} ({mix_change:+.1f}% MoM)",
                "FLAG" if mix_change > 5 else "OK",
                "Check denials, reimbursement aging, and cash-package conversion.",
            )
        )

        pms = self._mean_metric(current, "Doctor PMS %")
        pms_pct = pms * 100 if abs(pms) <= 1.5 else pms
        rows.append(
            flag_line(
                "PMS / pricing-control proxy",
                f"Doctor PMS {pms_pct:.1f}%",
                "FLAG" if pms_pct < 90 else "OK",
                "Use PMS as a supporting signal, then validate charge per case, payer mix, discount approvals, and bundle coding separately.",
            )
        )

        coe = self._mean_metric(current, "Actual COE Compliance %")
        digital = self._mean_metric(current, "Digital Actual CR%")
        coe_pct = coe * 100 if abs(coe) <= 1.5 else coe
        digital_pct = digital * 100 if abs(digital) <= 1.5 else digital
        rows.append(
            flag_line(
                "Service-level compliance",
                f"COE {coe_pct:.1f}%, Digital CR {digital_pct:.1f}%",
                "FLAG" if coe_pct < 90 or digital_pct < 90 else "OK",
                "Validate under-billing or rejection risk in non-compliant workflows.",
            )
        )

        cancellation_losses = self._sum_metric(
            current, "Total Losses Revenue_Cancellation_Modification"
        )
        cancellation_loss_pct = (
            cancellation_losses / total_revenue * 100 if total_revenue else 0
        )
        rows.append(
            flag_line(
                "Cancellation / modification losses",
                f"{self._format_metric_value('Total Losses Revenue_Cancellation_Modification', cancellation_losses)} ({cancellation_loss_pct:.1f}% of revenue)",
                "FLAG" if cancellation_loss_pct > 2 else "OK",
                "Review post-booking cancellation and modification patterns.",
            )
        )

        retention = self._mean_metric(current, "Patient Retention %")
        acquisition = self._mean_metric(current, "Patient Acquisition %")
        retention_pct = retention * 100 if abs(retention) <= 1.5 else retention
        acquisition_pct = acquisition * 100 if abs(acquisition) <= 1.5 else acquisition
        rows.append(
            flag_line(
                "Patient retention / acquisition",
                f"retention {retention_pct:.1f}%, acquisition {acquisition_pct:.1f}%",
                "FLAG" if retention_pct < 80 else "OK",
                "Check repeat-visit discipline, follow-up conversion, and acquisition sources.",
            )
        )

        flagged_count = sum(1 for row in rows if "-> FLAG." in row)
        scope = self._scope_text(bu=bu, year=year, month=month)
        return "\n".join(
            [
                f"Total Revenue Diagnostic Checklist{scope}",
                "",
                "Executive readout:",
                (
                    f"{current_label} Total Revenue is {self._format_metric_value('Total Revenue', total_revenue)}, "
                    f"changing {revenue_change:+.1f}% versus {previous_label}. "
                    f"{flagged_count} of 10 diagnostic checks are flagged."
                ),
                "",
                "Latest-month checks:",
                *rows,
                "",
                "Next drill-down:",
                "- Start with FLAG rows, then compare affected doctors or BUs using the same KPI.",
                "- For revenue erosion, prioritize case volume, charge per case, leakage, PMS, and cancellation losses first.",
            ]
        )

    def _format_doctor_pms_pricing_controls(self) -> str:
        return "\n".join(
            [
                "Doctor PMS % and Pricing / Discount Controls",
                "",
                "You are right: in the knowledge base, Doctor PMS % is formally defined as `(Achieved PMS Score / Total PMS Score) x 100`. That makes it a doctor performance/compliance KPI, not a direct discount-control formula.",
                "",
                "The relationship to Pricing / Discount Controls is indirect:",
                "- The relationship map links Doctor PMS % to Revenue Achievement and Compliance Metrics.",
                "- The relationship map also links Charge per case to Doctor PMS % as a low-weight influence.",
                "- So PMS can support a pricing investigation, but it should not be used alone as proof of discount leakage or price-policy violation.",
                "",
                "Practical use:",
                "- If Doctor PMS % is low and Charge per case is also falling, investigate pricing, discounts, case mix, payer mix, and bundle coding.",
                "- If Doctor PMS % is low but Charge per case is stable, treat it primarily as a performance/compliance issue.",
                "- Discount approvals need their own source KPI or audit data; Doctor PMS % is only a proxy signal.",
            ]
        )

    def _format_revenue_achievement_gap_analysis(
        self,
        year: int | None = None,
        bu: str | None = None,
    ) -> str:
        selected_year = year or int(self.data.df["Year"].max())
        df = self._scoped_df(bu=bu, year=selected_year)
        required = {
            "Target Revenue",
            "Total Revenue",
            "Target No. cases",
            "No. Cases",
        }
        if df.empty or not required.issubset(df.columns):
            return f"No revenue achievement data found for {selected_year}{self._scope_text(bu=bu)}."

        target_revenue = self._sum_metric(df, "Target Revenue")
        actual_revenue = self._sum_metric(df, "Total Revenue")
        gap = target_revenue - actual_revenue
        achievement = (actual_revenue / target_revenue * 100) if target_revenue else 0
        target_cases = self._sum_metric(df, "Target No. cases")
        actual_cases = self._sum_metric(df, "No. Cases")
        target_yield = target_revenue / target_cases if target_cases else 0
        actual_yield = actual_revenue / actual_cases if actual_cases else 0

        candidates = [
            (
                "Case volume shortfall",
                max(target_cases - actual_cases, 0) * target_yield,
                f"{actual_cases:,.0f} cases vs {target_cases:,.0f} target",
            ),
            (
                "Cancellation/modification losses",
                self._sum_metric(df, "Total Losses Revenue_Cancellation_Modification"),
                "lost revenue from cancelled or modified activity",
            ),
            (
                "Revenue leakage",
                self._sum_metric(df, "Total Leakage Revenue Losses"),
                "missed services or workflow leakage",
            ),
            (
                "Yield per case below target",
                max(target_yield - actual_yield, 0) * actual_cases,
                f"${actual_yield:,.0f} actual vs ${target_yield:,.0f} implied target per case",
            ),
        ]
        drivers = sorted(candidates, key=lambda item: item[1], reverse=True)[:3]
        no_show = self._mean_metric(df, "No-Show %")
        no_show_pct = no_show * 100 if abs(no_show) <= 1.5 else no_show
        scope = self._scope_text(bu=bu, year=selected_year)

        lines = [
            f"Revenue achievement{scope}: {achievement:.1f}% of target.",
            f"Actual revenue was {self._format_metric_value('Total Revenue', actual_revenue)} vs target {self._format_metric_value('Target Revenue', target_revenue)}, leaving a gap of {self._format_metric_value('Total Revenue', gap)}.",
            "",
            "Top 3 possible drivers:",
        ]
        for index, (name, impact, detail) in enumerate(drivers, start=1):
            lines.append(
                f"{index}. {name}: approx. {self._format_metric_value('Total Revenue', impact)} impact; {detail}."
            )
        lines.extend(
            [
                "",
                "Knowledge-base basis: Target Revenue and Total Revenue map the gap to cases, charge per case, leakage, cancelled clinics, and no-show. Average no-show was "
                f"{no_show_pct:.1f}%, supporting the volume hypothesis.",
                "",
                "Executive recommendation: prioritize volume recovery first, then run a cancellation/leakage control sprint by BU and doctor; protect charge per case through payer/service-mix review.",
                "",
                "Assumptions and limitations: driver impacts are directional and not additive; the extract is aggregated, so patient, payer, specialty, and rejection-level root causes cannot be proven here.",
            ]
        )
        return "\n".join(lines)

    def _format_generic_missing_kpi_request(
        self,
        kpi_name: str,
        bu: str | None = None,
        year: int | None = None,
        month: int | None = None,
        query: str = "",
    ) -> str:
        metadata = self.data.get_kpi_metadata(kpi_name)
        relationships = self.data.get_kpi_relationships(kpi_name)
        formula, formula_source, formula_lookup_status = self._formula_with_fallback(
            kpi_name,
            query=query,
        )
        needed_fields = self._infer_missing_kpi_fields(
            kpi_name,
            metadata,
            relationships,
            formula=formula,
        )
        raw_data_domain = self._infer_raw_data_domain(kpi_name, metadata)
        if self.data.resolve_catalog_kpi(kpi_name):
            unavailable_reason = (
                f"{kpi_name} is configured in the knowledge base or KPI catalog, "
                "but it is not available as a numeric column in the loaded OPD dataset."
            )
        else:
            unavailable_reason = (
                f"{kpi_name} is not available as a numeric column in the loaded "
                "OPD dataset."
            )
        request_status = self._submit_missing_data_request(
            requested_kpi=kpi_name,
            unavailable_reason=unavailable_reason,
            needed_fields=needed_fields,
            raw_data_domain=raw_data_domain,
            scope={
                "bu": bu or "all",
                "year": year or "all",
                "month": month or "all",
            },
            recommended_definition={
                "name": kpi_name,
                "numerator": self._infer_formula_part(formula, "numerator"),
                "denominator": self._infer_formula_part(formula, "denominator"),
                "grain": self._infer_kpi_grain(kpi_name, metadata),
            },
        )

        scope = self._scope_text(bu=bu, year=year, month=month)
        lines = [
            f"{kpi_name}{scope}: unavailable in the current dataset",
            "",
            "Direct answer:",
            unavailable_reason,
            "",
            "Definition context:",
            f"- Business question: {metadata.get('Business_Question', 'Not configured')}",
            f"- Formula: {formula}",
            f"- Formula source: {formula_source}",
            *(
                [f"- Formula lookup status: {formula_lookup_status}"]
                if formula_lookup_status
                else []
            ),
            f"- Primary driver: {metadata.get('Primary_Driver_KPI', 'Not configured')}",
            f"- Secondary driver: {metadata.get('Secondary_Driver_KPI', 'Not configured')}",
            "",
            "Needed data fields:",
            self._format_list(needed_fields),
            "",
            "Data request:",
            request_status,
        ]
        return "\n".join(lines)

    def _format_patient_level_unavailable_request(
        self,
        metric: str,
        bu: str | None = None,
        year: int | None = None,
        month: int | None = None,
    ) -> str:
        metadata = self.data.get_kpi_metadata(metric)
        scope = self._scope_text(bu=bu, year=year, month=month)
        needed_fields = self._patient_level_needed_fields(metric)
        unavailable_reason = (
            f"The loaded OPD dataset contains aggregated {metric} values, but it "
            "does not include patient-level raw records, appointment IDs, patient "
            "identifiers, or encounter-level event timestamps needed to list the "
            "underlying patients."
        )
        request_status = self._submit_missing_data_request(
            requested_kpi=f"Patient-level {metric}",
            unavailable_reason=unavailable_reason,
            needed_fields=needed_fields,
            raw_data_domain=self._infer_patient_level_domain(metric),
            scope={
                "bu": bu or "all",
                "year": year or "all",
                "month": month or "all",
            },
            recommended_definition={
                "name": f"Patient-level {metric}",
                "numerator": metric,
                "denominator": "Patient or appointment-level source records",
                "grain": self._infer_patient_level_grain(metric),
            },
        )

        return "\n".join(
            [
                f"Patient-level details for {metric}{scope}: unavailable",
                "",
                "Direct answer:",
                unavailable_reason,
                "",
                "What is available now:",
                "- Aggregated KPI values can be summarized by BU, doctor, and month.",
                "- Patient-level rows cannot be listed from the current extract.",
                "",
                "Knowledge-base context:",
                f"- Business question: {metadata.get('Business_Question', 'Not configured')}",
                f"- Formula: {metadata.get('Formula_Logic', metadata.get('Financial_Impact_Formula', 'Not configured'))}",
                "",
                "Needed patient-level fields:",
                self._format_list(needed_fields),
                "",
                "Data request:",
                request_status,
            ]
        )

    def _format_dataverse_derived_kpi(
        self,
        kpi_name: str,
        normalized: str,
        bu: str | None = None,
        year: int | None = None,
        month: int | None = None,
        query: str = "",
    ) -> str | None:
        formula_lookup = self._lookup_dataverse_kpi_formula(kpi_name or query)
        if not formula_lookup.get("found"):
            return None

        formula = str(formula_lookup.get("formula", ""))
        numerator_label = self._infer_formula_part(formula, "numerator")
        denominator_label = self._infer_formula_part(formula, "denominator")
        numerator_column = self.data.resolve_kpi(numerator_label)
        denominator_column = self.data.resolve_kpi(denominator_label)
        if not numerator_column or not denominator_column:
            return None

        resolved_name = formula_lookup.get("kpi_name") or kpi_name
        if self._asks_for_doctor_ranking(normalized):
            return self._format_derived_doctor_comparison(
                resolved_name,
                formula,
                numerator_column,
                denominator_column,
                bu=bu,
                year=year,
                month=month,
            )

        return self._format_derived_kpi_value(
            resolved_name,
            formula,
            numerator_column,
            denominator_column,
            bu=bu,
            year=year,
            month=month,
        )

    def _format_derived_kpi_value(
        self,
        kpi_name: str,
        formula: str,
        numerator_column: str,
        denominator_column: str,
        bu: str | None = None,
        year: int | None = None,
        month: int | None = None,
    ) -> str:
        df = self._scoped_df(bu=bu, year=year, month=month)
        if df.empty:
            return f"No source data for {kpi_name}{self._scope_text(bu=bu, year=year, month=month)}."

        if not any([bu, year, month]) and "Date" in df.columns:
            df = df[df["Date"] == df["Date"].max()]

        numerator = self._sum_metric(df, numerator_column)
        denominator = self._sum_metric(df, denominator_column)
        value = (numerator / denominator * 100) if denominator else 0.0
        scope = self._scope_text(bu=bu, year=year, month=month)
        return "\n".join(
            [
                f"{kpi_name}{scope}: {value:.1f}%",
                "",
                "Calculated from Dataverse formula:",
                f"- Formula: {formula}",
                f"- Numerator: {numerator_column} = {self._format_metric_value(numerator_column, numerator)}",
                f"- Denominator: {denominator_column} = {self._format_metric_value(denominator_column, denominator)}",
                f"- Formula source: {self.config.dataverse_formula_source_label}",
            ]
        )

    def _format_derived_doctor_comparison(
        self,
        kpi_name: str,
        formula: str,
        numerator_column: str,
        denominator_column: str,
        bu: str | None = None,
        year: int | None = None,
        month: int | None = None,
    ) -> str:
        df = self._scoped_df(bu=bu, year=year, month=month)
        required = {"Doctor Name", numerator_column, denominator_column}
        if bu is None:
            required.add("BU")
        if df.empty or not required.issubset(df.columns):
            return f"No source data for {kpi_name}{self._scope_text(bu=bu, year=year, month=month)}."

        group_columns = ["Doctor Name"] if bu else ["BU", "Doctor Name"]
        grouped = df.groupby(group_columns)[[numerator_column, denominator_column]].sum().reset_index()
        grouped[kpi_name] = grouped.apply(
            lambda row: (
                row[numerator_column] / row[denominator_column] * 100
                if row[denominator_column]
                else 0.0
            ),
            axis=1,
        )
        grouped = grouped.sort_values(kpi_name, ascending=False).head(5).reset_index(drop=True)
        grouped["rank"] = grouped.index + 1

        scope = self._scope_text(bu=bu, year=year, month=month)
        lines = [
            f"Top doctors by {kpi_name}{scope}:",
            "",
            f"Formula from {self.config.dataverse_formula_source_label}: {formula}",
        ]
        for _, row in grouped.iterrows():
            doctor_label = self._doctor_identity_label(row["Doctor Name"], row.get("BU"))
            lines.append(f"{row['rank']}. {doctor_label}: {row[kpi_name]:.1f}%")
        return "\n".join(lines)

    def _patient_level_needed_fields(self, metric: str) -> list[str]:
        metric_lower = metric.lower()
        common = [
            "Patient identifier or anonymized patient key",
            "Encounter or appointment identifier",
            "Doctor, BU, specialty, and appointment date",
        ]
        if "show" in metric_lower:
            return common + [
                "Appointment status including booked, attended, cancelled, and no-show",
                "Scheduled appointment time and check-in or attendance timestamp",
                "Cancellation timestamp and cancellation reason where applicable",
                "Reminder delivery and confirmation status where available",
            ]
        if "retention" in metric_lower or "acquisition" in metric_lower:
            return common + [
                "Visit sequence or first-visit flag",
                "Previous visit date and follow-up visit date",
                "Patient new/returning classification",
            ]
        if "revenue" in metric_lower or "charge" in metric_lower:
            return common + [
                "Service code, payer, billed amount, discount amount, and net revenue",
                "Invoice or transaction identifier",
            ]
        return common + [
            f"{metric} source value at patient or encounter level",
            "Source event timestamp and operational status",
        ]

    def _infer_patient_level_domain(self, metric: str) -> str:
        metric_lower = metric.lower()
        if any(term in metric_lower for term in ["show", "booking", "cancelled"]):
            return "appointments"
        if any(term in metric_lower for term in ["revenue", "charge", "credit", "cash"]):
            return "patient_finance"
        return "patient_encounters"

    def _infer_patient_level_grain(self, metric: str) -> str:
        metric_lower = metric.lower()
        if any(term in metric_lower for term in ["show", "booking", "cancelled"]):
            return "appointment-level row with patient key, appointment ID, doctor, BU, date, and appointment status"
        if any(term in metric_lower for term in ["revenue", "charge", "credit", "cash"]):
            return "patient encounter or invoice line with doctor, BU, payer, service, date, and amount"
        return "patient encounter-level row with patient key, doctor, BU, date, and KPI source event"

    def _infer_missing_kpi_fields(
        self,
        kpi_name: str,
        metadata: dict,
        relationships,
        formula: str = "",
    ) -> list[str]:
        fields = []
        numerator = self._infer_formula_part(formula, "numerator")
        denominator = self._infer_formula_part(formula, "denominator")
        for value in [numerator, denominator]:
            if self._is_configured_value(value):
                fields.append(
                    f"{value} by BU, doctor, date, and relevant operational grain"
                )

        kpi_text = self.data.normalize_lookup_text(kpi_name)
        if not fields and "waiting" in kpi_text:
            fields.extend(
                [
                    "Total patient waiting minutes by BU, doctor, date, and visit",
                    "Number of attended visits by BU, doctor, and date",
                ]
            )

        for value in [
            kpi_name,
            metadata.get("Primary_Driver_KPI", ""),
            metadata.get("Secondary_Driver_KPI", ""),
        ]:
            if value and str(value) != "nan" and not fields:
                fields.append(f"{value} raw value by BU, doctor, date, and relevant operational grain")

        if relationships is not None and not relationships.empty:
            for _, row in relationships.head(5).iterrows():
                child = row.get("Child_KPI")
                if child and str(child) != "nan":
                    fields.append(
                        f"{child} raw value by BU, doctor, date, and relevant operational grain"
                    )

        metadata_formula = str(
            metadata.get(
                "Formula_Logic",
                metadata.get("Financial_Impact_Formula", ""),
            )
        )
        if not fields and metadata_formula and metadata_formula != "nan":
            fields.append(
                f"Source fields needed to calculate formula: {metadata_formula}"
            )

        if not fields:
            fields.append(f"{kpi_name} source value by BU, doctor, date, and relevant operational grain")

        unique_fields = []
        for field in fields:
            if field not in unique_fields:
                unique_fields.append(field)
        return unique_fields

    def _infer_raw_data_domain(self, kpi_name: str, metadata: dict) -> str:
        text = self.data.normalize_lookup_text(
            " ".join(
                [
                    kpi_name,
                    str(metadata.get("Business_Question", "")),
                    str(metadata.get("Function_Owner", "")),
                ]
            )
        )
        if any(term in text for term in ["insurance", "claim", "payer", "credit"]):
            return "patient_claims"
        if any(
            term in text
            for term in ["booking", "slot", "appointment", "show", "waiting", "wait"]
        ):
            return "appointments"
        if any(term in text for term in ["revenue", "charge", "cash", "finance"]):
            return "finance"
        if "patient" in text:
            return "patient_encounters"
        return "opd_operations"

    @staticmethod
    def _infer_formula_part(formula: str, part: str) -> str:
        formula_text = str(formula).strip()
        if not formula_text or formula_text == "Not configured":
            return "To be defined by data owner"
        if "÷" in formula_text:
            left, right = formula_text.split("÷", 1)
            value = left if part == "numerator" else right
            return re.sub(
                r"\s*(?:x|\*)\s*100\s*$",
                "",
                value,
                flags=re.IGNORECASE,
            ).strip(" ()")
        if "/" in formula_text:
            left, right = formula_text.split("/", 1)
            value = left if part == "numerator" else right
            return re.sub(
                r"\s*(?:x|\*)\s*100\s*$",
                "",
                value,
                flags=re.IGNORECASE,
            ).strip(" ()")
        if part == "numerator":
            return formula_text
        return "To be defined by data owner"

    def _infer_kpi_grain(self, kpi_name: str, metadata: dict) -> str:
        domain = self._infer_raw_data_domain(kpi_name, metadata)
        if domain == "patient_claims":
            return "patient encounter / claim line with doctor, BU, payer, and date"
        if domain == "appointments":
            return "appointment slot with doctor, BU, date, status, and patient identifier"
        if domain == "finance":
            return "transaction or encounter with doctor, BU, payer, service, and date"
        return "row-level source data with BU, doctor, date, and relevant operational identifiers"

    def _submit_missing_data_request(
        self,
        requested_kpi: str,
        unavailable_reason: str,
        needed_fields: list[str],
        raw_data_domain: str,
        scope: dict | None = None,
        recommended_definition: dict | None = None,
    ) -> str:
        recommended_definition = recommended_definition or {
            "name": requested_kpi,
            "numerator": (
                "Approved claims"
                if "approval" in requested_kpi.lower()
                else "Rejected or denied claims"
            ),
            "denominator": "Submitted claims",
            "grain": "patient encounter / claim line with doctor, BU, payer, and date",
        }
        payload = {
            "sourceSystem": self.config.data_request_source_system,
            "requestedKpi": requested_kpi,
            "requestType": "missing_kpi_and_raw_data",
            "unavailableReason": unavailable_reason,
            "rawDataDomain": raw_data_domain,
            "neededFields": needed_fields,
            "neededFieldsText": "\n".join(f"- {field}" for field in needed_fields),
            "scope": scope or {},
            "recommendedKpiDefinition": recommended_definition,
        }

        if not self.config.power_automate_data_request_url:
            return (
                "- Power Automate request not sent because "
                "`POWER_AUTOMATE_DATA_REQUEST_URL` is not configured.\n"
                "- Configure that environment variable with the flow HTTP trigger URL "
                "to automatically request the missing KPI and patient-level raw data.\n"
                f"- Request payload preview: {json.dumps(payload, ensure_ascii=False)}"
            )

        try:
            body = json.dumps(payload).encode("utf-8")
            request = urllib.request.Request(
                self.config.power_automate_data_request_url,
                data=body,
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urllib.request.urlopen(request, timeout=10) as response:
                response_body = response.read().decode("utf-8", errors="replace")
            if response_body.strip():
                return (
                    "- Missing-data request sent to the Power Automate flow.\n"
                    f"- Flow response: {self._compact_text(response_body, max_length=300)}"
                )
            return "- Missing-data request sent to the Power Automate flow."
        except urllib.error.HTTPError as exc:
            error_body = exc.read().decode("utf-8", errors="replace")
            error_detail = (
                f"\n- Response body: {self._compact_text(error_body, max_length=600)}"
                if error_body.strip()
                else ""
            )
            if exc.code == 401:
                return (
                    "- Tried to send the missing-data request to Power Automate, but "
                    "the flow returned 401 Unauthorized.\n"
                    "- Check the Power Automate HTTP trigger authentication setting. "
                    "This app currently sends a plain POST request to the trigger URL, "
                    "so the trigger must either allow anonymous/SAS URL calls or the "
                    "app must be extended to send the required OAuth token.\n"
                    "- Also confirm that `POWER_AUTOMATE_DATA_REQUEST_URL` contains the "
                    "full HTTP POST URL from the trigger, including any query-string "
                    "signature parameters.\n"
                    f"{error_detail}\n"
                    f"- Request payload: {json.dumps(payload, ensure_ascii=False)}"
                )
            return (
                "- Tried to send the missing-data request to Power Automate, but it failed.\n"
                f"- HTTP status: {exc.code}\n"
                f"- Error: {exc}\n"
                f"{error_detail}\n"
                f"- Request payload: {json.dumps(payload, ensure_ascii=False)}"
            )
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            return (
                "- Tried to send the missing-data request to Power Automate, but it failed.\n"
                f"- Error: {exc}\n"
                f"- Request payload: {json.dumps(payload, ensure_ascii=False)}"
            )

    def _lookup_dataverse_kpi_formula(self, kpi_name: str) -> dict:
        payload = {
            "sourceSystem": self.config.data_request_source_system,
            "requestType": "kpi_formula_lookup",
            "kpiName": str(kpi_name or "").strip(),
        }

        if not payload["kpiName"]:
            return {"found": False, "status": "No KPI name provided."}

        if not self.config.dataverse_kpi_formula_lookup_url:
            return {
                "found": False,
                "status": (
                    "`DATAVERSE_KPI_FORMULA_LOOKUP_URL` is not configured, so "
                    "Dataverse formula fallback was not checked."
                ),
            }

        try:
            body = json.dumps(payload).encode("utf-8")
            request = urllib.request.Request(
                self.config.dataverse_kpi_formula_lookup_url,
                data=body,
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urllib.request.urlopen(request, timeout=10) as response:
                response_body = response.read().decode("utf-8", errors="replace")
            return self._parse_dataverse_formula_response(response_body)
        except urllib.error.HTTPError as exc:
            error_body = exc.read().decode("utf-8", errors="replace")
            detail = self._compact_text(error_body, max_length=400)
            status = f"Dataverse formula lookup failed with HTTP {exc.code}."
            if detail:
                status += f" Response body: {detail}"
            return {"found": False, "status": status}
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            return {
                "found": False,
                "status": f"Dataverse formula lookup failed: {exc}",
            }

    def _parse_dataverse_formula_response(self, response_body: str) -> dict:
        if not str(response_body or "").strip():
            return {"found": False, "status": "Dataverse formula lookup returned an empty response."}

        try:
            parsed = json.loads(response_body)
        except json.JSONDecodeError:
            return {
                "found": False,
                "status": (
                    "Dataverse formula lookup returned non-JSON content: "
                    f"{self._compact_text(response_body, max_length=300)}"
                ),
            }

        record = parsed
        if isinstance(parsed, list):
            record = parsed[0] if parsed else {}
        elif isinstance(parsed, dict):
            for key in ("record", "formulaRecord", "value", "items"):
                value = parsed.get(key)
                if isinstance(value, list):
                    record = value[0] if value else {}
                    break
                if isinstance(value, dict):
                    record = value
                    break

        if not isinstance(record, dict):
            return {"found": False, "status": "Dataverse formula lookup returned an unsupported JSON shape."}

        formula = self._first_present(
            record,
            [
                "formula",
                "Formula",
                "Formula_Logic",
                "formulaLogic",
                "adx_formula",
                "new_formula",
                "cr_formula",
            ],
        )
        found_value = record.get("found")
        found = bool(formula) if found_value is None else bool(found_value and formula)
        if not found:
            return {
                "found": False,
                "status": str(record.get("message") or "No matching Dataverse formula found."),
            }

        return {
            "found": True,
            "kpi_name": self._first_present(
                record,
                ["kpiName", "KPI_Name", "KPI", "name", "adx_kpiname", "new_kpiname"],
            ),
            "formula": formula,
            "definition": self._first_present(
                record,
                ["definition", "Definition", "description", "adx_definition", "new_definition"],
            ),
            "owner": self._first_present(
                record,
                ["owner", "Owner", "KPI_Owner_Role", "adx_owner", "new_owner"],
            ),
            "status": str(record.get("message") or "Formula found in Dataverse."),
        }

    @staticmethod
    def _first_present(record: dict, keys: list[str]) -> str:
        for key in keys:
            value = record.get(key)
            if value is not None and str(value).strip() and str(value).lower() != "nan":
                return str(value).strip()
        return ""

    def _is_configured_value(self, value: object) -> bool:
        text = str(value or "").strip()
        normalized = self.data.normalize_lookup_text(text)
        return bool(text) and normalized not in {
            "",
            "nan",
            "none",
            "null",
            "not configured",
            "not available",
            "to be defined",
            "tbd",
            "na",
            "n a",
        }

    def _formula_with_fallback(self, metric: str, query: str = "") -> tuple[str, str, str]:
        metadata = self.data.get_kpi_metadata(metric)
        formula = metadata.get(
            "Formula_Logic",
            metadata.get("Financial_Impact_Formula", "Not configured"),
        )
        if self._is_configured_value(formula):
            return str(formula), "loaded Excel knowledge base", ""

        lookup_key = metric or query
        dataverse_formula = self._lookup_dataverse_kpi_formula(lookup_key)
        if dataverse_formula.get("found"):
            return (
                str(dataverse_formula.get("formula", "")),
                self.config.dataverse_formula_source_label,
                "",
            )
        return str(formula or "Not configured"), "not configured", str(dataverse_formula.get("status", ""))

    def _format_dataverse_formula_lookup(self, kpi_name: str) -> str:
        result = self._lookup_dataverse_kpi_formula(kpi_name)
        if result.get("found"):
            return self._format_dataverse_formula_result(kpi_name, result)
        return (
            f"No Dataverse formula found for {kpi_name}.\n"
            f"Status: {result.get('status', 'No details returned.')}"
        )

    def _extract_formula_lookup_kpi_name(self, query: str) -> str:
        text = str(query or "").strip()
        if not text:
            return text

        patterns = [
            r"^\s*can\s+you\s+(?:please\s+)?(?:calculate|show|compare|rank)\s+",
            r"^\s*could\s+you\s+(?:please\s+)?(?:calculate|show|compare|rank)\s+",
            r"^\s*please\s+(?:calculate|show|compare|rank)\s+",
            r"^\s*what\s+is\s+the\s+formula\s+(?:of|for)\s+",
            r"^\s*what\s+is\s+formula\s+(?:of|for)\s+",
            r"^\s*which\s+doctor\s+(?:caused|has|had|shows|showed)\s+(?:the\s+)?(?:highest|lowest|top|worst|best)\s+",
            r"^\s*which\s+doctor\s+(?:has|had|shows|showed)\s+",
            r"^\s*who\s+(?:caused|has|had|shows|showed)\s+(?:the\s+)?(?:highest|lowest|top|worst|best)\s+",
            r"^\s*(?:show|compare|rank)\s+(?:the\s+)?(?:highest|lowest|top|worst|best)?\s*",
            r"^\s*what\s+is\s+the\s+",
            r"^\s*what\s+is\s+",
            r"^\s*formula\s+(?:of|for)\s+",
            r"^\s*calculate\s+",
            r"^\s*calculation\s+(?:of|for)\s+",
        ]
        cleaned = text
        for pattern in patterns:
            cleaned = re.sub(pattern, "", cleaned, flags=re.IGNORECASE).strip()

        cleaned = re.sub(r"^\s*the\s+", "", cleaned, flags=re.IGNORECASE).strip()
        cleaned = re.sub(
            r"^\s*doctors?\s+(?:by|per)\s+",
            "",
            cleaned,
            flags=re.IGNORECASE,
        ).strip()
        cleaned = re.sub(r"[?.!]+$", "", cleaned).strip()
        cleaned = re.sub(
            r"\b(?:by|per)\s+doctor\b.*$",
            "",
            cleaned,
            flags=re.IGNORECASE,
        ).strip()
        cleaned = re.sub(
            r"\bfor\s+(?:each|every|all)\s+doctors?\b.*$",
            "",
            cleaned,
            flags=re.IGNORECASE,
        ).strip()
        cleaned = re.sub(
            r"\b(?:for|by|per)\s+(?:each|every|all)\s+doctors?\b.*$",
            "",
            cleaned,
            flags=re.IGNORECASE,
        ).strip()
        for bu in self.data.get_bu_list():
            cleaned = re.sub(
                rf"\s+\b(?:in|for)\s+{re.escape(str(bu))}\b\s*$",
                "",
                cleaned,
                flags=re.IGNORECASE,
            ).strip()
        return cleaned or text

    def _format_dataverse_formula_result(
        self,
        requested_kpi: str,
        result: dict,
        scope: str = "",
        value_unavailable: bool = False,
    ) -> str:
        resolved_name = result.get("kpi_name") or requested_kpi
        lines = [
            (
                f"{resolved_name}{scope}: value unavailable in the current OPD dataset"
                if value_unavailable
                else f"Formula for {resolved_name}: {result.get('formula', 'Not configured')}"
            ),
            "",
            f"Source: {self.config.dataverse_formula_source_label}",
        ]
        if value_unavailable:
            lines.extend(
                [
                    f"Formula: {result.get('formula', 'Not configured')}",
                    (
                        "The formula exists in Dataverse, but the loaded workbook does "
                        "not contain this KPI as a numeric column, so I should not "
                        "invent an ASH value."
                    ),
                ]
            )
        if result.get("definition"):
            lines.append(f"Definition: {result['definition']}")
        if result.get("owner"):
            lines.append(f"Owner: {result['owner']}")
        return "\n".join(lines)

    def _format_kpi_knowledge_lookup(self, metric: str, query: str) -> str:
        metadata = self.data.get_kpi_metadata(metric)
        relationships = self.data.get_kpi_relationships(metric)
        playbook = self.data.get_playbook(metric)
        source_summary = self._kpi_knowledge_source_summary(metric)
        formula, formula_source, formula_lookup_status = self._formula_with_fallback(
            metric,
            query=query,
        )
        direct_answer = self._kpi_knowledge_direct_answer(metric, query, metadata, formula)

        details = [
            direct_answer,
            "",
            f"Knowledge-base details for {metric}:",
            f"- Owner: {metadata.get('KPI_Owner_Role', 'Not configured')}",
            f"- Function owner: {metadata.get('Function_Owner', 'Not configured')}",
            f"- Business question: {metadata.get('Business_Question', 'Not configured')}",
            f"- Formula: {formula}",
            f"- Formula source: {formula_source}",
            f"- Primary driver: {metadata.get('Primary_Driver_KPI', 'Not configured')}",
            f"- Secondary driver: {metadata.get('Secondary_Driver_KPI', 'Not configured')}",
            "",
            "Investigation steps:",
            self._format_list(
                [
                    metadata.get("Investigation_Step_1", ""),
                    metadata.get("Investigation_Step_2", ""),
                    metadata.get("Investigation_Step_3", ""),
                    metadata.get("Investigation_Step_4", ""),
                ]
            ),
        ]

        if not relationships.empty:
            details.extend(["", "Mapped relationships:"])
            for _, row in relationships.head(5).iterrows():
                child = row.get("Child_KPI", "Unknown driver")
                relationship = (
                    row.get("Relationship_Type") or row.get("Relationship") or "driver"
                )
                weight = row.get("Driver_Weight") or row.get("Weight") or "Unweighted"
                details.append(f"- {child}: {relationship}, {weight}")

        if not playbook.empty:
            details.extend(["", "Investigation playbook scenarios:"])
            for _, row in playbook.head(3).iterrows():
                scenario = row.get("Scenario", "Scenario")
                threshold = row.get("Threshold", "No threshold")
                severity = row.get("Severity", "No severity")
                action = row.get("Recommended_Action") or row.get("Action") or ""
                line = f"- {scenario}: {threshold}, {severity}"
                if action:
                    line += f". Action: {action}"
                details.append(line)

        if formula_lookup_status:
            details.extend(["", f"Dataverse formula fallback: {formula_lookup_status}"])

        details.extend(["", f"Knowledge source: {source_summary}"])
        return "\n".join(details)

    def _kpi_knowledge_direct_answer(
        self,
        metric: str,
        query: str,
        metadata: dict,
        formula: str,
    ) -> str:
        normalized = self.data.normalize_lookup_text(query)
        if "formula" in normalized or "calculate" in normalized or "calculation" in normalized:
            return f"Formula for {metric}: {formula}"
        if "owner" in normalized:
            return (
                f"Owner for {metric}: "
                f"{metadata.get('KPI_Owner_Role', 'Not configured')}"
            )
        if "business question" in normalized:
            return (
                f"Business question for {metric}: "
                f"{metadata.get('Business_Question', 'Not configured')}"
            )
        if "driver" in normalized or "relationship" in normalized or "related" in normalized:
            primary = metadata.get("Primary_Driver_KPI", "Not configured")
            secondary = metadata.get("Secondary_Driver_KPI", "Not configured")
            return f"Main drivers for {metric}: primary = {primary}; secondary = {secondary}"
        if "action" in normalized or "recommendation" in normalized:
            return (
                f"Recommended action for {metric}: "
                f"{metadata.get('Recommended_Action', 'Not configured')}"
            )
        if any(term in normalized for term in ["what is", "describe", "overview", "tell me about"]):
            business_question = metadata.get("Business_Question", "Not configured")
            return (
                f"{metric} is a KPI used to answer: {business_question}"
                if business_question != "Not configured" and business_question
                else f"{metric}: definition not configured in the knowledge base."
            )
        return f"Answer for {metric}: see the key knowledge-base fields below."

    def _kpi_knowledge_source_summary(self, metric: str) -> str:
        if self.knowledge_store is None:
            return "loaded Excel knowledge base"

        results = self.knowledge_store.search_kpi(metric, metric, limit=5)
        if not results:
            return "loaded Excel knowledge base"

        source_labels = []
        for result in results:
            metadata = result.get("metadata", {})
            sheet = str(metadata.get("sheet", "")).strip()
            if not sheet:
                continue
            label = self._readable_knowledge_sheet_name(sheet)
            if label not in source_labels:
                source_labels.append(label)

        if not source_labels:
            return "loaded Excel knowledge base"
        return f"{metric} records from " + ", ".join(source_labels)

    @staticmethod
    def _readable_knowledge_sheet_name(sheet_name: str) -> str:
        normalized = sheet_name.lower()
        if "knowledge_map" in normalized:
            return "KPI knowledge map"
        if "relationship_map" in normalized:
            return "relationship map"
        if "formula_definition" in normalized:
            return "formula definition"
        if "investigation_playbook" in normalized:
            return "investigation playbook"
        if sheet_name == "kpi_catalog":
            return "KPI catalog"
        return sheet_name

    def _format_root_cause(self, metric: str, bu: str | None = None) -> str:
        analysis = self.analytics.root_cause_analysis(metric, bu=bu)
        if "error" in analysis:
            return analysis["error"]

        scope = f" in {bu}" if bu else ""
        quantitative_drivers = []
        qualitative_drivers = []

        for driver in analysis.get("primary_drivers", []):
            relationship = driver.get("relationship") or "Relationship"
            weight = driver.get("weight") or "Unweighted"
            if driver.get("available_in_dataset"):
                quantitative_drivers.append(
                    f"- {driver['driver']} ({relationship}, {weight}): "
                    f"{driver['change_pct']:+.1f}%"
                )
            else:
                qualitative_drivers.append(
                    f"- {driver['driver']} ({relationship}, {weight}): "
                    "defined in the knowledge base but not available as a dataset column"
                )

        if not quantitative_drivers:
            quantitative_drivers.append(
                "- No quantitative driver columns were available for this KPI."
            )

        metadata = analysis.get("metadata", {})
        investigations = analysis.get("recommended_investigations") or [
            metadata.get("Investigation_Step_1", ""),
            metadata.get("Investigation_Step_2", ""),
            metadata.get("Investigation_Step_3", ""),
        ]
        actions = analysis.get("recommended_actions") or [
            metadata.get("Recommended_Action", "")
        ]

        direction_sentence = self._root_cause_direction_sentence(
            metric,
            analysis["variance_pct"],
            analysis["trend"],
        )

        return f"""
Root Cause Analysis: {metric}{scope}
- Current value: {self._format_metric_value(metric, analysis["current_value"])}
- Previous value: {self._format_metric_value(metric, analysis["previous_value"])}
- Change: {analysis["variance_pct"]:+.1f}%
- Severity: {str(analysis["severity"]).upper()}
- Operational direction: {self._operational_direction(metric, analysis["variance_pct"])}

Executive readout:
{direction_sentence}

Quantitative drivers from knowledge base:
{chr(10).join(quantitative_drivers)}

Knowledge-base qualitative drivers:
{chr(10).join(qualitative_drivers) if qualitative_drivers else "- None"}

Recommended investigations:
{self._format_list(investigations)}

Recommended actions:
{self._format_list(actions)}
""".strip()

    def _format_doctor_kpi_justification(
        self,
        doctor_name: str,
        metric: str,
        bu: str | None = None,
    ) -> str:
        if not bu and self._doctor_has_multiple_bus(doctor_name):
            return self._format_all_doctor_bu_kpi_justifications(doctor_name, metric)

        df = self._doctor_scoped_df(doctor_name, bu=bu).copy()
        if df.empty:
            scope = f" in {bu}" if bu else ""
            return f"No data found for Dr. {doctor_name}{scope}."
        if metric not in df.columns:
            return self._unknown_metric_message(metric)

        latest_date = df["Date"].max()
        date_values = sorted(df["Date"].dropna().unique())
        previous_date = date_values[-2] if len(date_values) >= 2 else latest_date
        current = df[df["Date"] == latest_date]
        previous = df[df["Date"] == previous_date]

        current_value = self.analytics._aggregate_metric(current, metric)
        previous_value = self.analytics._aggregate_metric(previous, metric)
        change_pct = (
            (current_value - previous_value) / previous_value * 100
            if previous_value
            else 0
        )

        peer_df = self.data.df.copy()
        if bu:
            peer_df = peer_df[peer_df["BU"] == bu]
        peer_current = peer_df[
            (peer_df["Date"] == latest_date)
            & ~self._doctor_name_mask(peer_df, doctor_name)
        ]
        if peer_current.empty:
            peer_current = current
        peer_value = self._peer_doctor_metric_average(peer_current, metric)

        doctor_revenue = (
            float(df["Total Revenue"].sum()) if "Total Revenue" in df else 0
        )
        target_revenue = (
            float(df["Target Revenue"].sum()) if "Target Revenue" in df else 0
        )
        achievement = (doctor_revenue / target_revenue * 100) if target_revenue else 0
        no_show = df["No-Show %"].mean() * 100 if "No-Show %" in df else None
        retention = (
            df["Patient Retention %"].mean() * 100
            if "Patient Retention %" in df
            else None
        )

        root_cause = self.analytics.root_cause_analysis(
            metric, doctor=doctor_name, bu=bu
        )
        drivers = (
            root_cause.get("primary_drivers", []) if "error" not in root_cause else []
        )
        driver_lines = []
        for driver in drivers:
            if driver.get("available_in_dataset"):
                driver_lines.append(
                    f"- {driver['driver']} ({driver.get('relationship', 'Driver')}, "
                    f"{driver.get('weight', 'Unweighted')}): {driver['change_pct']:+.1f}%"
                )
            else:
                driver_lines.append(
                    f"- {driver['driver']}: defined in the knowledge base but not available quantitatively."
                )
        if not driver_lines:
            driver_lines.append(
                "- No direct quantitative drivers are mapped for this KPI."
            )

        metadata = self.data.get_kpi_metadata(metric)
        scope = f" in {bu}" if bu else ""
        doctor_label = self._doctor_identity_label(doctor_name, bu)
        peer_gap = current_value - peer_value
        peer_gap_text = (
            f"{self._format_metric_value(metric, abs(peer_gap))} above peer average"
            if peer_gap >= 0
            else f"{self._format_metric_value(metric, abs(peer_gap))} below peer average"
        )

        recommendations = [
            metadata.get("Recommended_Action", ""),
            "Review the doctor's KPI movement by month and compare with peers in the same BU.",
            "Validate whether operational drivers such as no-show, retention, leakage, and case mix explain the variance.",
        ]

        no_show_line = (
            f"- Average no-show: {no_show:.1f}%"
            if no_show is not None
            else "- Average no-show: not available"
        )
        retention_line = (
            f"- Average retention: {retention:.1f}%"
            if retention is not None
            else "- Average retention: not available"
        )

        return f"""
Justification: Dr. {doctor_label}'s {metric} performance{scope}

Executive readout:
Dr. {doctor_label}'s current {metric} is {self._format_metric_value(metric, current_value)}, changing {change_pct:+.1f}% versus the previous period. Against the selected peer group, this is {peer_gap_text}. Revenue achievement is {achievement:.1f}%, so the KPI should be interpreted alongside volume, retention, no-show, and leakage behavior rather than in isolation.

Evidence:
- Current period: {self._format_metric_value(metric, current_value)}
- Previous period: {self._format_metric_value(metric, previous_value)}
- Peer average: {self._format_metric_value(metric, peer_value)}
- Total revenue: ${doctor_revenue:,.0f}
- Target revenue: ${target_revenue:,.0f}
- Revenue achievement: {achievement:.1f}%
{no_show_line}
{retention_line}

Knowledge-base drivers to check:
{chr(10).join(driver_lines)}

Recommended actions:
{self._format_list(recommendations)}
""".strip()

    def _format_kpi_trend(
        self,
        metric: str,
        months: int = 6,
        bu: str | None = None,
        year: int | None = None,
        month: int | None = None,
    ) -> str:
        df = self._scoped_df(bu=bu, year=year, month=month)
        if metric not in df.columns:
            return self._unknown_metric_message(metric)
        if df.empty:
            return f"No trend data for {metric}{self._scope_text(bu=bu, year=year, month=month)}"

        aggregate = "sum" if self._is_additive_metric(metric) else "mean"
        trend_data = (
            df.dropna(subset=["YearMonth", metric])
            .groupby("YearMonth", as_index=False)[metric]
            .agg(aggregate)
            .sort_values("YearMonth")
        )
        if trend_data.empty:
            return f"No trend data for {metric}{self._scope_text(bu=bu, year=year, month=month)}"

        if not year and not month:
            months = max(1, min(int(months or 12), 24))
            trend_data = trend_data.tail(months)

        first = float(trend_data[metric].iloc[0])
        last = float(trend_data[metric].iloc[-1])
        change = ((last - first) / first * 100) if first else 0
        direction = (
            "Increasing" if change > 5 else "Decreasing" if change < -5 else "Stable"
        )
        scope = self._scope_text(bu=bu, year=year, month=month)

        lines = [
            f"{direction} monthly trend for {metric}{scope} (change: {change:+.1f}%)",
            f"Aggregation: {aggregate}",
            "",
        ]
        for _, row in trend_data.iterrows():
            lines.append(
                f"{row['YearMonth']}: {self._format_metric_value(metric, row[metric])}"
            )
        return "\n".join(lines)

    def _format_metric_overview(
        self,
        metric: str,
        bu: str | None = None,
        year: int | None = None,
        month: int | None = None,
    ) -> str:
        df = self._scoped_df(bu=bu, year=year, month=month)
        if metric not in df.columns or df.empty:
            return f"No data for metric: {metric}"

        current, previous = self.analytics._current_previous_periods(df)
        current_value = self.analytics._aggregate_metric(current, metric)
        previous_value = self.analytics._aggregate_metric(previous, metric)
        change_pct = (
            (current_value - previous_value) / previous_value * 100
            if previous_value
            else 0
        )
        metadata = self.data.get_kpi_metadata(metric)
        scope = self._scope_text(bu=bu, year=year, month=month)

        lines = [
            f"KPI Overview: {metric}{scope}",
            "",
            "Executive readout:",
            (
                f"The current value is {self._format_metric_value(metric, current_value)}, "
                f"versus {self._format_metric_value(metric, previous_value)} in the previous period "
                f"({change_pct:+.1f}%). Operational direction: "
                f"{self._operational_direction(metric, change_pct)}."
            ),
            "",
            "Knowledge-base context:",
            f"- Business question: {metadata.get('Business_Question', 'Not configured')}",
            f"- Formula: {metadata.get('Formula_Logic', metadata.get('Financial_Impact_Formula', 'Not configured'))}",
            f"- Owner: {metadata.get('KPI_Owner_Role', 'Not configured')}",
            "",
            "Recommended action:",
            f"- {metadata.get('Recommended_Action', 'Review by BU, doctor, and month to identify concentration and drivers.')}",
        ]
        return "\n".join(lines)

    def _format_kpi_value(
        self,
        metric: str,
        bu: str | None = None,
        year: int | None = None,
        month: int | None = None,
    ) -> str:
        df = self._scoped_df(bu=bu, year=year, month=month)
        if metric not in df.columns or df.empty:
            return f"No data for metric: {metric}{self._scope_text(bu=bu, year=year, month=month)}"

        if not any([bu, year, month]) and "Date" in df.columns:
            df = df[df["Date"] == df["Date"].max()]

        metric_lower = metric.lower()
        scope = self._scope_text(bu=bu, year=year, month=month)

        if metric == "Revenue_Achievement_%":
            actual = self._sum_metric(df, "Total Revenue")
            target = self._sum_metric(df, "Target Revenue")
            value = (actual / target * 100) if target else 0
            return "\n".join(
                [
                    f"Revenue Achievement %{scope}: {value:.1f}%",
                    f"- Actual revenue: {self._format_metric_value('Total Revenue', actual)}",
                    f"- Target revenue: {self._format_metric_value('Target Revenue', target)}",
                    "- Formula used: Total Revenue / Target Revenue x 100",
                ]
            )

        if metric == "Cases_Achievement_%":
            actual = self._sum_metric(df, "No. Cases")
            target = self._sum_metric(df, "Target No. cases")
            value = (actual / target * 100) if target else 0
            return "\n".join(
                [
                    f"Cases Achievement %{scope}: {value:.1f}%",
                    f"- Actual cases: {actual:,.0f}",
                    f"- Target cases: {target:,.0f}",
                    "- Formula used: No. Cases / Target No. cases x 100",
                ]
            )

        aggregate = "sum" if self._is_additive_metric(metric) else "mean"
        value = self._sum_metric(df, metric) if aggregate == "sum" else self._mean_metric(df, metric)
        return "\n".join(
            [
                f"{metric}{scope}: {self._format_metric_value(metric, value)}",
                f"- Aggregation used: {aggregate}",
                (
                    "- Note: this is a percentage KPI averaged across the selected rows."
                    if "%" in metric or "cr%" in metric_lower
                    else "- Note: this is calculated from the selected dataset rows."
                ),
            ]
        )

    def _format_doctor_comparison(
        self,
        metric: str,
        bu: str | None = None,
        year: int | None = None,
        month: int | None = None,
    ) -> str:
        df = self._scoped_df(bu=bu, year=year, month=month)
        if metric not in df.columns or df.empty:
            return f"No data for metric: {metric}"

        aggregate = "sum" if self._is_additive_metric(metric) else "mean"
        group_columns = ["Doctor Name"] if bu else ["BU", "Doctor Name"]
        if aggregate == "sum":
            ranking = df.groupby(group_columns)[metric].sum().reset_index()
        else:
            ranking = df.groupby(group_columns)[metric].mean().reset_index()
        ranking = (
            ranking.sort_values(metric, ascending=False).head(5).reset_index(drop=True)
        )
        ranking["rank"] = ranking.index + 1

        if ranking.empty:
            return f"No data for metric: {metric}"

        scope = self._scope_text(bu=bu, year=year, month=month)
        lines = [f"Top doctors by {metric}{scope}:"]
        for _, row in ranking.iterrows():
            doctor_label = self._doctor_identity_label(row["Doctor Name"], row.get("BU"))
            lines.append(
                f"{row['rank']}. {doctor_label}: "
                f"{self._format_metric_value(metric, row[metric])}"
            )
        return "\n".join(lines)

    def _format_bu_comparison(
        self,
        metric: str,
        year: int | None = None,
        month: int | None = None,
        bus: list[str] | None = None,
    ) -> str:
        df = self._scoped_df(year=year, month=month)
        if bus:
            df = df[df["BU"].isin(bus)]
        if metric not in df.columns or df.empty:
            return f"No data for metric: {metric}"

        aggregate = "sum" if self._is_additive_metric(metric) else "mean"
        if aggregate == "sum":
            ranking = df.groupby("BU")[metric].sum().reset_index()
        else:
            ranking = df.groupby("BU")[metric].mean().reset_index()

        ranking = ranking.sort_values(metric, ascending=False).reset_index(drop=True)
        scope = self._scope_text(year=year, month=month)
        best = ranking.iloc[0]
        worst = ranking.iloc[-1]

        difference_label = (
            "BU revenue spread" if metric == "Total Revenue" else "BU difference"
        )

        lines = [
            f"BU comparison: {metric}{scope}",
            "",
            "Executive readout:",
            (
                f"{best['BU']} is higher at "
                f"{self._format_metric_value(metric, best[metric])}; "
                f"{worst['BU']} is lower at "
                f"{self._format_metric_value(metric, worst[metric])}. "
                f"The {difference_label} is "
                f"{self._format_metric_value(metric, best[metric] - worst[metric])}. "
                f"This comparison uses {aggregate} aggregation based on the metric type."
            ),
            "",
            "BU ranking:",
        ]
        for index, (_, row) in enumerate(ranking.iterrows(), start=1):
            lines.append(
                f"{index}. {row['BU']}: {self._format_metric_value(metric, row[metric])}"
            )

        lines.extend(
            [
                "",
                "Recommended actions:",
                "- Compare the highest and lowest BUs by doctor, specialty, and month to identify where the spread is concentrated.",
                "- Use the KPI relationship map to check the strongest operational drivers before deciding on corrective actions.",
            ]
        )
        return "\n".join(lines)

    def _format_doctors_by_threshold(
        self,
        metric: str,
        operator: str,
        threshold: float,
        bu: str | None = None,
        year: int | None = None,
        month: int | None = None,
    ) -> str:
        df = self._scoped_df(bu=bu, year=year, month=month)
        if metric not in df.columns or df.empty:
            return f"No data for metric: {metric}"
        if ("%" in metric or "cr%" in metric.lower()) and threshold > 1:
            threshold = threshold / 100

        group_columns = ["Doctor Name"] if bu else ["BU", "Doctor Name"]
        if self._is_additive_metric(metric):
            grouped = df.groupby(group_columns)[metric].sum().reset_index()
        else:
            grouped = df.groupby(group_columns)[metric].mean().reset_index()

        if operator == "above":
            result = grouped[grouped[metric] > threshold].sort_values(
                metric, ascending=False
            )
            phrase = "above"
        else:
            result = grouped[grouped[metric] < threshold].sort_values(
                metric, ascending=True
            )
            phrase = "below"

        scope = self._scope_text(bu=bu, year=year, month=month)
        threshold_text = self._format_metric_value(metric, threshold)
        if result.empty:
            return f"No doctors have {metric} {phrase} {threshold_text}{scope}."

        lines = [
            f"Doctors with {metric} {phrase} {threshold_text}{scope}:",
            "",
            "Executive readout:",
            f"{len(result)} doctor(s) are {phrase} the requested threshold. "
            "These should be prioritized for operational review if the KPI is unfavorable at that level.",
            "",
            "Doctor list:",
        ]
        for index, (_, row) in enumerate(result.iterrows(), start=1):
            doctor_label = self._doctor_identity_label(row["Doctor Name"], row.get("BU"))
            lines.append(
                f"{index}. {doctor_label}: "
                f"{self._format_metric_value(metric, row[metric])}"
            )

        lines.extend(
            [
                "",
                "Recommended actions:",
                "- Review these doctors by month to confirm whether the issue is persistent or isolated.",
                "- Check related drivers from the knowledge base, especially booking quality, reminder compliance, retention, and missed opportunities where relevant.",
                "- Prioritize doctors furthest from the threshold for immediate operational follow-up.",
            ]
        )
        return "\n".join(lines)

    def _format_bu_summary(self, bu: str) -> str:
        df_bu = self.data.df[self.data.df["BU"] == bu]
        if df_bu.empty:
            return self._unknown_bu_message(bu)

        total_revenue = float(df_bu["Total Revenue"].sum())
        target_revenue = float(df_bu["Target Revenue"].sum())
        achievement = (total_revenue / target_revenue * 100) if target_revenue else 0

        return f"""
BU Summary: {bu}
- Total Revenue: ${total_revenue:,.0f}
- Target Revenue: ${target_revenue:,.0f}
- Achievement: {achievement:.1f}%
- Total Cases: {df_bu["No. Cases"].sum():,.0f}
- Avg No-Show: {df_bu["No-Show %"].mean() * 100:.1f}%
- Avg Retention: {df_bu["Patient Retention %"].mean() * 100:.1f}%
- Avg Service Leakage: {df_bu["Service Leakage %"].mean() * 100:.1f}%
""".strip()

    def _format_doctor_performance(
        self,
        doctor_name: str,
        bu: str | None = None,
    ) -> str:
        if not bu and self._doctor_has_multiple_bus(doctor_name):
            return self._format_all_doctor_bu_performance(doctor_name)

        df_doctor = self._doctor_scoped_df(doctor_name, bu=bu)
        if df_doctor.empty:
            return (
                f"Doctor '{doctor_name}' not found. Available doctors: "
                f"{', '.join(self.data.get_doctor_display_list()[:10])}"
            )

        total_revenue = float(df_doctor["Total Revenue"].sum())
        target_revenue = float(df_doctor["Target Revenue"].sum())
        achievement = (total_revenue / target_revenue * 100) if target_revenue else 0
        doctor_label = self._doctor_identity_label(doctor_name, bu)

        return f"""
Doctor: {doctor_label}
- Total Revenue: ${total_revenue:,.0f}
- Target Revenue: ${target_revenue:,.0f}
- Achievement: {achievement:.1f}%
- Cases: {df_doctor["No. Cases"].sum():,.0f}
- Avg No-Show: {df_doctor["No-Show %"].mean() * 100:.1f}%
- Avg Retention: {df_doctor["Patient Retention %"].mean() * 100:.1f}%
""".strip()

    def _format_all_doctor_bu_performance(self, doctor_name: str) -> str:
        bus = self.data.get_bus_for_doctor(doctor_name)
        if not bus:
            return (
                f"Doctor '{doctor_name}' not found. Available doctors: "
                f"{', '.join(self.data.get_doctor_display_list()[:10])}"
            )

        sections = [
            f"Doctor performance for all available Dr. {doctor_name} profiles:",
            "",
        ]
        for bu in bus:
            df_doctor = self._doctor_scoped_df(doctor_name, bu=bu)
            total_revenue = float(df_doctor["Total Revenue"].sum())
            target_revenue = float(df_doctor["Target Revenue"].sum())
            achievement = (
                total_revenue / target_revenue * 100 if target_revenue else 0
            )
            sections.extend(
                [
                    f"Dr. {self._doctor_identity_label(doctor_name, bu)}",
                    f"- Total Revenue: ${total_revenue:,.0f}",
                    f"- Target Revenue: ${target_revenue:,.0f}",
                    f"- Achievement: {achievement:.1f}%",
                    f"- Cases: {df_doctor['No. Cases'].sum():,.0f}",
                    f"- Avg No-Show: {df_doctor['No-Show %'].mean() * 100:.1f}%",
                    f"- Avg Retention: {df_doctor['Patient Retention %'].mean() * 100:.1f}%",
                    "",
                ]
            )
        return "\n".join(sections).strip()

    def _format_all_doctor_bu_kpi_justifications(
        self,
        doctor_name: str,
        metric: str,
    ) -> str:
        bus = self.data.get_bus_for_doctor(doctor_name)
        if not bus:
            return (
                f"Doctor '{doctor_name}' not found. Available doctors: "
                f"{', '.join(self.data.get_doctor_display_list()[:10])}"
            )

        sections = [
            f"{metric} justification for all available Dr. {doctor_name} profiles:",
            "",
        ]
        for bu in bus:
            sections.append(self._format_doctor_kpi_justification(doctor_name, metric, bu=bu))
            sections.append("")
        return "\n\n".join(section for section in sections if section).strip()

    def _format_all_doctor_bu_profiles(
        self,
        doctor_name: str,
        year: int | None = None,
        month: int | None = None,
    ) -> str:
        bus = self.data.get_bus_for_doctor(doctor_name)
        if not bus:
            return (
                f"Doctor '{doctor_name}' not found. Available doctors: "
                f"{', '.join(self.data.get_doctor_display_list()[:10])}"
            )

        sections = [
            f"KPI profile for all available Dr. {doctor_name} profiles:",
            "",
        ]
        for bu in bus:
            sections.append(
                self._format_compact_doctor_kpi_profile(
                    doctor_name,
                    bu=bu,
                    year=year,
                    month=month,
                )
            )
            sections.append("")
        return "\n\n".join(section for section in sections if section).strip()

    def _format_compact_doctor_kpi_profile(
        self,
        doctor_name: str,
        bu: str,
        year: int | None = None,
        month: int | None = None,
    ) -> str:
        df = self._scoped_df(bu=bu, year=year, month=month)
        df_doctor = self._doctor_filter(df, doctor_name).copy()
        if df_doctor.empty:
            return f"No data found for Dr. {self._doctor_identity_label(doctor_name, bu)}."

        total_revenue = float(df_doctor["Total Revenue"].sum())
        target_revenue = float(df_doctor["Target Revenue"].sum())
        revenue_achievement = total_revenue / target_revenue if target_revenue else 0
        total_cases = float(df_doctor["No. Cases"].sum())
        target_cases = float(df_doctor["Target No. cases"].sum())
        cases_achievement = total_cases / target_cases if target_cases else 0

        kpi_candidates = [
            "Revenue_Achievement_%",
            "Cases_Achievement_%",
            "Doctor PMS %",
            "No-Show %",
            "Patient Retention %",
            "Service Leakage %",
        ]
        peer_df = df[~df.index.isin(df_doctor.index)]
        if peer_df.empty:
            peer_df = df

        lines = [
            f"Dr. {self._doctor_identity_label(doctor_name, bu)}",
            (
                f"- Revenue: ${total_revenue:,.0f} / ${target_revenue:,.0f} "
                f"({revenue_achievement * 100:.1f}% achievement)"
            ),
            (
                f"- Cases: {total_cases:,.0f} / {target_cases:,.0f} "
                f"({cases_achievement * 100:.1f}% achievement)"
            ),
            "- KPI evidence:",
        ]

        for metric in [item for item in kpi_candidates if item in df_doctor.columns]:
            doctor_value = self.analytics._aggregate_metric(df_doctor, metric)
            peer_value = self._peer_doctor_metric_average(peer_df, metric)
            gap = doctor_value - peer_value
            gap_pct = (gap / peer_value * 100) if peer_value else 0
            direction = self._doctor_gap_direction(metric, gap)
            lines.append(
                f"  - {metric}: {self._format_metric_value(metric, doctor_value)} "
                f"vs BU peer {self._format_metric_value(metric, peer_value)} "
                f"({gap_pct:+.1f}% gap, {direction})"
            )

        return "\n".join(lines)

    def _search_kpi_knowledge(self, query: str, metric: str | None = None) -> str:
        normalized_query = self.data.normalize_lookup_text(query)
        asks_formula = any(
            term in normalized_query
            for term in ["formula", "calculate", "calculation"]
        )

        if self.knowledge_store is None:
            if asks_formula:
                return self._format_dataverse_formula_lookup(metric or query)
            return (
                "The Chroma knowledge store is not available. The agent can still "
                "use the loaded Excel knowledge base through its analytics tools."
            )

        if metric:
            results = self.knowledge_store.search_kpi(metric, query, limit=5)
        else:
            results = self.knowledge_store.search(query, limit=5)
        if not results:
            if asks_formula:
                return self._format_dataverse_formula_lookup(metric or query)
            return f"No Chroma knowledge-base results found for: {query}"

        lines = [f"Chroma knowledge-base results for: {query}"]
        for index, result in enumerate(results, start=1):
            metadata = result.get("metadata", {})
            source = metadata.get("source", "knowledge_base")
            sheet = metadata.get("sheet", "")
            kpi = metadata.get("kpi", "")
            document = self._compact_text(result.get("document", ""), max_length=650)

            heading_parts = [f"{index}. {source}"]
            if kpi:
                heading_parts.append(str(kpi))
            if sheet:
                heading_parts.append(f"sheet={sheet}")

            lines.append(" | ".join(heading_parts))
            lines.append(document)

        return "\n".join(lines)

    def _format_doctor_kpi_profile(
        self,
        doctor_name: str,
        bu: str | None = None,
        year: int | None = None,
        month: int | None = None,
    ) -> str:
        if not bu and self._doctor_has_multiple_bus(doctor_name):
            return self._format_all_doctor_bu_profiles(
                doctor_name,
                year=year,
                month=month,
            )

        df = self._scoped_df(bu=bu, year=year, month=month)
        df_doctor = self._doctor_filter(df, doctor_name).copy()
        if df_doctor.empty:
            scope = self._scope_text(bu=bu, year=year, month=month)
            return (
                f"Doctor '{doctor_name}' not found{scope}. Available doctors: "
                f"{', '.join(self.data.get_doctor_display_list()[:10])}"
            )

        total_revenue = float(df_doctor["Total Revenue"].sum())
        target_revenue = float(df_doctor["Target Revenue"].sum())
        revenue_achievement = total_revenue / target_revenue if target_revenue else 0
        total_cases = float(df_doctor["No. Cases"].sum())
        target_cases = float(df_doctor["Target No. cases"].sum())
        cases_achievement = total_cases / target_cases if target_cases else 0

        kpi_candidates = [
            "Revenue_Achievement_%",
            "Cases_Achievement_%",
            "Doctor PMS %",
            "No-Show %",
            "Patient Retention %",
            "Service Leakage %",
            "Cross Referral %",
            "Patient Acquisition %",
            "Actual COE Compliance %",
            "Digital Actual CR%",
            "No. Missed Opportunity",
            "No. Cancelled Clinics",
        ]
        kpis = [metric for metric in kpi_candidates if metric in df_doctor.columns]

        scope = self._scope_text(bu=bu, year=year, month=month)
        doctor_label = self._doctor_identity_label(doctor_name, bu)
        lines = [
            f"Doctor KPI Performance and Justification: Dr. {doctor_label}{scope}",
            "",
            "Executive readout:",
            (
                f"Dr. {doctor_label} generated ${total_revenue:,.0f} against a "
                f"${target_revenue:,.0f} target ({revenue_achievement * 100:.1f}% achievement) "
                f"across {total_cases:,.0f} cases ({cases_achievement * 100:.1f}% of case target). "
                "The KPI justification below compares the doctor's performance with the selected peer group "
                "and explains the likely operational drivers from the knowledge base."
            ),
            "",
            "KPI evidence:",
        ]

        peer_df = df[~df.index.isin(df_doctor.index)]
        if peer_df.empty:
            peer_df = df

        for metric in kpis:
            doctor_value = self.analytics._aggregate_metric(df_doctor, metric)
            peer_value = self._peer_doctor_metric_average(peer_df, metric)
            gap = doctor_value - peer_value
            gap_pct = (gap / peer_value * 100) if peer_value else 0
            direction = self._doctor_gap_direction(metric, gap)
            drivers = self._kpi_driver_summary(metric)

            lines.append(
                f"- {metric}: {self._format_metric_value(metric, doctor_value)} "
                f"vs peer {self._format_metric_value(metric, peer_value)} "
                f"({gap_pct:+.1f}% gap, {direction}). {drivers}"
            )

        recommendations = self._doctor_profile_recommendations(df_doctor, peer_df)
        lines.extend(
            [
                "",
                "Priority recommendations:",
                self._format_list(recommendations),
            ]
        )
        return "\n".join(lines)

    @staticmethod
    def _period_label(df) -> str:
        if df.empty or "YearMonth" not in df.columns:
            return "selected period"
        value = df["YearMonth"].dropna()
        return str(value.iloc[0]) if not value.empty else "selected period"

    @staticmethod
    def _sum_metric(df, metric: str) -> float:
        if df.empty or metric not in df.columns:
            return 0.0
        return float(df[metric].dropna().sum())

    @staticmethod
    def _mean_metric(df, metric: str) -> float:
        if df.empty or metric not in df.columns:
            return 0.0
        values = df[metric].dropna()
        return float(values.mean()) if not values.empty else 0.0

    @staticmethod
    def _pct_change(current: float, previous: float) -> float:
        return ((current - previous) / previous * 100) if previous else 0.0

    def _format_metric_value(self, metric: str, value: float | int | None) -> str:
        if value is None:
            return "N/A"

        metric_lower = metric.lower()
        numeric_value = float(value)
        if "%" in metric or "cr%" in metric_lower:
            display_value = (
                numeric_value * 100 if abs(numeric_value) <= 1.5 else numeric_value
            )
            return f"{display_value:.1f}%"
        if any(word in metric_lower for word in ["revenue", "losses", "charge"]):
            return f"${numeric_value:,.0f}"
        if any(
            word in metric_lower
            for word in [
                "cases",
                "booking",
                "services",
                "opportunity",
                "clinics",
                "visits",
            ]
        ):
            return f"{numeric_value:,.0f}"
        return f"{numeric_value:,.2f}"

    def _root_cause_direction_sentence(
        self,
        metric: str,
        variance_pct: float,
        trend: str,
    ) -> str:
        metric_lower = metric.lower()
        increase_is_bad = any(
            term in metric_lower
            for term in ["leakage", "loss", "no-show", "cancelled", "missed"]
        )
        if abs(variance_pct) < 5:
            return (
                f"{metric} is broadly stable, so the priority is to identify pockets "
                "of underperformance by doctor, BU, and month rather than treating it "
                "as a system-wide movement."
            )

        if variance_pct > 0 and increase_is_bad:
            return (
                f"{metric} increased, which is unfavorable for this KPI. The main "
                "focus should be on the high-weight drivers below and where they are "
                "concentrated operationally."
            )
        if variance_pct < 0 and increase_is_bad:
            return (
                f"{metric} decreased, which is favorable for this KPI. Still, the "
                "drivers below should be monitored to confirm the improvement is "
                "repeatable."
            )
        if trend == "improving":
            return (
                f"{metric} is improving. The next step is to protect the drivers that "
                "created the gain and check whether the improvement is consistent "
                "across doctors and BUs."
            )
        return (
            f"{metric} is declining. The investigation should start with the strongest "
            "mapped drivers and then move into doctor-level and BU-level segmentation."
        )

    def _operational_direction(self, metric: str, variance_pct: float) -> str:
        if abs(variance_pct) < 5:
            return "stable"

        metric_lower = metric.lower()
        increase_is_bad = any(
            term in metric_lower
            for term in ["leakage", "loss", "no-show", "cancelled", "missed"]
        )
        if increase_is_bad:
            return "worsening" if variance_pct > 0 else "improving"
        return "improving" if variance_pct > 0 else "worsening"

    def _doctor_gap_direction(self, metric: str, gap: float) -> str:
        if abs(gap) < 0.000001:
            return "in line with peers"

        metric_lower = metric.lower()
        higher_is_bad = any(
            term in metric_lower
            for term in ["leakage", "loss", "no-show", "cancelled", "missed"]
        )
        if higher_is_bad:
            return "unfavorable" if gap > 0 else "favorable"
        return "favorable" if gap > 0 else "unfavorable"

    def _peer_doctor_metric_average(self, peer_df, metric: str) -> float:
        if peer_df.empty or metric not in peer_df.columns:
            return 0

        group_columns = (
            ["BU", "Doctor Name"]
            if "BU" in peer_df.columns
            else ["Doctor Name"]
        )
        if self._is_additive_metric(metric):
            per_doctor = peer_df.groupby(group_columns)[metric].sum()
        else:
            per_doctor = peer_df.groupby(group_columns)[metric].mean()
        return float(per_doctor.mean()) if not per_doctor.empty else 0

    def _kpi_driver_summary(self, metric: str) -> str:
        relationships = self.data.get_kpi_relationships(metric)
        if relationships.empty:
            metadata = self.data.get_kpi_metadata(metric)
            action = metadata.get("Recommended_Action")
            if action:
                return f"Recommended action: {action}"
            return "No specific knowledge-base driver is mapped, so validate by BU, month, and doctor mix."

        driver_names = []
        for _, row in relationships.head(3).iterrows():
            driver = row.get("Child_KPI")
            relationship = (
                row.get("Relationship_Type") or row.get("Relationship") or "driver"
            )
            weight = row.get("Driver_Weight") or row.get("Weight") or ""
            if driver:
                label = f"{driver} ({relationship}"
                if weight:
                    label += f", {weight}"
                label += ")"
                driver_names.append(label)

        if not driver_names:
            return "Knowledge-base relationship exists, but no readable driver names were configured."
        return f"Main drivers to check: {', '.join(driver_names)}."

    def _doctor_profile_recommendations(self, df_doctor, peer_df) -> list[str]:
        recommendations = []

        if "Revenue_Achievement_%" in df_doctor and "Revenue_Achievement_%" in peer_df:
            doctor_revenue = self.analytics._aggregate_metric(
                df_doctor, "Revenue_Achievement_%"
            )
            peer_revenue = self._peer_doctor_metric_average(
                peer_df, "Revenue_Achievement_%"
            )
            if doctor_revenue < peer_revenue:
                recommendations.append(
                    "Prioritize revenue achievement: review case volume, charge per case, booking conversion, and missed opportunities."
                )

        if "No-Show %" in df_doctor and "No-Show %" in peer_df:
            doctor_no_show = self.analytics._aggregate_metric(df_doctor, "No-Show %")
            peer_no_show = self._peer_doctor_metric_average(peer_df, "No-Show %")
            if doctor_no_show > peer_no_show:
                recommendations.append(
                    "Reduce no-show exposure through reminder compliance, confirmation workflows, and doctor-level slot review."
                )

        if "Patient Retention %" in df_doctor and "Patient Retention %" in peer_df:
            doctor_retention = self.analytics._aggregate_metric(
                df_doctor, "Patient Retention %"
            )
            peer_retention = self._peer_doctor_metric_average(
                peer_df, "Patient Retention %"
            )
            if doctor_retention < peer_retention:
                recommendations.append(
                    "Improve retention by reviewing follow-up discipline, care continuity, and post-visit communication."
                )

        if "Service Leakage %" in df_doctor and "Service Leakage %" in peer_df:
            doctor_leakage = self.analytics._aggregate_metric(
                df_doctor, "Service Leakage %"
            )
            peer_leakage = self._peer_doctor_metric_average(
                peer_df, "Service Leakage %"
            )
            if doctor_leakage > peer_leakage:
                recommendations.append(
                    "Investigate service leakage by specialty, missed opportunity, workflow compliance, and follow-up visit behavior."
                )

        if not recommendations:
            recommendations.append(
                "Maintain current performance controls and monitor monthly movement against peer averages."
            )
        recommendations.append(
            "Use the strongest KPI gaps above as the order of investigation instead of treating all KPIs equally."
        )
        return recommendations

    def _resolve_optional_bu(self, bu_name: str) -> str | None:
        if not bu_name or not str(bu_name).strip():
            return None
        return self.data.resolve_bu(bu_name)

    def _scoped_df(
        self,
        bu: str | None = None,
        year: int | None = None,
        month: int | None = None,
    ):
        df = self.data.df.copy()
        if bu:
            df = df[df["BU"] == bu]
        if year and "Year" in df.columns:
            df = df[df["Year"].astype("Int64") == year]
        if month and "Month_Num" in df.columns:
            df = df[df["Month_Num"].astype("Int64") == month]
        return df

    def _scope_text(
        self,
        bu: str | None = None,
        year: int | None = None,
        month: int | None = None,
    ) -> str:
        parts = []
        if bu:
            parts.append(str(bu))
        if month:
            parts.append(self._month_name(month))
        if year:
            parts.append(str(year))
        return f" in {' / '.join(parts)}" if parts else ""

    def _is_additive_metric(self, metric: str) -> bool:
        metric_lower = metric.lower()
        if "%" in metric or "cr%" in metric_lower or "achievement" in metric_lower:
            return False
        return any(
            word in metric_lower
            for word in [
                "revenue",
                "losses",
                "cases",
                "booking",
                "services",
                "opportunity",
                "clinics",
                "visits",
            ]
        )

    def _extract_year_from_text(self, text: str) -> int | None:
        match = re.search(r"\b(20\d{2})\b", text)
        if not match:
            return None
        return int(match.group(1))

    def _parse_year(self, value: str | int | None) -> int | None:
        if value in (None, ""):
            return None
        match = re.search(r"\b(20\d{2})\b", str(value))
        return int(match.group(1)) if match else None

    def _parse_month(self, value: str | int | None) -> int | None:
        if value in (None, ""):
            return None
        value_text = str(value).strip()
        if value_text.isdigit():
            month = int(value_text)
            return month if 1 <= month <= 12 else None
        return self._extract_month_from_text(value_text)

    def _extract_month_from_text(self, text: str) -> int | None:
        month_lookup = {
            "january": 1,
            "jan": 1,
            "february": 2,
            "feb": 2,
            "march": 3,
            "mar": 3,
            "april": 4,
            "apr": 4,
            "may": 5,
            "june": 6,
            "jun": 6,
            "july": 7,
            "jul": 7,
            "august": 8,
            "aug": 8,
            "september": 9,
            "sep": 9,
            "sept": 9,
            "october": 10,
            "oct": 10,
            "november": 11,
            "nov": 11,
            "december": 12,
            "dec": 12,
        }
        normalized = self.data.normalize_lookup_text(text)
        for word in normalized.split():
            if word in month_lookup:
                return month_lookup[word]
        return None

    def _extract_threshold_filter(self, text: str) -> dict | None:
        """Extract threshold filters such as above 20% or below 10."""
        patterns = [
            (
                r"\b(above|over|greater than|more than|higher than)\s+(\d+(?:\.\d+)?)\s*(%)?",
                "above",
            ),
            (r"\b(below|under|less than|lower than)\s+(\d+(?:\.\d+)?)\s*(%)?", "below"),
        ]
        for pattern, operator in patterns:
            match = re.search(pattern, text, flags=re.IGNORECASE)
            if not match:
                continue

            value = float(match.group(2))
            if match.group(3) == "%":
                value = value / 100
            return {"operator": operator, "threshold": value}

        return None

    def _extract_bu_from_text(self, text: str) -> str | None:
        bus = self._extract_bus_from_text(text)
        return bus[0] if bus else None

    def _extract_bus_from_text(self, text: str) -> list[str]:
        normalized = f" {self.data.normalize_lookup_text(text)} "
        matches = []
        for bu in self.data.get_bu_list():
            bu_normalized = self.data.normalize_lookup_text(bu)
            if f" {bu_normalized} " in normalized:
                matches.append(bu)
        return matches

    def _resolve_bu_list(self, value: str) -> list[str]:
        if not value or not str(value).strip():
            return []
        selected = []
        for part in re.split(r"[,;/|]|\band\b|\bvs\b|\bversus\b", str(value), flags=re.IGNORECASE):
            resolved = self.data.resolve_bu(part.strip())
            if resolved and resolved not in selected:
                selected.append(resolved)
        if selected:
            return selected
        resolved = self.data.resolve_bu(value)
        return [resolved] if resolved else []

    @staticmethod
    def _month_name(month: int) -> str:
        names = [
            "",
            "January",
            "February",
            "March",
            "April",
            "May",
            "June",
            "July",
            "August",
            "September",
            "October",
            "November",
            "December",
        ]
        return names[month] if 1 <= month <= 12 else str(month)

    def _extract_doctor_from_text(self, text: str) -> str | None:
        lowered = text.lower()
        for doctor in self.data.get_doctor_list():
            if str(doctor).lower() in lowered:
                return doctor
        return None

    def _doctor_name_mask(self, df, doctor_name: str):
        normalized = self.data.normalize_lookup_text(doctor_name)
        return (
            df["Doctor Name"].astype(str).map(self.data.normalize_lookup_text)
            == normalized
        )

    def _doctor_filter(self, df, doctor_name: str):
        if "Doctor Name" not in df.columns:
            return df.iloc[0:0]
        return df[self._doctor_name_mask(df, doctor_name)]

    def _doctor_scoped_df(self, doctor_name: str, bu: str | None = None):
        df = self.data.df.copy()
        if bu:
            df = df[df["BU"] == bu]
        return self._doctor_filter(df, doctor_name)

    def _doctor_has_multiple_bus(self, doctor_name: str) -> bool:
        return len(self.data.get_bus_for_doctor(doctor_name)) > 1

    @staticmethod
    def _doctor_identity_label(doctor_name: str, bu: str | None = None) -> str:
        return f"{doctor_name} ({bu})" if bu else str(doctor_name)

    def _unknown_metric_message(self, metric_name: str) -> str:
        return (
            f"KPI '{metric_name}' was not found. Available KPI examples: "
            f"{', '.join(self._available_kpi_columns()[:20])}"
        )

    def _unknown_knowledge_kpi_message(
        self,
        query: str,
        dataverse_status: str = "",
    ) -> str:
        suggestions = self._knowledge_kpi_suggestions(query)
        suggestion_text = (
            f" Closest available KPI(s): {', '.join(suggestions)}."
            if suggestions
            else ""
        )
        dataverse_text = f" Dataverse fallback: {dataverse_status}" if dataverse_status else ""
        return (
            "I could not find that KPI in the loaded dataset or KPI knowledge base, "
            "so I should not provide a definition or formula for it as if it were "
            f"configured.{suggestion_text}{dataverse_text}"
        )

    def _knowledge_kpi_suggestions(self, query: str) -> list[str]:
        normalized_query = set(self.data.normalize_lookup_text(query).split())
        scored = []
        for metric in self._available_kpi_columns():
            normalized_metric = set(self.data.normalize_lookup_text(metric).split())
            overlap = len(normalized_query & normalized_metric)
            if overlap:
                scored.append((overlap, metric))

        scored.sort(reverse=True)
        return [metric for _, metric in scored[:5]]

    def _unknown_bu_message(self, bu_name: str) -> str:
        return f"BU '{bu_name}' not found. Available BUs: {', '.join(self.data.get_bu_list())}"

    def _available_kpi_columns(self) -> list[str]:
        non_kpi = {
            "Date",
            "YearMonth",
            "Year",
            "Month_Num",
            "Month",
            "Month No",
            "BU",
            "Doctor Name",
        }
        return [column for column in self.data.df.columns if column not in non_kpi]

    @staticmethod
    def _format_list(items: list[str]) -> str:
        clean_items = [
            str(item).strip() for item in items if item and str(item).strip()
        ]
        if not clean_items:
            return "- No knowledge-base recommendation configured."
        return "\n".join(f"- {item}" for item in clean_items)

    @staticmethod
    def _message_content_to_text(message) -> str:
        content = message.content if isinstance(message, AIMessage) else str(message)
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            parts = []
            for item in content:
                if isinstance(item, str):
                    parts.append(item)
                elif isinstance(item, dict):
                    parts.append(item.get("text") or str(item))
                else:
                    parts.append(str(item))
            return "\n".join(parts)
        return str(content)

    @staticmethod
    def _compact_text(value: str, max_length: int = 500) -> str:
        text = re.sub(r"\s+", " ", str(value)).strip()
        if len(text) <= max_length:
            return text
        return text[: max_length - 3].rstrip() + "..."

    def _fallback_response(self) -> str:
        return f"""
Agent running in basic mode because the LLM is not available.

To enable full AI capabilities:
1. Create a Groq API key: https://console.groq.com/keys
2. Set GROQ_API_KEY in your environment
3. Restart the agent

Currently loaded: {len(self.data.df)} records, {len(self.data.get_doctor_list())} doctors.
""".strip()
