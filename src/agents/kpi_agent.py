"""Main KPI agent with LangChain and metadata-driven analytics."""

from __future__ import annotations

import re
import sys
from pathlib import Path

from langchain.agents import create_agent
from langchain.tools import tool
from langchain_core.messages import AIMessage, HumanMessage
from langchain_ollama import ChatOllama

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.analytics.engine import AnalyticsEngine
from src.config import config
from src.data.loader import OPDDataLoader


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

        self.llm = None
        self.agent_executor = None
        self.chat_history = []

        self._init_llm()
        self._create_agent()

    def _init_llm(self):
        """Initialize Ollama LLM."""
        try:
            self.llm = ChatOllama(
                model=self.config.llm_model,
                temperature=self.config.temperature,
                base_url=self.config.ollama_base_url,
                num_ctx=self.config.llm_num_ctx,
                num_predict=self.config.llm_num_predict,
                num_thread=self.config.llm_num_thread,
                num_gpu=self.config.llm_num_gpu,
                keep_alive=self.config.llm_keep_alive,
            )
            print(f"LLM initialized: {self.config.llm_model}")
        except Exception as exc:
            print(f"LLM not available: {exc}")
            print(f"Run: ollama pull {self.config.llm_model}")
            self.llm = None

    def _create_agent(self):
        """Create the LangChain agent and expose safe analytical tools."""

        @tool
        def get_doctor_performance(doctor_name: str) -> str:
            """Get performance summary for a specific doctor."""
            return self._format_doctor_performance(doctor_name)

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
        def get_kpi_trend(kpi_name: str, months: int = 6, bu_name: str = "") -> str:
            """Get recent trend for a KPI. Optional bu_name can be ASH, SMH, or HJH."""
            metric = self.data.resolve_kpi(kpi_name)
            if metric is None:
                return self._unknown_metric_message(kpi_name)

            bu = self._resolve_optional_bu(bu_name)
            if bu_name and bu is None:
                return self._unknown_bu_message(bu_name)

            return self._format_kpi_trend(metric, months=months, bu=bu)

        @tool
        def get_bu_summary(bu_name: str) -> str:
            """Get performance summary for a Business Unit."""
            bu = self.data.resolve_bu(bu_name)
            if bu is None:
                return self._unknown_bu_message(bu_name)
            return self._format_bu_summary(bu)

        tools = [
            get_doctor_performance,
            analyze_root_cause,
            compare_doctors,
            get_kpi_trend,
            get_bu_summary,
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
        doctors = ", ".join(self.data.get_doctor_list()[:20])
        return (
            "You are a professional OPD KPI Analytics Agent for healthcare "
            "operations.\n\n"
            "Rules:\n"
            "1. Use tools for all dataset questions. Never invent numbers.\n"
            "2. The tools resolve natural KPI wording through the loaded knowledge-base "
            "workbook and dataset columns.\n"
            "3. If the user mentions a BU, pass it to the tool.\n"
            "4. For root-cause questions, call analyze_root_cause.\n"
            "5. Explain what the numbers mean operationally. Include drivers, "
            "risks, and next actions.\n"
            "6. Keep answers concise but thorough: executive summary, evidence, "
            "interpretation, and recommendations.\n\n"
            f"Available BUs: {bus}.\n"
            f"Available doctors include: {doctors}.\n"
            f"Available KPI columns include: {kpis}."
        )

    def chat(self, user_input: str) -> str:
        """Process user message."""
        direct_response = self._direct_analytics_response(user_input)
        if direct_response:
            return direct_response

        if self.agent_executor is None:
            return self._fallback_response()

        try:
            messages = self.chat_history + [HumanMessage(user_input)]
            response = self.agent_executor.invoke({"messages": messages})
            output = self._message_content_to_text(response["messages"][-1])
            self.chat_history.extend([HumanMessage(user_input), AIMessage(output)])
            return output
        except Exception as exc:
            return f"Error: {exc}"

    def _direct_analytics_response(self, user_input: str) -> str | None:
        """Handle high-confidence analytics paths without waiting for the LLM loop."""
        text = user_input.lower()
        metric = self.data.resolve_kpi(user_input)
        bu = self._extract_bu_from_text(user_input)
        doctor = self._extract_doctor_from_text(user_input)
        threshold_filter = self._extract_threshold_filter(user_input)
        year = self._extract_year_from_text(user_input)

        if metric and threshold_filter and "doctor" in text:
            return self._format_doctors_by_threshold(
                metric,
                operator=threshold_filter["operator"],
                threshold=threshold_filter["threshold"],
                bu=bu,
                year=year,
            )

        if (
            doctor
            and metric
            and any(
                term in text for term in ["justify", "explain", "performance", "why"]
            )
        ):
            return self._format_doctor_kpi_justification(doctor, metric, bu=bu)

        if doctor and any(
            term in text
            for term in ["justification", "justifications", "justify", "explain"]
        ):
            return self._format_doctor_kpi_profile(doctor, bu=bu, year=year)

        if metric and any(
            term in text for term in ["root cause", "root causes", "why", "analyze"]
        ):
            return self._format_root_cause(metric, bu=bu)

        if metric and any(term in text for term in ["trend", "over time", "changed"]):
            return self._format_kpi_trend(metric, bu=bu, year=year)

        if (
            metric
            and any(term in text for term in ["compare", "across"])
            and any(
                term in text
                for term in ["bu", "bus", "business unit", "business units"]
            )
        ):
            return self._format_bu_comparison(metric, year=year)

        if metric and any(
            term in text for term in ["compare", "top", "rank", "which doctors"]
        ):
            return self._format_doctor_comparison(metric, bu=bu, year=year)

        if metric and any(term in text for term in ["what", "show", "tell", "give"]):
            return self._format_metric_overview(metric, bu=bu, year=year)

        if doctor and any(term in text for term in ["performance", "summary", "show"]):
            return self._format_doctor_performance(doctor)

        if bu and "summary" in text:
            return self._format_bu_summary(bu)

        return None

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
        df = self.data.df[
            self.data.df["Doctor Name"].str.contains(
                doctor_name,
                case=False,
                na=False,
                regex=False,
            )
        ].copy()
        if bu:
            df = df[df["BU"] == bu]
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
        peer_value = self.analytics._aggregate_metric(peer_df, metric)

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
Justification: Dr. {doctor_name}'s {metric} performance{scope}

Executive readout:
Dr. {doctor_name}'s current {metric} is {self._format_metric_value(metric, current_value)}, changing {change_pct:+.1f}% versus the previous period. Against the selected peer group, this is {peer_gap_text}. Revenue achievement is {achievement:.1f}%, so the KPI should be interpreted alongside volume, retention, no-show, and leakage behavior rather than in isolation.

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
    ) -> str:
        months = max(1, min(int(months or 6), 24))
        df = self._scoped_df(bu=bu, year=year)
        df = df.sort_values("Date").tail(months)

        if metric not in df.columns:
            return self._unknown_metric_message(metric)

        trend_data = df[["YearMonth", metric]].dropna()
        if trend_data.empty:
            return f"No trend data for {metric}{' in ' + bu if bu else ''}"

        first = float(trend_data[metric].iloc[0])
        last = float(trend_data[metric].iloc[-1])
        change = ((last - first) / first * 100) if first else 0
        direction = (
            "Increasing" if change > 5 else "Decreasing" if change < -5 else "Stable"
        )
        scope = self._scope_text(bu=bu, year=year)

        lines = [f"{direction} trend for {metric}{scope} (change: {change:+.1f}%)", ""]
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
    ) -> str:
        df = self._scoped_df(bu=bu, year=year)
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
        scope = self._scope_text(bu=bu, year=year)

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

    def _format_doctor_comparison(
        self,
        metric: str,
        bu: str | None = None,
        year: int | None = None,
    ) -> str:
        df = self._scoped_df(bu=bu, year=year)
        if metric not in df.columns or df.empty:
            return f"No data for metric: {metric}"

        aggregate = "sum" if self._is_additive_metric(metric) else "mean"
        if aggregate == "sum":
            ranking = df.groupby("Doctor Name")[metric].sum().reset_index()
        else:
            ranking = df.groupby("Doctor Name")[metric].mean().reset_index()
        ranking = (
            ranking.sort_values(metric, ascending=False).head(5).reset_index(drop=True)
        )
        ranking["rank"] = ranking.index + 1

        if ranking.empty:
            return f"No data for metric: {metric}"

        scope = self._scope_text(bu=bu, year=year)
        lines = [f"Top doctors by {metric}{scope}:"]
        for _, row in ranking.iterrows():
            lines.append(
                f"{row['rank']}. {row['Doctor Name']}: "
                f"{self._format_metric_value(metric, row[metric])}"
            )
        return "\n".join(lines)

    def _format_bu_comparison(self, metric: str, year: int | None = None) -> str:
        df = self._scoped_df(year=year)
        if metric not in df.columns or df.empty:
            return f"No data for metric: {metric}"

        aggregate = "sum" if self._is_additive_metric(metric) else "mean"
        if aggregate == "sum":
            ranking = df.groupby("BU")[metric].sum().reset_index()
        else:
            ranking = df.groupby("BU")[metric].mean().reset_index()

        ranking = ranking.sort_values(metric, ascending=False).reset_index(drop=True)
        scope = self._scope_text(year=year)
        best = ranking.iloc[0]
        worst = ranking.iloc[-1]

        lines = [
            f"BU comparison: {metric}{scope}",
            "",
            "Executive readout:",
            (
                f"{best['BU']} has the highest {metric} at "
                f"{self._format_metric_value(metric, best[metric])}, while "
                f"{worst['BU']} has the lowest at "
                f"{self._format_metric_value(metric, worst[metric])}. "
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
                "- Compare the highest and lowest BUs by doctor, specialty, and month to identify where the gap is concentrated.",
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
    ) -> str:
        df = self._scoped_df(bu=bu, year=year)
        if metric not in df.columns or df.empty:
            return f"No data for metric: {metric}"
        if ("%" in metric or "cr%" in metric.lower()) and threshold > 1:
            threshold = threshold / 100

        if self._is_additive_metric(metric):
            grouped = df.groupby("Doctor Name")[metric].sum().reset_index()
        else:
            grouped = df.groupby("Doctor Name")[metric].mean().reset_index()

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

        scope = self._scope_text(bu=bu, year=year)
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
            lines.append(
                f"{index}. {row['Doctor Name']}: "
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

    def _format_doctor_performance(self, doctor_name: str) -> str:
        df_doctor = self.data.df[
            self.data.df["Doctor Name"].str.contains(
                doctor_name,
                case=False,
                na=False,
                regex=False,
            )
        ]
        if df_doctor.empty:
            return (
                f"Doctor '{doctor_name}' not found. Available doctors: "
                f"{', '.join(self.data.get_doctor_list()[:10])}"
            )

        total_revenue = float(df_doctor["Total Revenue"].sum())
        target_revenue = float(df_doctor["Target Revenue"].sum())
        achievement = (total_revenue / target_revenue * 100) if target_revenue else 0

        return f"""
Doctor: {doctor_name}
- Total Revenue: ${total_revenue:,.0f}
- Target Revenue: ${target_revenue:,.0f}
- Achievement: {achievement:.1f}%
- Cases: {df_doctor["No. Cases"].sum():,.0f}
- Avg No-Show: {df_doctor["No-Show %"].mean() * 100:.1f}%
- Avg Retention: {df_doctor["Patient Retention %"].mean() * 100:.1f}%
""".strip()

    def _format_doctor_kpi_profile(
        self,
        doctor_name: str,
        bu: str | None = None,
        year: int | None = None,
    ) -> str:
        df = self._scoped_df(bu=bu, year=year)
        df_doctor = df[
            df["Doctor Name"].str.contains(
                doctor_name,
                case=False,
                na=False,
                regex=False,
            )
        ].copy()
        if df_doctor.empty:
            scope = self._scope_text(bu=bu, year=year)
            return (
                f"Doctor '{doctor_name}' not found{scope}. Available doctors: "
                f"{', '.join(self.data.get_doctor_list()[:10])}"
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

        scope = self._scope_text(bu=bu, year=year)
        lines = [
            f"Doctor KPI Performance and Justification: Dr. {doctor_name}{scope}",
            "",
            "Executive readout:",
            (
                f"Dr. {doctor_name} generated ${total_revenue:,.0f} against a "
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
        if any(word in metric_lower for word in ["revenue", "losses"]):
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

        if self._is_additive_metric(metric):
            per_doctor = peer_df.groupby("Doctor Name")[metric].sum()
        else:
            per_doctor = peer_df.groupby("Doctor Name")[metric].mean()
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
    ):
        df = self.data.df.copy()
        if bu:
            df = df[df["BU"] == bu]
        if year and "Year" in df.columns:
            df = df[df["Year"].astype("Int64") == year]
        return df

    def _scope_text(
        self,
        bu: str | None = None,
        year: int | None = None,
    ) -> str:
        parts = []
        if bu:
            parts.append(str(bu))
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
        lowered = text.lower()
        for bu in self.data.get_bu_list():
            if str(bu).lower() in lowered:
                return bu
        return None

    def _extract_doctor_from_text(self, text: str) -> str | None:
        lowered = text.lower()
        for doctor in self.data.get_doctor_list():
            if str(doctor).lower() in lowered:
                return doctor
        return None

    def _unknown_metric_message(self, metric_name: str) -> str:
        return (
            f"KPI '{metric_name}' was not found. Available KPI examples: "
            f"{', '.join(self._available_kpi_columns()[:20])}"
        )

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

    def _fallback_response(self) -> str:
        return f"""
Agent running in basic mode because the LLM is not available.

To enable full AI capabilities:
1. Install Ollama: https://ollama.com
2. Run: ollama pull {self.config.llm_model}
3. Restart the agent

Currently loaded: {len(self.data.df)} records, {len(self.data.get_doctor_list())} doctors.
""".strip()
