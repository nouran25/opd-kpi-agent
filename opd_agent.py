"""
OPD KPI Intelligence Agent
===================================================
Capabilities:
- Natural language question answering
- Automated root cause analysis with statistical rigor
- Multi-dimensional data slicing (doctor, BU, date, specialty)
- Trend detection and anomaly flagging
- KPI relationship traversal (driver/cause analysis)
- Automated report generation
- Conversational memory

Tech Stack:
- LLM: Qwen 2.5 (via Ollama)
- Framework: LangGraph (stateful agent)
- Data: Pandas + SQLite (for query caching)
- Vector Store: Chroma (for RAG on knowledge base)
- UI: Gradio (web interface)
"""

import os
import re
import json
import sqlite3
import warnings
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from collections import defaultdict

import pandas as pd
import numpy as np
from scipy import stats

# LangChain & LLM imports
from langchain_ollama import ChatOllama
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain.agents import AgentExecutor, create_react_agent
from langchain.tools import tool
from langchain.memory import ConversationBufferMemory

# Web UI
import gradio as gr

warnings.filterwarnings("ignore")

# ============================================================================
# CONFIGURATION
# ============================================================================


@dataclass
class Config:
    """Agent configuration"""

    # Model settings
    llm_model: str = "qwen2.5:7b"
    embedding_model: str = "nomic-embed-text"  # Free embedding model
    temperature: float = 0.1
    # Data paths
    dataset_path: str = "OPD dataset.xlsx"
    knowledge_path: str = "Knowledge base.xlsx"
    # Database
    db_path: str = "opd_analytics.db"
    vector_store_path: str = "./chroma_db"
    # Analysis thresholds
    revenue_leakage_threshold: float = 0.10  # 10%
    no_show_threshold: float = 0.25  # 25%
    retention_threshold: float = 0.60  # 60%


# ============================================================================
# DATA LAYER
# ============================================================================


class OPDDataLayer:
    """Handles all data operations with caching"""

    def __init__(self, config: Config):
        self.config = config
        self.df = None
        self.knowledge_base = None
        self._load_data()
        self._init_database()

    def _load_data(self):
        """Load and preprocess all data"""
        # Load OPD dataset
        self.df = pd.read_excel(self.config.dataset_path, sheet_name="OPD_KPI_Dataset")

        # Clean column names
        self.df.columns = self.df.columns.str.strip()

        # Convert date
        self.df["Date"] = pd.to_datetime(self.df["Month"])
        self.df["YearMonth"] = self.df["Date"].dt.to_period("M")
        self.df["Year"] = self.df["Date"].dt.year
        self.df["Month_Num"] = self.df["Date"].dt.month
        self.df["Quarter"] = self.df["Date"].dt.quarter

        # Load knowledge base
        self.knowledge_base = {}
        sheets = [
            "adx_kpi_knowledge_map_x0009__x0009__x0009_",
            "adx_kpi_relationship_map_x0009__x0009__x0009_",
            "adx_kpi_formula_definition_x0009__x0009__x0009_",
            "adx_kpi_investigation_playbook",
            "adx_kpi_filter_compatibility",
        ]

        for sheet in sheets:
            try:
                self.knowledge_base[sheet] = pd.read_excel(
                    self.config.knowledge_path, sheet_name=sheet
                )
            except Exception as e:
                print(f"Warning: Could not load {sheet}: {e}")

        # Create derived metrics
        self._create_derived_metrics()

    def _create_derived_metrics(self):
        """Create additional analytical metrics"""
        # Revenue Achievement
        self.df["Revenue_Achievement_%"] = (
            self.df["Total Revenue"] / self.df["Target Revenue"] * 100
        )

        # Cases Achievement
        self.df["Cases_Achievement_%"] = (
            self.df["No. Cases"] / self.df["Target No. cases"] * 100
        )

        # Revenue per Case (already have Charge per case, but confirming)
        self.df["Revenue_per_Case"] = self.df["Total Revenue"] / self.df["No. Cases"]

        # Leakage Impact (Revenue lost / Total Revenue)
        self.df["Leakage_Impact_%"] = (
            self.df["Total Leakage Revenue Losses"] / self.df["Total Revenue"] * 100
        )

        # Overall Performance Score (composite)
        self.df["Performance_Score"] = (
            self.df["Revenue_Achievement_%"].clip(0, 100) * 0.4
            + self.df["Cases_Achievement_%"].clip(0, 100) * 0.3
            + (1 - self.df["Service Leakage %"]) * 100 * 0.15
            + self.df["Patient Retention %"] * 100 * 0.15
        )

    def _init_database(self):
        """Initialize SQLite for query caching"""
        self.conn = sqlite3.connect(self.config.db_path)
        self.df.to_sql("kpi_data", self.conn, if_exists="replace", index=False)

    def get_doctor_list(self) -> List[str]:
        """Get unique doctor names"""
        return self.df["Doctor Name"].dropna().unique().tolist()

    def get_bu_list(self) -> List[str]:
        """Get unique BUs"""
        return self.df["BU"].dropna().unique().tolist()

    def get_available_years(self) -> List[int]:
        """Get available years"""
        return sorted(self.df["Year"].dropna().unique().tolist())

    def query_kpi_data(
        self,
        kpi_name: str = None,
        doctor: str = None,
        bu: str = None,
        year: int = None,
        month: int = None,
    ) -> pd.DataFrame:
        """Flexible query interface"""
        df = self.df.copy()

        if kpi_name and kpi_name in df.columns:
            # Return the specific KPI
            pass

        if doctor:
            df = df[df["Doctor Name"] == doctor]
        if bu:
            df = df[df["BU"] == bu]
        if year:
            df = df[df["Year"] == year]
        if month:
            df = df[df["Month_Num"] == month]

        return df

    def get_doctor_summary(self, doctor_name: str) -> Dict:
        """Get comprehensive summary for a doctor"""
        df_doctor = self.df[self.df["Doctor Name"] == doctor_name]

        if df_doctor.empty:
            return {"error": f"Doctor {doctor_name} not found"}

        return {
            "doctor": doctor_name,
            "bus": df_doctor["BU"].iloc[0] if not df_doctor.empty else None,
            "total_revenue": df_doctor["Total Revenue"].sum(),
            "avg_revenue_per_month": df_doctor["Total Revenue"].mean(),
            "total_target_revenue": df_doctor["Target Revenue"].sum(),
            "revenue_achievement": df_doctor["Total Revenue"].sum()
            / df_doctor["Target Revenue"].sum()
            * 100,
            "total_cases": df_doctor["No. Cases"].sum(),
            "avg_cases_per_month": df_doctor["No. Cases"].mean(),
            "avg_charge_per_case": df_doctor["Charge per case"].mean(),
            "avg_pms_score": df_doctor["Doctor PMS %"].mean() * 100,
            "avg_retention": df_doctor["Patient Retention %"].mean() * 100,
            "avg_no_show": df_doctor["No-Show %"].mean() * 100,
            "avg_service_leakage": df_doctor["Service Leakage %"].mean() * 100,
            "total_leakage_loss": df_doctor["Total Leakage Revenue Losses"].sum(),
            "months_of_data": len(df_doctor),
            "trend": self._calculate_trend(df_doctor),
        }

    def _calculate_trend(self, df_doctor: pd.DataFrame) -> Dict:
        """Calculate trend direction and significance"""
        if len(df_doctor) < 3:
            return {"direction": "insufficient_data"}

        revenue_trend = df_doctor.sort_values("Date")["Total Revenue"].values
        slope = np.polyfit(range(len(revenue_trend)), revenue_trend, 1)[0]

        if slope > df_doctor["Total Revenue"].mean() * 0.05:
            direction = "increasing"
        elif slope < -df_doctor["Total Revenue"].mean() * 0.05:
            direction = "decreasing"
        else:
            direction = "stable"

        return {"direction": direction, "slope": slope}

    def get_kpi_relationships(self, kpi_name: str) -> Dict:
        """Get KPI relationships from knowledge base"""
        rel_map = self.knowledge_base.get(
            "adx_kpi_relationship_map_x0009__x0009__x0009_"
        )
        if rel_map is None:
            return {"drivers": [], "impacts": []}

        drivers = rel_map[rel_map["Child_KPI"] == kpi_name][
            ["Parent_KPI", "Relationship_Type", "Weight"]
        ].to_dict("records")
        impacts = rel_map[rel_map["Parent_KPI"] == kpi_name][
            ["Child_KPI", "Relationship_Type", "Weight"]
        ].to_dict("records")

        return {"drivers": drivers, "impacts": impacts}

    def get_investigation_playbook(
        self, kpi_name: str, variance_pct: float = None
    ) -> List[Dict]:
        """Get relevant investigation steps from playbook"""
        playbook = self.knowledge_base.get("adx_kpi_investigation_playbook")
        if playbook is None:
            return []

        relevant = []
        for _, row in playbook[playbook["KPI"] == kpi_name].iterrows():
            threshold = row["Threshold"]
            if variance_pct is not None:
                # Parse threshold
                if "<" in str(threshold) and variance_pct < 0:
                    threshold_val = float(re.search(r"(\d+)", threshold).group(1))
                    if abs(variance_pct) > threshold_val:
                        relevant.append(row.to_dict())
                elif ">" in str(threshold) and variance_pct > 0:
                    threshold_val = float(re.search(r"(\d+)", threshold).group(1))
                    if variance_pct > threshold_val:
                        relevant.append(row.to_dict())
            else:
                relevant.append(row.to_dict())

        return relevant


# ============================================================================
# ANALYTICS ENGINE
# ============================================================================


class AnalyticsEngine:
    """Statistical analysis and root cause detection"""

    def __init__(self, data_layer: OPDDataLayer):
        self.data = data_layer

    def detect_anomalies(
        self, df: pd.DataFrame, metric: str, threshold: float = 2.0
    ) -> pd.DataFrame:
        """Detect statistical anomalies using Z-score"""
        if metric not in df.columns:
            return pd.DataFrame()

        values = df[metric].dropna()
        if len(values) < 3:
            return pd.DataFrame()

        z_scores = np.abs(stats.zscore(values))
        anomalies = df.iloc[z_scores > threshold].copy()
        anomalies["z_score"] = z_scores[z_scores > threshold]

        return anomalies

    def root_cause_analysis(
        self,
        kpi_name: str,
        doctor: str = None,
        bu: str = None,
        period: str = "last_month",
    ) -> Dict:
        """
        Perform statistical root cause analysis
        """
        df = self.data.query_kpi_data(doctor=doctor, bu=bu)

        if df.empty:
            return {"error": "No data available"}

        # Get current vs previous period
        if period == "last_month":
            latest_date = df["Date"].max()
            previous_date = latest_date - timedelta(days=30)

            current = df[df["Date"] == latest_date]
            previous = df[df["Date"] == previous_date]
        else:
            # Use all data for trend
            current = df.sort_values("Date").tail(1)
            previous = df.sort_values("Date").head(1)

        if current.empty or previous.empty:
            return {"error": "Insufficient data for comparison"}

        # Calculate variance
        current_value = (
            current[kpi_name].iloc[0] if kpi_name in current.columns else None
        )
        previous_value = (
            previous[kpi_name].iloc[0] if kpi_name in previous.columns else None
        )

        if current_value is None or previous_value is None:
            return {"error": f"KPI {kpi_name} not found"}

        variance_abs = current_value - previous_value
        variance_pct = (
            (variance_abs / previous_value * 100) if previous_value != 0 else 0
        )

        # Get drivers from knowledge base
        relationships = self.data.get_kpi_relationships(kpi_name)

        # Analyze each driver
        driver_analysis = []
        for driver in relationships.get("drivers", []):
            driver_name = driver["Parent_KPI"]
            if driver_name in current.columns:
                driver_current = current[driver_name].iloc[0]
                driver_previous = previous[driver_name].iloc[0]
                driver_change = (
                    ((driver_current - driver_previous) / driver_previous * 100)
                    if driver_previous != 0
                    else 0
                )

                driver_analysis.append(
                    {
                        "driver": driver_name,
                        "relationship": driver["Relationship_Type"],
                        "weight": driver["Weight"],
                        "current": driver_current,
                        "previous": driver_previous,
                        "change_pct": driver_change,
                        "contribution_to_kpi": abs(driver_change)
                        * (1 if driver["Weight"] == "High" else 0.5),
                    }
                )

        # Sort by contribution
        driver_analysis.sort(key=lambda x: x["contribution_to_kpi"], reverse=True)

        # Get playbook recommendations
        playbook = self.data.get_investigation_playbook(kpi_name, variance_pct)

        return {
            "kpi": kpi_name,
            "current_value": current_value,
            "previous_value": previous_value,
            "variance_abs": variance_abs,
            "variance_pct": variance_pct,
            "trend": "declining"
            if variance_pct < -5
            else "improving"
            if variance_pct > 5
            else "stable",
            "severity": "critical"
            if abs(variance_pct) > 20
            else "high"
            if abs(variance_pct) > 10
            else "medium"
            if abs(variance_pct) > 5
            else "low",
            "primary_drivers": driver_analysis[:3],
            "recommended_investigations": [
                p.get("Recommended_Investigation", "") for p in playbook
            ],
            "recommended_actions": [p.get("Recommended_Action", "") for p in playbook],
            "escalation": playbook[0].get("Escalation", "Manager")
            if playbook
            else "Manager",
        }

    def compare_periods(
        self, metric: str, period1: Tuple[str, str], period2: Tuple[str, str]
    ) -> Dict:
        """
        Compare performance between two time periods
        period = (start_date, end_date) format 'YYYY-MM-DD'
        """
        df = self.data.df

        mask1 = (df["Date"] >= period1[0]) & (df["Date"] <= period1[1])
        mask2 = (df["Date"] >= period2[0]) & (df["Date"] <= period2[1])

        period1_data = df[mask1][metric]
        period2_data = df[mask2][metric]

        return {
            "metric": metric,
            "period1_value": period1_data.sum()
            if "Revenue" in metric
            else period1_data.mean(),
            "period2_value": period2_data.sum()
            if "Revenue" in metric
            else period2_data.mean(),
            "change_pct": (
                (period1_data.sum() - period2_data.sum()) / period2_data.sum() * 100
            )
            if "Revenue" in metric
            else (
                (period1_data.mean() - period2_data.mean()) / period2_data.mean() * 100
            ),
            "period1_dates": f"{period1[0]} to {period1[1]}",
            "period2_dates": f"{period2[0]} to {period2[1]}",
        }

    def doctor_ranking(
        self, metric: str, bu: str = None, top_n: int = 10
    ) -> pd.DataFrame:
        """Rank doctors by performance metric"""
        df = self.data.query_kpi_data(bu=bu)

        # Aggregate by doctor
        if "Revenue" in metric:
            ranking = df.groupby("Doctor Name")[metric].sum().reset_index()
        else:
            ranking = df.groupby("Doctor Name")[metric].mean().reset_index()

        ranking = ranking.sort_values(metric, ascending=False).head(top_n)
        ranking["rank"] = range(1, len(ranking) + 1)

        return ranking


# ============================================================================
# LLM AGENT WITH TOOLS
# ============================================================================


class OPDAgent:
    """LangGraph-based AI agent with tool calling capabilities"""

    def __init__(self, config: Config):
        self.config = config
        self.data_layer = OPDDataLayer(config)
        self.analytics = AnalyticsEngine(self.data_layer)
        self.llm = None
        self.agent_executor = None
        self.memory = ConversationBufferMemory(
            memory_key="chat_history", return_messages=True
        )

        self._initialize_llm()
        self._create_agent()

    def _initialize_llm(self):
        """Initialize the local LLM"""
        try:
            self.llm = ChatOllama(
                model=self.config.llm_model,
                temperature=self.config.temperature,
                base_url="http://localhost:11434",
            )
            print(f"✅ LLM initialized: {self.config.llm_model}")
        except Exception as e:
            print("⚠️ Ollama not running. Starting with fallback mode.")
            print("   Run: ollama pull {self.config.llm_model}")
            self.llm = None

    def _create_agent(self):
        """Create the agent with tools"""

        # Define tools
        @tool
        def get_doctor_performance(doctor_name: str) -> str:
            """Get comprehensive performance summary for a specific doctor.
            Input: doctor name (e.g., 'Ahmed', 'Omar')"""
            summary = self.data_layer.get_doctor_summary(doctor_name)
            if "error" in summary:
                return summary["error"]

            return f"""
            Doctor: {summary["doctor"]} (BU: {summary["bus"]})
            📊 Performance Summary:
            - Total Revenue: ${summary["total_revenue"]:,.0f}
            - Revenue Achievement: {summary["revenue_achievement"]:.1f}%
            - Avg Cases/Month: {summary["avg_cases_per_month"]:.0f}
            - Avg Charge/Case: ${summary["avg_charge_per_case"]:,.0f}
            - PMS Score: {summary["avg_pms_score"]:.1f}%
            - Patient Retention: {summary["avg_retention"]:.1f}%
            - No-Show Rate: {summary["avg_no_show"]:.1f}%
            - Service Leakage: {summary["avg_service_leakage"]:.1f}%
            - Leakage Losses: ${summary["total_leakage_loss"]:,.0f}
            - Trend: {summary["trend"]["direction"]}
            """

        @tool
        def analyze_root_cause(kpi_name: str, doctor_name: str = None) -> str:
            """Analyze root causes for KPI underperformance.
            Input: KPI name (e.g., 'Total Revenue', 'No. Cases'), optional doctor name"""
            analysis = self.analytics.root_cause_analysis(kpi_name, doctor=doctor_name)

            if "error" in analysis:
                return analysis["error"]

            severity_emoji = {
                "critical": "🔴",
                "high": "🟠",
                "medium": "🟡",
                "low": "🟢",
            }

            result = f"""
            {severity_emoji.get(analysis["severity"], "📊")} ROOT CAUSE ANALYSIS: {kpi_name}
            ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
            
            📈 Performance Change: {analysis["variance_pct"]:+.1f}% ({analysis["trend"]})
            📍 Current Value: {analysis["current_value"]:,.0f}
            📍 Previous Value: {analysis["previous_value"]:,.0f}
            🚨 Severity: {analysis["severity"].upper()}
            
            🔍 PRIMARY DRIVERS TO INVESTIGATE:
            """

            for i, driver in enumerate(analysis["primary_drivers"], 1):
                impact = "⬆️" if driver["change_pct"] > 0 else "⬇️"
                result += f"""
            {i}. {driver["driver"]} {impact} {driver["change_pct"]:+.1f}%
               - Current: {driver["current"]:.0f} | Previous: {driver["previous"]:.0f}
               - Relationship: {driver["relationship"]} (Weight: {driver["weight"]})
            """

            result += f"""
            
            📋 RECOMMENDED INVESTIGATIONS:
            {chr(10).join([f"   • {inv}" for inv in analysis["recommended_investigations"][:3]])}
            
            ✅ RECOMMENDED ACTIONS:
            {chr(10).join([f"   • {action}" for action in analysis["recommended_actions"][:3]])}
            
            ⚡ ESCALATION: {analysis["escalation"]}
            """

            return result

        @tool
        def compare_doctors(metric: str, bu: str = None) -> str:
            """Compare doctors on a specific metric.
            Input: metric (e.g., 'Total Revenue', 'Patient Retention %'), optional BU"""
            ranking = self.analytics.doctor_ranking(metric, bu=bu, top_n=5)

            if ranking.empty:
                return f"No data available for metric: {metric}"

            result = f"\n🏆 TOP 5 DOCTORS by {metric}\n"
            result += "━" * 50 + "\n"

            for _, row in ranking.iterrows():
                icon = (
                    "🥇"
                    if row["rank"] == 1
                    else "🥈"
                    if row["rank"] == 2
                    else "🥉"
                    if row["rank"] == 3
                    else "📌"
                )
                result += (
                    f"{icon} {row['rank']}. {row['Doctor Name']}: {row[metric]:,.0f}\n"
                )

            return result

        @tool
        def get_kpi_definition(kpi_name: str) -> str:
            """Get definition, formula, and owner for a KPI.
            Input: KPI name (e.g., 'Total Revenue', 'No. Cases')"""
            definitions = self.data_layer.knowledge_base.get(
                "adx_kpi_formula_definition_x0009__x0009__x0009_"
            )
            if definitions is None:
                return "Knowledge base not loaded"

            kpi_def = definitions[definitions["KPI_Name"] == kpi_name]
            if kpi_def.empty:
                return f"KPI '{kpi_name}' not found"

            row = kpi_def.iloc[0]

            # Get owner
            owners = self.data_layer.knowledge_base.get(
                "adx_kpi_knowledge_map_x0009__x0009__x0009_"
            )
            owner = ""
            if owners is not None:
                owner_row = owners[owners["KPI_Name"] == kpi_name]
                if not owner_row.empty:
                    owner = owner_row.iloc[0]["KPI_Owner_Role"]

            return f"""
            📖 KPI DEFINITION: {kpi_name}
            ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
            
            Formula Type: {row["Formula_Type"]}
            Formula Logic: {row["Formula_Logic"]}
            Source: {row["Source"]}
            Owner: {owner if owner else "Not specified"}
            """

        @tool
        def get_kpi_trend(kpi_name: str, months: int = 6) -> str:
            """Get trend analysis for a KPI over time.
            Input: KPI name, number of months (default 6)"""
            df = self.data_layer.df.sort_values("Date").tail(months)

            if kpi_name not in df.columns:
                return f"KPI '{kpi_name}' not found"

            trend_data = df[["YearMonth", kpi_name]].dropna()

            if trend_data.empty:
                return f"No trend data for {kpi_name}"

            # Calculate trend
            values = trend_data[kpi_name].values
            if len(values) > 1:
                first = values[0]
                last = values[-1]
                change = ((last - first) / first * 100) if first != 0 else 0

                if change > 5:
                    direction = "📈 INCREASING"
                elif change < -5:
                    direction = "📉 DECREASING"
                else:
                    direction = "➡️ STABLE"
            else:
                change = 0
                direction = "⚠️ INSUFFICIENT DATA"

            result = f"""
            📊 {direction} TREND: {kpi_name} (Last {len(trend_data)} months)
            ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
            
            Period Change: {change:+.1f}%
            """

            # Add monthly breakdown
            for _, row in trend_data.iterrows():
                result += f"\n   {row['YearMonth']}: {row[kpi_name]:,.0f}"

            return result

        @tool
        def get_bu_summary(bu_name: str) -> str:
            """Get summary for a Business Unit.
            Input: BU name (e.g., 'ASH', 'SMH', 'HJH')"""
            df_bu = self.data_layer.query_kpi_data(bu=bu_name)

            if df_bu.empty:
                return f"BU '{bu_name}' not found"

            return f"""
            🏢 BU SUMMARY: {bu_name}
            ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
            
            📊 Overall Performance:
            - Total Revenue: ${df_bu["Total Revenue"].sum():,.0f}
            - Total Target: ${df_bu["Target Revenue"].sum():,.0f}
            - Revenue Achievement: {(df_bu["Total Revenue"].sum() / df_bu["Target Revenue"].sum() * 100):.1f}%
            - Total Cases: {df_bu["No. Cases"].sum():,.0f}
            - Avg No-Show: {df_bu["No-Show %"].mean() * 100:.1f}%
            - Avg Retention: {df_bu["Patient Retention %"].mean() * 100:.1f}%
            
            📈 Top Doctor: {df_bu.groupby("Doctor Name")["Total Revenue"].sum().idxmax()}
            📉 Doctor needing support: {df_bu.groupby("Doctor Name")["Revenue_Achievement_%"].mean().idxmin()}
            """

        @tool
        def generate_report(doctor_name: str = None, bu: str = None) -> str:
            """Generate a comprehensive performance report.
            Input: optional doctor name or BU"""

            if doctor_name:
                summary = self.data_layer.get_doctor_summary(doctor_name)
                if "error" in summary:
                    return summary["error"]

                # Get root causes for top issues
                issues = []
                if summary["revenue_achievement"] < 85:
                    rca = self.analytics.root_cause_analysis(
                        "Total Revenue", doctor=doctor_name
                    )
                    issues.append(
                        f"Revenue below target: {summary['revenue_achievement']:.0f}% achieved"
                    )
                    issues.append(
                        f"  → Primary driver: {rca.get('primary_drivers', [{}])[0].get('driver', 'Unknown')}"
                    )

                if summary["avg_no_show"] > 25:
                    issues.append(f"High no-show rate: {summary['avg_no_show']:.0f}%")
                    issues.append("  → Recommendation: Activate reminder automation")

                if summary["avg_service_leakage"] > 8:
                    issues.append(
                        f"Service leakage: {summary['avg_service_leakage']:.0f}%"
                    )
                    issues.append(
                        "  → Recommendation: Enforce service completion tracking"
                    )

                report = f"""
                📋 PERFORMANCE REPORT: Dr. {doctor_name}
                ═══════════════════════════════════════════════════
                
                📈 REVENUE PERFORMANCE
                ├─ Total Revenue: ${summary["total_revenue"]:,.0f}
                ├─ Target Revenue: ${summary["total_target_revenue"]:,.0f}
                └─ Achievement: {summary["revenue_achievement"]:.0f}%
                
                👥 VOLUME METRICS
                ├─ Total Cases: {summary["total_cases"]:.0f}
                ├─ Avg Cases/Month: {summary["avg_cases_per_month"]:.0f}
                └─ Avg Charge/Case: ${summary["avg_charge_per_case"]:,.0f}
                
                ⚕️ QUALITY METRICS
                ├─ PMS Score: {summary["avg_pms_score"]:.0f}%
                ├─ Patient Retention: {summary["avg_retention"]:.0f}%
                ├─ No-Show Rate: {summary["avg_no_show"]:.0f}%
                └─ Service Leakage: {summary["avg_service_leakage"]:.0f}%
                
                🔴 IDENTIFIED ISSUES & ACTIONS
                """

                for issue in issues:
                    report += f"\n   ⚠️ {issue}"

                report += f"""
                
                💰 FINANCIAL IMPACT
                ├─ Leakage Losses: ${summary["total_leakage_loss"]:,.0f}
                └─ Recovery Opportunity: ${summary["total_leakage_loss"] * 0.7:,.0f}
                
                ⏱️ Trend: {summary["trend"]["direction"]}
                """

                return report

            elif bu:
                return self.get_bu_summary(bu_name=bu)

            else:
                return "Please specify a doctor name or BU for the report"

        # Create tools list
        tools = [
            get_doctor_performance,
            analyze_root_cause,
            compare_doctors,
            get_kpi_definition,
            get_kpi_trend,
            get_bu_summary,
            generate_report,
        ]

        mkdir src, data, notebooks, tests, config

    def chat(self, user_input: str) -> str:
        """Process user input and return response"""
        if self.agent_executor is None:
            return self._fallback_response(user_input)

        try:
            response = self.agent_executor.invoke(
                {"input": user_input, "chat_history": self.memory.chat_memory.messages}
            )

            # Store in memory
            self.memory.chat_memory.add_user_message(user_input)
            self.memory.chat_memory.add_ai_message(response["output"])

            return response["output"]
        except Exception as e:
            return f"⚠️ Error: {str(e)}. Please try rephrasing your question."

    def _fallback_response(self, user_input: str) -> str:
        """Fallback when LLM not available"""
        input_lower = user_input.lower()

        if "doctor" in input_lower and any(
            name in input_lower for name in self.data_layer.get_doctor_list()[:5]
        ):
            # Extract doctor name (simplified)
            doctors = self.data_layer.get_doctor_list()
            for doc in doctors:
                if doc.lower() in input_lower:
                    summary = self.data_layer.get_doctor_summary(doc)
                    if "error" not in summary:
                        return f"""
                        📊 Performance for Dr. {doc}:
                        • Revenue: ${summary["total_revenue"]:,.0f} ({summary["revenue_achievement"]:.0f}% of target)
                        • Cases: {summary["total_cases"]:.0f}
                        • Retention: {summary["avg_retention"]:.0f}%
                        • No-Show: {summary["avg_no_show"]:.0f}%
                        
                        💡 To start the LLM agent, run: ollama pull {self.config.llm_model}
                        """

        return f"""
        🤖 OPD KPI Agent is ready with {len(self.data_layer.get_doctor_list())} doctors, {len(self.data_layer.df)} records.
        
        💡 To enable full AI capabilities, please install and run Ollama:
        1. Download from https://ollama.com
        2. Run: ollama pull {self.config.llm_model}
        3. Restart the agent
        
        📋 Available commands (basic mode):
        - "Show doctor Ahmed performance"
        - "Compare doctors by Total Revenue"
        - "What is No-Show % trend?"
        """


# ============================================================================
# GRADIO WEB INTERFACE
# ============================================================================


def create_web_interface(agent: OPDAgent):
    """Create Gradio web interface"""

    def respond(message, history):
        """Process message and return response"""
        response = agent.chat(message)
        return response

    with gr.Blocks(title="OPD KPI Intelligence Agent", theme=gr.themes.Soft()) as demo:
        gr.Markdown("""
        # 🏥 OPD KPI Intelligence Agent
        
        **Professional Analytics Powered by Open-Source AI**
        
        Ask natural language questions about doctor performance, KPI trends, root causes, and get actionable recommendations.
        """)

        with gr.Row():
            with gr.Column(scale=3):
                chatbot = gr.Chatbot(
                    label="Conversation", height=500, bubble_full_width=False
                )
                msg = gr.Textbox(
                    label="Ask me anything about OPD KPIs",
                    placeholder="e.g., Why is Total Revenue down? Compare doctors by Patient Retention. Show me Ahmed's performance...",
                    lines=2,
                )
                clear = gr.ClearButton([msg, chatbot])

            with gr.Column(scale=1):
                gr.Markdown("""
                ### 💡 Example Questions
                
                **Performance Analysis**
                - Show me Doctor Ahmed's performance
                - Compare doctors by Total Revenue
                - Who are the top 5 doctors by Patient Retention?
                
                **Root Cause Analysis**
                - Why is Total Revenue below target?
                - Root cause for high No-Show %
                - Analyze Doctor Omar's revenue decline
                
                **KPI Definitions**
                - What is Service Leakage %?
                - Show me formula for Charge per case
                
                **Trend Analysis**
                - Show me No-Show % trend
                - How has Patient Retention changed?
                
                **Reports & Summaries**
                - Generate report for Doctor Khaled
                - Show BU summary for ASH
                """)

        msg.submit(respond, [msg, chatbot], [chatbot]).then(lambda: "", None, [msg])

    return demo


# ============================================================================
# MAIN EXECUTION
# ============================================================================


def main():
    """Main entry point"""
    print("""
    ╔═══════════════════════════════════════╗
    ║      OPD KPI Intelligence Agent       ║
    ╚═══════════════════════════════════════╝
    """)

    # Check if Ollama is running
    import subprocess

    try:
        result = subprocess.run(["ollama", "list"], capture_output=True, text=True)
        if result.returncode == 0:
            print("✅ Ollama detected. Full AI capabilities enabled.")
        else:
            print("⚠️ Ollama not running. Starting in basic mode.")
            print("   To enable AI: ollama pull qwen2.5:7b && ollama serve")
    except FileNotFoundError:
        print("⚠️ Ollama not installed. Visit https://ollama.com")

    # Initialize agent
    config = Config()
    agent = OPDAgent(config)

    # Create and launch web interface
    demo = create_web_interface(agent)

    print("\n🚀 Launching web interface at http://localhost:7860")
    demo.launch(share=False, server_name="0.0.0.0", server_port=7860)


if __name__ == "__main__":
    main()
