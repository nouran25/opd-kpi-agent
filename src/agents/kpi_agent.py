"""Main KPI Agent with LangChain integration"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.agents import (
    create_react_agent,
    AgentExecutor,
    create_tool_calling_agent,
)
from langchain.memory import ConversationBufferMemory
from langchain.tools import tool

from src.config import config
from src.data.loader import OPDDataLoader
from src.analytics.engine import AnalyticsEngine


class OPDKpiAgent:
    """Main KPI Analytics Agent"""

    def __init__(self):
        self.config = config
        self.config.ensure_directories()

        # Initialize data layer
        print("📊 Loading data...")
        self.data = OPDDataLoader(self.config)
        self.data.load_all()

        # Initialize analytics
        self.analytics = AnalyticsEngine(self.data)

        # Initialize LLM
        self.llm = None
        self.agent_executor = None
        self.memory = ConversationBufferMemory(
            memory_key="chat_history", return_messages=True
        )

        self._init_llm()
        self._create_agent()

    def _init_llm(self):
        """Initialize Ollama LLM"""
        try:
            self.llm = ChatOllama(
                model=self.config.llm_model,
                temperature=self.config.temperature,
                base_url=self.config.ollama_base_url,
            )
            print(f"✅ LLM initialized: {self.config.llm_model}")
        except Exception as e:
            print(f"⚠️ LLM not available: {e}")
            print(f"   Run: ollama pull {self.config.llm_model}")
            self.llm = None

    def _create_agent(self):
        """Create agent with tools"""

        @tool
        def get_doctor_performance(doctor_name: str) -> str:
            """Get performance summary for a specific doctor"""
            df_doctor = self.data.df[
                self.data.df["Doctor Name"].str.contains(
                    doctor_name, case=False, na=False
                )
            ]
            if df_doctor.empty:
                return f"Doctor '{doctor_name}' not found. Available doctors: {', '.join(self.data.get_doctor_list()[:10])}"

            return f"""
            📊 Doctor: {doctor_name}
            • Total Revenue: ${df_doctor["Total Revenue"].sum():,.0f}
            • Target Revenue: ${df_doctor["Target Revenue"].sum():,.0f}
            • Achievement: {(df_doctor["Total Revenue"].sum() / df_doctor["Target Revenue"].sum() * 100):.1f}%
            • Cases: {df_doctor["No. Cases"].sum():,.0f}
            • Avg No-Show: {df_doctor["No-Show %"].mean() * 100:.1f}%
            • Avg Retention: {df_doctor["Patient Retention %"].mean() * 100:.1f}%
            """

        @tool
        def analyze_root_cause(kpi_name: str) -> str:
            """Analyze root causes for KPI underperformance"""
            analysis = self.analytics.root_cause_analysis(kpi_name)
            if "error" in analysis:
                return analysis["error"]

            return f"""
            🔍 Root Cause Analysis: {kpi_name}
            • Change: {analysis["variance_pct"]:+.1f}%
            • Severity: {analysis["severity"].upper()}
            • Trend: {analysis["trend"]}
            
            Primary Drivers:
            {chr(10).join([f"  - {d['driver']}: {d['change_pct']:+.1f}%" for d in analysis["primary_drivers"]])}
            """

        @tool
        def compare_doctors(metric: str) -> str:
            """Compare doctors on a specific metric"""
            ranking = self.analytics.doctor_ranking(metric, top_n=5)
            if ranking.empty:
                return f"No data for metric: {metric}"

            result = f"🏆 Top Doctors by {metric}:\n"
            for _, row in ranking.iterrows():
                result += f"{row['rank']}. {row['Doctor Name']}: {row[metric]:,.0f}\n"
            return result

        @tool
        def get_kpi_trend(kpi_name: str, months: int = 6) -> str:
            """Get trend for a KPI"""
            df = self.data.df.sort_values("Date").tail(months)
            if kpi_name not in df.columns:
                return f"KPI '{kpi_name}' not found"

            trend_data = df[["YearMonth", kpi_name]].dropna()
            if len(trend_data) > 1:
                first = trend_data[kpi_name].iloc[0]
                last = trend_data[kpi_name].iloc[-1]
                change = ((last - first) / first * 100) if first != 0 else 0
                direction = (
                    "📈 Increasing"
                    if change > 5
                    else "📉 Decreasing"
                    if change < -5
                    else "➡️ Stable"
                )
            else:
                change = 0
                direction = "⚠️ Insufficient data"

            result = f"{direction} Trend for {kpi_name} (Change: {change:+.1f}%)\n\n"
            for _, row in trend_data.iterrows():
                result += f"{row['YearMonth']}: {row[kpi_name]:,.0f}\n"
            return result

        @tool
        def get_bu_summary(bu_name: str) -> str:
            """Get summary for a Business Unit"""
            df_bu = self.data.df[self.data.df["BU"] == bu_name]
            if df_bu.empty:
                return f"BU '{bu_name}' not found. Available: {', '.join(self.data.get_bu_list())}"

            return f"""
            🏢 BU Summary: {bu_name}
            • Total Revenue: ${df_bu["Total Revenue"].sum():,.0f}
            • Target: ${df_bu["Target Revenue"].sum():,.0f}
            • Achievement: {(df_bu["Total Revenue"].sum() / df_bu["Target Revenue"].sum() * 100):.1f}%
            • Total Cases: {df_bu["No. Cases"].sum():,.0f}
            • Avg No-Show: {df_bu["No-Show %"].mean() * 100:.1f}%
            • Avg Retention: {df_bu["Patient Retention %"].mean() * 100:.1f}%
            """

        tools = [
            get_doctor_performance,
            analyze_root_cause,
            compare_doctors,
            get_kpi_trend,
            get_bu_summary,
        ]

        if self.llm:
            prompt = ChatPromptTemplate.from_messages(
                [
                    (
                        "system",
                        """You are a professional OPD KPI Analytics Agent. Use the tools to answer questions about doctor performance, KPI trends, and root causes. Be concise and actionable.""",
                    ),
                    MessagesPlaceholder(variable_name="chat_history"),
                    ("human", "{input}"),
                    MessagesPlaceholder(variable_name="agent_scratchpad"),
                ]
            )

            agent = create_tool_calling_agent(self.llm, tools, prompt)
            self.agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=False)
            print("✅ Agent created successfully")

    def chat(self, user_input: str) -> str:
        """Process user message"""
        if self.agent_executor is None:
            return self._fallback_response(user_input)

        try:
            response = self.agent_executor.invoke(
                {"input": user_input, "chat_history": self.memory.chat_memory.messages}
            )
            self.memory.chat_memory.add_user_message(user_input)
            self.memory.chat_memory.add_ai_message(response["output"])
            return response["output"]
        except Exception as e:
            return f"Error: {str(e)}"

    def _fallback_response(self, user_input: str) -> str:
        """Fallback when LLM unavailable"""
        return f"""
        ⚠️ Agent running in basic mode (LLM not available).
        
        To enable full AI capabilities:
        1. Install Ollama: https://ollama.com
        2. Run: ollama pull {self.config.llm_model}
        3. Restart the agent
        
        Currently loaded: {len(self.data.df)} records, {len(self.data.get_doctor_list())} doctors
        """
