#!/usr/bin/env python
"""OPD KPI Intelligence Agent - Main Application."""

import socket
import sys
from pathlib import Path

import gradio as gr

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

sys.path.insert(0, str(Path(__file__).parent))

from src.agents.kpi_agent import OPDKpiAgent
from src.config import config


WELCOME_MESSAGE = """Welcome to the OPD KPI Intelligence Agent.

I have access to the OPD dataset for 2023-2025 across ASH, SMH, and HJH business units, covering 11 doctors and 24 KPIs.

You can ask me anything, for example:

- "Why is Dr. Ahmed's revenue below target in Q1 2024?"
- "What are the root causes of high service leakage in HJH?"
- "Compare patient retention across all BUs in 2023"
- "Which doctors have a no-show rate above 20%?"
- "Justify Dr. Mahmoud's PMS performance in ASH"
"""

WELCOME_HISTORY = [{"role": "assistant", "content": WELCOME_MESSAGE}]

RECOMMENDED_QUESTIONS = [
    {
        "category": "Doctor",
        "label": "Ahmed KPI justification",
        "question": "Show me Doctor Ahmed's performance, and give me justifications for his KPIs",
    },
    {
        "category": "Root Cause",
        "label": "HJH service leakage",
        "question": "What are the root causes of high service leakage in HJH?",
    },
    {
        "category": "BU Comparison",
        "label": "Retention by BU",
        "question": "Compare patient retention across all BUs in 2023",
    },
    {
        "category": "Threshold",
        "label": "High no-show doctors",
        "question": "Which doctors have a no-show rate above 20%?",
    },
    {
        "category": "PMS",
        "label": "Mahmoud PMS in ASH",
        "question": "Justify Dr. Mahmoud's PMS performance in ASH",
    },
]

APP_CSS = """
body, .gradio-container {
  background: #f6f8fb !important;
}
.opd-shell {
  max-width: 1280px;
  margin: 0 auto;
  padding: 18px;
}
.opd-hero {
  border: 1px solid #dfe6ef;
  background: #ffffff;
  border-radius: 14px;
  padding: 20px 22px;
  box-shadow: 0 14px 35px rgba(22, 32, 51, 0.07);
  margin-bottom: 14px;
}
.opd-kicker {
  color: #0f766e;
  font-size: 13px;
  font-weight: 700;
  letter-spacing: .08em;
  margin-bottom: 8px;
  text-transform: uppercase;
}
.opd-title {
  color: #162033;
  font-size: 30px;
  line-height: 1.15;
  font-weight: 760;
  margin: 0;
}
.opd-subtitle {
  color: #607089;
  font-size: 15px;
  line-height: 1.55;
  max-width: 780px;
  margin-top: 8px;
}
.opd-badges {
  display: flex;
  flex-wrap: wrap;
  gap: 8px;
  margin-top: 14px;
}
.opd-badge {
  border: 1px solid #dfe6ef;
  background: #f8fafc;
  border-radius: 999px;
  color: #334155;
  font-size: 13px;
  font-weight: 650;
  padding: 6px 10px;
}
.opd-panel {
  border: 1px solid #dfe6ef;
  background: #ffffff;
  border-radius: 14px;
  padding: 14px;
  box-shadow: 0 12px 28px rgba(22, 32, 51, 0.055);
}
.opd-panel-title {
  color: #162033;
  font-size: 14px;
  font-weight: 760;
  margin: 0 0 10px;
}
.opd-panel-note {
  color: #64748b;
  font-size: 12px;
  line-height: 1.45;
  margin: 8px 0 0;
}
#chatbot {
  border: 1px solid #dfe6ef;
  border-radius: 14px;
  overflow: hidden;
  box-shadow: 0 12px 30px rgba(22, 32, 51, 0.055);
}
#prompt-box textarea {
  border-radius: 12px !important;
  font-size: 15px !important;
}
.opd-chip {
  width: 100%;
  border-radius: 12px !important;
  border: 1px solid #dfe6ef !important;
  background: #ffffff !important;
  color: #162033 !important;
  justify-content: flex-start !important;
  text-align: left !important;
  min-height: 48px;
  box-shadow: 0 8px 20px rgba(22, 32, 51, 0.05);
  white-space: normal !important;
}
.opd-chip:hover {
  border-color: rgba(15, 118, 110, .55) !important;
  box-shadow: 0 12px 26px rgba(15, 118, 110, 0.11);
  transform: translateY(-1px);
}
.opd-input-row {
  border: 1px solid #dfe6ef;
  background: #ffffff;
  border-radius: 14px;
  padding: 12px;
  box-shadow: 0 12px 30px rgba(22, 32, 51, 0.055);
}
.opd-toolbar button {
  border-radius: 12px !important;
}
footer {
  display: none !important;
}
"""


def create_web_interface(agent: OPDKpiAgent):
    """Create Gradio web interface."""

    def apply_filters(message, bu_filter, doctor_filter, year_filter, kpi_filter):
        filters = []
        if bu_filter and bu_filter != "All BUs":
            filters.append(f"BU: {bu_filter}")
        if doctor_filter and doctor_filter != "All doctors":
            filters.append(f"Doctor: {doctor_filter}")
        if year_filter and year_filter != "All years":
            filters.append(f"Year: {year_filter}")
        if kpi_filter and kpi_filter != "All KPIs":
            filters.append(f"KPI: {kpi_filter}")

        if not filters:
            return message
        return f"{message}\n\nUse these selected filters: {', '.join(filters)}."

    def respond(message, history, bu_filter, doctor_filter, year_filter, kpi_filter):
        history = history or WELCOME_HISTORY.copy()
        if not message or not message.strip():
            return history

        agent_message = apply_filters(
            message,
            bu_filter,
            doctor_filter,
            year_filter,
            kpi_filter,
        )
        response = agent.chat(agent_message)
        return history + [
            {"role": "user", "content": message},
            {"role": "assistant", "content": response},
        ]

    def submit_message(message, history, bu_filter, doctor_filter, year_filter, kpi_filter):
        return "", respond(
            message,
            history,
            bu_filter,
            doctor_filter,
            year_filter,
            kpi_filter,
        )

    def submit_recommendation(
        question,
        history,
        bu_filter,
        doctor_filter,
        year_filter,
        kpi_filter,
    ):
        return "", respond(
            question,
            history,
            bu_filter,
            doctor_filter,
            year_filter,
            kpi_filter,
        )

    def clear_conversation():
        agent.chat_history = []
        return "", WELCOME_HISTORY.copy()

    doctor_choices = ["All doctors"] + agent.data.get_doctor_list()
    bu_choices = ["All BUs"] + agent.data.get_bu_list()
    year_choices = ["All years"] + [
        str(year)
        for year in sorted(agent.data.df["Year"].dropna().astype(int).unique())
    ]
    kpi_choices = ["All KPIs"] + agent._available_kpi_columns()

    with gr.Blocks(title="OPD KPI Intelligence Agent") as demo:
        with gr.Column(elem_classes=["opd-shell"]):
            gr.HTML(
                """
                <section class="opd-hero">
                  <div class="opd-kicker">Healthcare Operations Analytics</div>
                  <h1 class="opd-title">OPD KPI Intelligence Agent</h1>
                  <div class="opd-subtitle">
                    Ask focused questions about doctors, business units, KPI trends,
                    root causes, leakage, retention, no-shows, and target achievement.
                  </div>
                  <div class="opd-badges">
                    <span class="opd-badge">2023-2025 Dataset</span>
                    <span class="opd-badge">ASH / SMH / HJH</span>
                    <span class="opd-badge">11 Doctors</span>
                    <span class="opd-badge">24 KPIs</span>
                    <span class="opd-badge">Knowledge-base driven</span>
                  </div>
                </section>
                """
            )

            with gr.Row(equal_height=False):
                with gr.Column(scale=1, min_width=260):
                    gr.HTML(
                        """
                        <div class="opd-panel">
                          <div class="opd-panel-title">Analysis filters</div>
                          <div class="opd-panel-note">
                            Optional. Pick filters here, then type naturally.
                          </div>
                        </div>
                        """
                    )
                    bu_filter = gr.Dropdown(
                        choices=bu_choices,
                        value="All BUs",
                        label="Business Unit",
                    )
                    doctor_filter = gr.Dropdown(
                        choices=doctor_choices,
                        value="All doctors",
                        label="Doctor",
                    )
                    year_filter = gr.Dropdown(
                        choices=year_choices,
                        value="All years",
                        label="Year",
                    )
                    kpi_filter = gr.Dropdown(
                        choices=kpi_choices,
                        value="All KPIs",
                        label="KPI",
                    )

                    gr.HTML('<div class="opd-panel-title">Prompt cards</div>')
                    recommendation_buttons = []
                    for item in RECOMMENDED_QUESTIONS:
                        recommendation_buttons.append(
                            gr.Button(
                                f"{item['category']} | {item['label']}",
                                elem_classes=["opd-chip"],
                            )
                        )

                with gr.Column(scale=3, min_width=520):
                    chatbot = gr.Chatbot(
                        value=WELCOME_HISTORY.copy(),
                        height=600,
                        layout="bubble",
                        buttons=["copy"],
                        show_label=False,
                        elem_id="chatbot",
                    )

                    with gr.Column(elem_classes=["opd-input-row"]):
                        msg = gr.Textbox(
                            label="Ask a KPI question",
                            placeholder="Ask about doctor performance, BU trends, revenue leakage, retention, no-shows...",
                            lines=1,
                            max_lines=4,
                            autofocus=True,
                            submit_btn="Send",
                            elem_id="prompt-box",
                        )
                        with gr.Row(elem_classes=["opd-toolbar"]):
                            clear = gr.Button("Clear conversation")

        filter_inputs = [bu_filter, doctor_filter, year_filter, kpi_filter]
        msg.submit(
            submit_message,
            [msg, chatbot, *filter_inputs],
            [msg, chatbot],
        )
        for button, item in zip(recommendation_buttons, RECOMMENDED_QUESTIONS):
            button.click(
                submit_recommendation,
                inputs=[gr.State(item["question"]), chatbot, *filter_inputs],
                outputs=[msg, chatbot],
            )
        clear.click(clear_conversation, outputs=[msg, chatbot])

    return demo


def find_available_port(start_port: int, attempts: int = 20) -> int:
    """Find an open localhost port for Gradio."""
    for port in range(start_port, start_port + attempts):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.settimeout(0.2)
            if sock.connect_ex(("127.0.0.1", port)) != 0:
                return port
    return start_port


def main():
    print("OPD KPI Intelligence Agent")
    print("Initializing agent...")
    agent = OPDKpiAgent()

    demo = create_web_interface(agent)
    server_port = find_available_port(config.server_port)
    print(f"Launching web interface at http://{config.server_host}:{server_port}")

    demo.launch(
        server_name=config.server_host,
        server_port=server_port,
        share=False,
        theme=gr.themes.Soft(),
        css=APP_CSS,
    )


if __name__ == "__main__":
    main()
