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
  background:
    radial-gradient(circle at top left, rgba(20, 184, 166, 0.16), transparent 34rem),
    linear-gradient(135deg, #0b1117 0%, #10151d 48%, #16130f 100%) !important;
  color: #e7eef7 !important;
}
.gradio-container {
  min-height: 100vh;
}
.opd-shell {
  max-width: 1280px;
  margin: 0 auto;
  padding: 18px;
}
.opd-hero {
  border: 1px solid rgba(148, 163, 184, 0.22);
  background: linear-gradient(135deg, rgba(15, 23, 32, 0.94), rgba(24, 28, 31, 0.9));
  border-radius: 12px;
  padding: 20px 22px;
  box-shadow: 0 18px 48px rgba(0, 0, 0, 0.34);
  margin-bottom: 14px;
}
.opd-kicker {
  color: #5eead4;
  font-size: 13px;
  font-weight: 700;
  letter-spacing: .08em;
  margin-bottom: 8px;
  text-transform: uppercase;
}
.opd-title {
  color: #f8fafc;
  font-size: 30px;
  line-height: 1.15;
  font-weight: 760;
  margin: 0;
}
.opd-subtitle {
  color: #a9b6c8;
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
  border: 1px solid rgba(94, 234, 212, 0.2);
  background: rgba(15, 23, 42, 0.68);
  border-radius: 999px;
  color: #d9f7f2;
  font-size: 13px;
  font-weight: 650;
  padding: 6px 10px;
}
.opd-panel {
  border: 1px solid rgba(148, 163, 184, 0.18);
  background: rgba(15, 23, 32, 0.88);
  border-radius: 12px;
  padding: 14px;
  box-shadow: 0 14px 34px rgba(0, 0, 0, 0.24);
}
.opd-panel-title {
  color: #f8fafc;
  font-size: 14px;
  font-weight: 760;
  margin: 0 0 10px;
}
.opd-panel-note {
  color: #94a3b8;
  font-size: 12px;
  line-height: 1.45;
  margin: 8px 0 0;
}
#chatbot {
  border: 1px solid rgba(148, 163, 184, 0.18);
  border-radius: 12px;
  overflow: hidden;
  box-shadow: 0 18px 48px rgba(0, 0, 0, 0.28);
  background: rgba(8, 13, 20, 0.86) !important;
}
#prompt-box textarea {
  border-radius: 10px !important;
  font-size: 15px !important;
}
.opd-chip {
  width: 100%;
  border-radius: 10px !important;
  border: 1px solid rgba(148, 163, 184, 0.18) !important;
  background: rgba(15, 23, 32, 0.92) !important;
  color: #e7eef7 !important;
  justify-content: flex-start !important;
  text-align: left !important;
  min-height: 48px;
  box-shadow: 0 10px 24px rgba(0, 0, 0, 0.2);
  white-space: normal !important;
}
.opd-chip:hover {
  border-color: rgba(45, 212, 191, .7) !important;
  box-shadow: 0 14px 28px rgba(20, 184, 166, 0.14);
  transform: translateY(-1px);
}
.opd-input-row {
  border: 1px solid rgba(148, 163, 184, 0.18);
  background: rgba(15, 23, 32, 0.9);
  border-radius: 12px;
  padding: 12px;
  box-shadow: 0 18px 44px rgba(0, 0, 0, 0.26);
}
.opd-toolbar button {
  border-radius: 10px !important;
}
.gradio-container label,
.gradio-container .label-wrap,
.gradio-container .block-title,
.gradio-container .form label {
  color: #dbe7f5 !important;
}
.gradio-container input,
.gradio-container textarea,
.gradio-container select,
.gradio-container .wrap,
.gradio-container .container,
.gradio-container .block,
.gradio-container .form,
.gradio-container .input-container {
  background-color: rgba(9, 14, 22, 0.86) !important;
  border-color: rgba(148, 163, 184, 0.2) !important;
  color: #e7eef7 !important;
}
.gradio-container input::placeholder,
.gradio-container textarea::placeholder {
  color: #7f8ea3 !important;
}
.gradio-container button {
  background: rgba(19, 27, 38, 0.96) !important;
  border: 1px solid rgba(148, 163, 184, 0.22) !important;
  color: #e7eef7 !important;
}
.gradio-container button:hover {
  border-color: rgba(45, 212, 191, 0.62) !important;
  color: #ffffff !important;
}
.gradio-container button.primary,
#prompt-box button {
  background: linear-gradient(135deg, #14b8a6, #0f766e) !important;
  border-color: rgba(94, 234, 212, 0.52) !important;
  color: #ffffff !important;
}
.message.user {
  background: #123e3a !important;
  color: #effffb !important;
}
.message.bot {
  background: #151d28 !important;
  color: #e7eef7 !important;
}
.prose,
.prose * {
  color: #e7eef7 !important;
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

    def submit_message(
        message, history, bu_filter, doctor_filter, year_filter, kpi_filter
    ):
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
    print(f"Launching web interface at http://127.0.0.1:{server_port}")

    launch_kwargs = {
        "server_name": config.server_host,
        "server_port": server_port,
        "share": False,
        "theme": gr.themes.Soft(
            primary_hue="teal",
            neutral_hue="slate",
        ),
        "css": APP_CSS,
    }
    try:
        demo.launch(**launch_kwargs)
    except ValueError as exc:
        if "localhost is not accessible" not in str(exc):
            raise
        print("Localhost check failed; retrying with Gradio's share link.")
        demo.launch(**{**launch_kwargs, "share": True})


if __name__ == "__main__":
    main()
