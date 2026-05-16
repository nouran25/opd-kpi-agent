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
    "Why is Dr. Ahmed's revenue below target in Q1 2024?",
    "What are the root causes of high service leakage in HJH?",
    "Compare patient retention across all BUs in 2023",
    "Which doctors have a no-show rate above 20%?",
    "Justify Dr. Mahmoud's PMS performance in ASH",
]

APP_CSS = """
body, .gradio-container {
  background: #f6f8fb !important;
}
.opd-shell {
  max-width: 1180px;
  margin: 0 auto;
  padding: 20px 18px 10px;
}
.opd-hero {
  border: 1px solid #dfe6ef;
  background: linear-gradient(135deg, #ffffff 0%, #eef7f6 55%, #eef3ff 100%);
  border-radius: 18px;
  padding: 22px 24px;
  box-shadow: 0 18px 45px rgba(22, 32, 51, 0.08);
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
  font-size: 32px;
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
#chatbot {
  border: 1px solid #dfe6ef;
  border-radius: 18px;
  overflow: hidden;
  box-shadow: 0 18px 45px rgba(22, 32, 51, 0.07);
}
#prompt-box textarea {
  border-radius: 14px !important;
  font-size: 15px !important;
}
.opd-chip {
  width: 100%;
  border-radius: 14px !important;
  border: 1px solid #dfe6ef !important;
  background: #ffffff !important;
  color: #162033 !important;
  justify-content: flex-start !important;
  text-align: left !important;
  min-height: 46px;
  box-shadow: 0 8px 20px rgba(22, 32, 51, 0.05);
}
.opd-chip:hover {
  border-color: rgba(15, 118, 110, .55) !important;
  box-shadow: 0 12px 26px rgba(15, 118, 110, 0.11);
  transform: translateY(-1px);
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

    def respond(message, history):
        history = history or WELCOME_HISTORY.copy()
        if not message or not message.strip():
            return history

        response = agent.chat(message)
        return history + [
            {"role": "user", "content": message},
            {"role": "assistant", "content": response},
        ]

    def submit_message(message, history):
        return "", respond(message, history)

    def submit_recommendation(question, history):
        return "", respond(question, history)

    def clear_conversation():
        agent.chat_history = []
        return "", WELCOME_HISTORY.copy()

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
                </section>
                """
            )

            chatbot = gr.Chatbot(
                value=WELCOME_HISTORY.copy(),
                height=560,
                layout="bubble",
                buttons=["copy"],
                show_label=False,
                elem_id="chatbot",
            )

            gr.Markdown("**Recommended questions**")
            recommendation_buttons = [
                gr.Button(f"- {question}", elem_classes=["opd-chip"])
                for question in RECOMMENDED_QUESTIONS
            ]

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
                clear = gr.Button("Clear")

        msg.submit(submit_message, [msg, chatbot], [msg, chatbot])
        for button, question in zip(recommendation_buttons, RECOMMENDED_QUESTIONS):
            button.click(
                submit_recommendation,
                inputs=[gr.State(question), chatbot],
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
