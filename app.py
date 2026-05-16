#!/usr/bin/env python
"""OPD KPI Intelligence Agent - Main Application"""

import gradio as gr
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.agents.kpi_agent import OPDKpiAgent
from src.config import config


def create_web_interface(agent: OPDKpiAgent):
    """Create Gradio web interface"""

    def respond(message, history):
        return agent.chat(message)

    with gr.Blocks(title="OPD KPI Intelligence Agent", theme=gr.themes.Soft()) as demo:
        gr.Markdown("""
        # 🏥 OPD KPI Intelligence Agent
        
        **AI-Powered Analytics for Healthcare Operations**
        
        Ask natural language questions about doctor performance, KPI trends, and root causes.
        """)

        with gr.Row():
            with gr.Column(scale=3):
                chatbot = gr.Chatbot(
                    label="Conversation",
                    height=500,
                    bubble_full_width=False,
                    show_copy_button=True,
                )
                msg = gr.Textbox(
                    label="Ask me anything",
                    placeholder="e.g., Show me Doctor Ahmed's performance, Why is Total Revenue down?, Compare doctors by retention...",
                    lines=2,
                )
                with gr.Row():
                    clear = gr.ClearButton([msg, chatbot])

            with gr.Column(scale=1):
                gr.Markdown("""
                ### 💡 Example Questions
                
                **Performance**
                - Show me Doctor Ahmed's performance
                - Compare doctors by Total Revenue
                
                **Analysis**
                - Why is Total Revenue down?
                - Analyze root cause for high No-Show
                
                **Trends**
                - Show me No-Show % trend
                - How has Patient Retention changed?
                
                **BU Summary**
                - Show ASH BU summary
                - Compare SMH vs HJH
                """)

        msg.submit(respond, [msg, chatbot], [chatbot]).then(lambda: "", None, [msg])

    return demo


def main():
    print("""
    ╔═════════════════════════════════════╗
    ║     OPD KPI Intelligence Agent      ║
    ╚═════════════════════════════════════╝
    """)

    # Initialize agent
    print("🚀 Initializing agent...")
    agent = OPDKpiAgent()

    # Create and launch interface
    demo = create_web_interface(agent)
    print(
        f"\n🌐 Launching web interface at http://{config.server_host}:{config.server_port}"
    )

    demo.launch(
        server_name=config.server_host, server_port=config.server_port, share=False
    )


if __name__ == "__main__":
    main()
