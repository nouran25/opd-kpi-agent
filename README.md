# OPD KPI Intelligence Agent

An interactive Gradio application for analyzing Outpatient Department (OPD) KPI performance across doctors, business units, time periods, and operational drivers.

The app combines a structured analytics engine with an Ollama-powered LangChain agent. Fast, high-confidence KPI questions are answered directly from the dataset, while broader natural-language questions can use the local LLM and the available analytical tools.

## Features

- Modern Gradio chat interface with KPI filters and prompt cards
- Doctor performance summaries and KPI justifications
- BU-level comparison across ASH, SMH, and HJH
- Root cause analysis using the knowledge-base relationship map
- KPI trend analysis over time
- Threshold questions such as doctors above a no-show rate
- Natural-language KPI resolution, for example `service leakage`, `PMS`, or `patient retention`
- Local Ollama support for agentic reasoning without sending data to external APIs

## Example Questions

```text
Show me Doctor Ahmed's performance, and give me justifications for his KPIs
What are the root causes of high service leakage in HJH?
Compare patient retention across all BUs in 2023
Which doctors have a no-show rate above 20%?
Justify Dr. Mahmoud's PMS performance in ASH
Show the trend for service leakage in HJH
```

## Project Structure

```text
OPD Agent/
+-- app.py
+-- requirements.txt
+-- setup.py
+-- README.md
+-- data/
|   +-- OPD dataset.xlsx
|   +-- Knowledge base.xlsx
+-- src/
    +-- config.py
    +-- agents/
    |   +-- kpi_agent.py
    +-- analytics/
    |   +-- engine.py
    +-- data/
        +-- loader.py
```

## Code Overview

### `app.py`

Main Gradio application entry point.

Responsibilities:

- Builds the web interface
- Displays the welcome message, dataset badges, prompt cards, and chat panel
- Adds optional filters for BU, doctor, year, and KPI
- Sends user questions to the KPI agent
- Launches the local Gradio server

Run this file to start the app:

```powershell
python app.py
```

### `src/agents/kpi_agent.py`

Main intelligence layer for the application.

Responsibilities:

- Initializes the data loader, analytics engine, Ollama model, and LangChain agent
- Routes simple structured questions to fast dataframe-based answers
- Uses LangChain tools for more open-ended agent responses
- Formats professional responses with executive readouts, evidence, drivers, and recommendations
- Resolves doctors, BUs, years, KPIs, and threshold filters from natural-language questions

Important behavior:

- The agent does not rely on one-off hard-coded answers.
- It resolves KPI names using dataset columns and the knowledge base.
- It answers common analytical questions directly for speed.
- It falls back to the LLM only when needed.

### `src/data/loader.py`

Data ingestion and KPI metadata layer.

Responsibilities:

- Loads `data/OPD dataset.xlsx`
- Loads all sheets from `data/Knowledge base.xlsx`
- Prepares date columns such as `Date`, `Year`, `Month_Num`, and `YearMonth`
- Creates derived metrics:
  - `Revenue_Achievement_%`
  - `Cases_Achievement_%`
  - `Revenue_per_Case`
  - `Leakage_Impact_%`
- Builds a KPI alias index so natural wording can map to real dataset columns
- Provides helper methods for doctors, BUs, KPI metadata, playbooks, and KPI relationships

### `src/analytics/engine.py`

Statistical and operational analytics layer.

Responsibilities:

- Performs root cause analysis
- Compares current and previous periods
- Aggregates metrics correctly based on type:
  - sums for volume and financial metrics
  - averages for percentages and rates
- Detects anomalies using Z-scores
- Ranks doctors by KPI
- Uses the knowledge-base relationship map to identify KPI drivers

### `src/config.py`

Central configuration file.

Responsibilities:

- Defines model settings
- Defines Ollama connection settings
- Defines data paths
- Defines default analysis thresholds
- Defines Gradio host and port

You can override many settings with environment variables.

### `requirements.txt`

Python dependencies needed to run the app.

Main libraries:

- `gradio`
- `pandas`
- `numpy`
- `scipy`
- `openpyxl`
- `langchain`
- `langchain-ollama`
- `chromadb`

### `setup.py`

Optional Python package setup file. Useful if you want to install the project as a local package.

## Data Files

The app expects these files:

```text
data/OPD dataset.xlsx
data/Knowledge base.xlsx
```

The OPD dataset should include the main KPI table. The current loader expects a sheet named:

```text
OPD_KPI_Dataset
```

The knowledge base can include KPI definitions, relationship maps, and investigation playbooks. The agent uses it to resolve KPI meaning and explain root causes.

## Setup

### 1. Create and activate a virtual environment

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### 2. Install dependencies

```powershell
pip install -r requirements.txt
```

### 3. Install Ollama

Download and install Ollama from:

```text
https://ollama.com
```

### 4. Pull the configured model

The default model is `qwen2.5:7b`.

```powershell
ollama pull qwen2.5:7b
```

### 5. Start Ollama

Make sure Ollama is running before launching the app.

You can test it with:

```powershell
ollama list
```

### 6. Run the app

```powershell
python app.py
```

By default, the app launches on:

```text
http://0.0.0.0:7860
```

On your local machine, open:

```text
http://127.0.0.1:7860
```

## Configuration

The default configuration is in `src/config.py`.

Common environment variables:

| Variable | Default | Description |
|---|---:|---|
| `LLM_MODEL` | `qwen2.5:7b` | Ollama model name |
| `OLLAMA_BASE_URL` | `http://localhost:11434` | Ollama server URL |
| `TEMPERATURE` | `0.1` | LLM response randomness |
| `LLM_NUM_CTX` | `2048` | Ollama context window |
| `LLM_NUM_PREDICT` | `512` | Maximum generated tokens |
| `LLM_NUM_THREAD` | CPU count | CPU threads used by Ollama |
| `LLM_NUM_GPU` | `0` | GPU layers used by Ollama |
| `LLM_KEEP_ALIVE` | `30m` | How long Ollama keeps the model loaded |

Example:

```powershell
$env:LLM_MODEL="qwen2.5:7b"
$env:LLM_NUM_CTX="2048"
$env:LLM_NUM_PREDICT="512"
python app.py
```

## How the Agent Answers Quickly

The app uses a hybrid approach:

1. The user asks a natural-language question.
2. The agent extracts structured information:
   - doctor
   - business unit
   - year
   - KPI
   - threshold
   - requested operation
3. If the question maps clearly to a known analytics operation, the app answers directly from the dataframe.
4. If the question is broader or ambiguous, the agent can use Ollama and LangChain tools.

This keeps simple questions fast while preserving agent-style reasoning for more complex requests.

## Knowledge-Base Driven Analysis

The agent uses the knowledge base to avoid inventing KPI relationships.

It can read:

- KPI definitions
- KPI formulas
- Parent and child KPI relationships
- Driver weights
- Investigation steps
- Recommended actions

For example, root cause analysis for `Service Leakage %` uses the relationship map to identify drivers such as missed opportunities, workflow compliance, and follow-up behavior when those fields exist in the knowledge base.

## Troubleshooting

### Import error from LangChain

If you see an import error, reinstall dependencies:

```powershell
pip install -r requirements.txt --upgrade
```

### Ollama connection error

Confirm Ollama is running:

```powershell
ollama list
```

Then confirm the model exists:

```powershell
ollama pull qwen2.5:7b
```

### Very slow answers

The app should answer structured KPI questions quickly. If a question is slow, it may have fallen back to the LLM.

Try making the question more explicit:

```text
Compare Patient Retention % across all BUs in 2023
```

instead of:

```text
Tell me about retention
```

### Ollama crashes

If Ollama crashes because of GPU issues, force CPU mode:

```powershell
setx CUDA_VISIBLE_DEVICES -1
setx GPU_DEVICE_ORDINAL -1
```

Restart your terminal after running those commands.

### Data file not found

Make sure these files exist:

```text
data/OPD dataset.xlsx
data/Knowledge base.xlsx
```

Also confirm that the OPD workbook contains the `OPD_KPI_Dataset` sheet.

## Privacy Notes

This project is designed for local execution. When using local Ollama, dataset content stays on your machine.

Do not publish private healthcare, doctor, patient, or operational data to a public repository. If the Excel files contain confidential information, keep the repository private or remove/anonymize the files before sharing.

## Development Notes

Useful checks:

```powershell
python -m py_compile app.py src\agents\kpi_agent.py src\analytics\engine.py src\data\loader.py src\config.py
```

Run the app:

```powershell
python app.py
```

## License

Add a license before publishing the repository publicly. If the dataset is proprietary or confidential, keep the project private.
