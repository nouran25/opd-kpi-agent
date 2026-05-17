# OPD KPI Intelligence Agent

An interactive Gradio application for analyzing Outpatient Department (OPD) KPI performance across doctors, business units, time periods, and operational drivers.

The app combines a structured analytics engine with a Groq-powered LangChain agent. Fast, high-confidence KPI questions are answered directly from the dataset, while broader natural-language questions can use the hosted LLM and the available analytical tools.

## Features

- Modern Gradio chat interface with KPI filters and prompt cards
- Doctor performance summaries and KPI justifications
- BU-level comparison across ASH, SMH, and HJH
- Root cause analysis using the knowledge-base relationship map
- KPI trend analysis over time
- Threshold questions such as doctors above a no-show rate
- Natural-language KPI resolution, for example `service leakage`, `PMS`, or `patient retention`
- Groq-hosted LLM support for faster agentic reasoning without running a model locally

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

- Initializes the data loader, analytics engine, Groq model, and LangChain agent
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
- Defines hosted LLM settings
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
- `langchain-groq`
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

### 3. Create a Groq API key

Create an API key from:

```text
https://console.groq.com/keys
```

Then set it in your terminal:

```powershell
$env:GROQ_API_KEY="your_api_key_here"
```

The default Groq model is `openai/gpt-oss-120b` with `medium` reasoning effort.

### 4. Run the app

```powershell
python app.py
```

For a persistent Windows user environment variable, use:

```powershell
setx GROQ_API_KEY "your_api_key_here"
```

Restart your terminal after using `setx`.

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
| `GROQ_API_KEY` | Required | Groq API key |
| `LLM_MODEL` | `openai/gpt-oss-120b` | Groq model name |
| `LLM_REASONING_EFFORT` | `medium` | Reasoning effort for GPT-OSS models: `low`, `medium`, or `high` |
| `TEMPERATURE` | `0.0` | LLM response randomness |
| `LLM_MAX_TOKENS` | `1024` | Maximum generated tokens |
| `LLM_TIMEOUT` | `60` | LLM request timeout in seconds |
| `LLM_MAX_RETRIES` | `2` | LLM request retry count |

Example:

```powershell
$env:GROQ_API_KEY="your_api_key_here"
$env:LLM_MODEL="openai/gpt-oss-120b"
$env:LLM_REASONING_EFFORT="medium"
$env:LLM_MAX_TOKENS="1024"
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
4. If the question is broader or ambiguous, the agent can use Groq and LangChain tools.

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

### Groq API key error

Confirm `GROQ_API_KEY` is set in the same terminal where you run the app:

```powershell
echo $env:GROQ_API_KEY
```

Then restart the app. You can also choose a different Groq model:

```powershell
$env:LLM_MODEL="openai/gpt-oss-120b"
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

### Data file not found

Make sure these files exist:

```text
data/OPD dataset.xlsx
data/Knowledge base.xlsx
```

Also confirm that the OPD workbook contains the `OPD_KPI_Dataset` sheet.

## Privacy Notes

This project runs the app locally, but Groq inference is hosted. Any text sent to the LLM API is processed by Groq, so avoid sending private healthcare, doctor, patient, or operational details unless that is acceptable for your environment.

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
