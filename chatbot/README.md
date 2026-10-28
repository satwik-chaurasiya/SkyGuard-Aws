# SkyGuard AI ChatBot 🛡️

**Privacy-First, Offline AI Chatbot for Data Analysis**

> Your data stays on YOUR machine. Always. No exceptions.

## 🔒 Privacy Guarantees

| Guarantee | How |
|-----------|-----|
| **No internet** | `TRANSFORMERS_OFFLINE=1`, `HF_HUB_OFFLINE=1` enforced at startup |
| **No data sharing** | Data is copied to an isolated `.scratch_data/` directory — never uploaded |
| **Secure delete on exit** | Files are overwritten with zeros then deleted on exit, quit, or Ctrl+C |
| **On-demand purge** | `purge` command wipes all data instantly |
| **Per-file delete** | `delete <id>` securely removes a single dataset |
| **Crash-safe** | Python `atexit` handler ensures cleanup even on unexpected termination |

## ⚡ Quick Start

```bash
cd chatbot
pip install -r requirements.txt
python chatbot.py
```

## 📖 Commands

| Command | Description |
|---------|-------------|
| `load <file>` | Load a dataset (CSV, Excel, JSON, Parquet, TSV) |
| `list` | Show all loaded datasets |
| `use <id>` | Switch active dataset |
| `info` | Show summary of active dataset |
| `delete <id>` | Securely delete a specific dataset |
| `purge` | Securely delete ALL data |
| `help` | Show help |
| `exit` / `quit` | Wipe all data and exit |

## 💬 Ask Questions

Just type natural language questions about your data:

- "How many rows are there?"
- "What columns does it have?"
- "Show the first 10 rows"
- "What is the average temperature?"
- "Are there missing values?"
- "Show correlations"
- "Any duplicates?"
- "What is the max pressure?"

## 🧠 AI Engine

The chatbot uses a **dual-engine** approach:

1. **Rule-based engine** — Handles common data questions instantly using pandas (always available)
2. **Local transformer model** (`google/flan-t5-base`) — Handles complex/open-ended questions offline

The transformer model downloads once on first run (~990MB) and runs locally forever after. If unavailable, the rule engine covers all essential queries.

## 📁 Supported Formats

- CSV (`.csv`)
- TSV (`.tsv`)
- Excel (`.xlsx`, `.xls`)
- JSON (`.json`)
- Parquet (`.parquet`)

## 🏗️ Architecture

```
chatbot/
├── chatbot.py        # Main entry point — CLI interface
├── ai_engine.py      # AI engine (local transformer + rule-based)
├── data_manager.py   # Privacy-first data manager (ingest, delete, purge)
├── requirements.txt  # Dependencies
├── README.md         # This file
└── sample_data.csv   # Sample dataset for testing
```
