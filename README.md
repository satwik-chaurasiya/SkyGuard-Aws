# SkyGuard AI — National AWS Intelligence & Disaster Early Warning System
### Ministry of Earth Sciences (MoES) | India Meteorological Department (IMD)
*Smart India Hackathon 2026 (Problem Statement 26073)*

---

## 1. Executive Summary & What is SkyGuard AI?

SkyGuard AI is an advanced, fully offline desktop and command-line intelligent monitoring platform designed for Automatic Weather Stations (AWS). It combines unsupervised machine learning (Isolation Forest), explainable AI (XAI), Optical Fibre Distributed Acoustic Sensing (DAS), solar microgrid telemetry, multi-station mesh networking, and a privacy-first offline AI assistant to ensure robust, real-time meteorological monitoring and natural disaster early warning.

---

## 2. Core Features & Architecture

### 🤖 1. AI/ML Intelligent Anomaly Detection
- **Isolation Forest Model:** Unsupervised outlier detection trained on synthetic meteorological data with realistic multivariate physical covariance (temperature, pressure, humidity).
- **Multi-Rule Checks:** Point-level anomalies, frozen-streak detection, and communication dropouts.
- **Dynamic Re-calibration:** Adapts baseline models across 12 diverse geographical station profiles.

### 🔍 2. Explainable AI (XAI) & Self-Healing Imputation
- **SHAP-Style Feature Attribution:** Quantifies the percentage contribution of each metric (Temperature, Pressure, Humidity) to flagged anomalies.
- **Self-Healing Estimator:** Computes suggested corrected readings based on historical rolling baselines and spatial-temporal imputation.
- **Sensor Health & RUL (Remaining Useful Life):** Tracks degradation index and maintenance schedules.

### 🗺️ 3. Multi-Station Regional Profiles (12 Stations)
Simulates diverse climatic zones across India with station-specific elevations and optical fibre backhaul distances:
- **AWS-101:** New Delhi / Safdarjung (Subtropical Plains)
- **AWS-204:** Leh / High-Altitude (High-Altitude Cold Desert)
- **AWS-308:** Mumbai / Colaba Marine (Coastal Maritime)
- **AWS-415:** Jaisalmer / Thar Desert (Arid Desert Zone)
- **AWS-512:** Kolkata / Alipore Marine (Gangetic Delta)
- **AWS-620:** Bengaluru / Hebbal Deccan (Deccan Plateau)
- **AWS-714:** Chennai / Nungambakkam Coastal (Tropical East Coast)
- **AWS-822:** Guwahati / Brahmaputra Valley (North-East River Basin)
- **AWS-930:** Srinagar / Kashmir Valley (Himalayan Alpine)
- **AWS-104:** Bhopal / Central Plateau (Vindhya Plateau)
- **AWS-116:** Shimla / Western Himalayas (Mountain Ridge)
- **AWS-128:** Port Blair / Andaman Islands (Island Maritime)

### 🧶 4. Optical Fibre Distributed Acoustic Sensing (DAS) & Backhaul
- Monitors underground optical fibre health linking weather stations.
- Tracks OTDR attenuation, signal-to-noise ratio (SNR), and acoustic vibration/seismic strain anomalies.

### 🌪️ 5. Natural Disaster Early Warning & XAI
- Real-time risk assessment and automated alerts for extreme weather events: Cloudbursts, Cyclones, Heatwaves, Flash Floods, and Landslides.

### ⚡ 6. Solar Microgrid Power Status
- Monitors autonomous solar-battery power generation, battery State of Charge (SoC), voltage stability, and microgrid health to ensure uninterrupted AWS uptime in remote locations.

### 🌐 7. Multi-Station Mesh Network Map
- Visualizes robust peer-to-peer radio/fibre mesh connectivity between regional IMD stations, ensuring fault-tolerant data routing during terrestrial link failures.

### 🚨 8. Automated Webhooks & NDMA Dispatch Log
- Instantaneous dispatch log and mock webhook integration for alerting the National Disaster Management Authority (NDMA) and regional disaster response cells upon critical severity anomalies.

### 💬 9. Privacy-First Offline AI Assistant / Chatbot (`chatbot/`)
- Fully offline, air-gapped data analysis assistant (`google/flan-t5-base` + Pandas).
- Supports CSV, TSV, Excel (`.xlsx`, `.xls`), JSON, and Parquet datasets.
- **Strict Privacy Guarantees:** Data is copied to isolated scratch storage (`.chatbot_scratch/`), environment variables enforce complete network isolation (`TRANSFORMERS_OFFLINE=1`, `HF_HUB_OFFLINE=1`), and a robust `atexit` handler securely overwrites all temporary data files with zeros and deletes them upon exit.

---

## 3. Project Structure

```
SkyGuard-Aws/
├── main.py               # Main GUI application (Real-time monitor, Mesh map, Chatbot tab)
├── aws_simulator.py      # AWS sensor simulation, station profiles, fibre & solar telemetry
├── anomaly_detector.py   # Isolation Forest ML model, XAI feature attribution, imputation
├── chatbot/              # Standalone privacy-first offline AI data analyst CLI
│   ├── chatbot.py        # CLI interface for dataset querying
│   ├── ai_engine.py      # Dual-engine rule-based + local transformer model
│   ├── data_manager.py   # Secure data ingest, purge, and zero-wipe exit handler
│   └── sample_data.csv   # Sample dataset
├── tests/                # Unit tests for simulator and anomaly detector
├── DEMO_SCRIPT.md        # Word-for-word judge presentation script
├── requirements.txt      # Python dependencies
└── build_exe.bat         # PyInstaller build script for standalone Windows EXE
```

---

## 4. Installation & Quick Start

### Prerequisites
- Python 3.10+ installed.

### Setup & Run GUI Application
```cmd
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
python main.py
```

### Run Standalone Chatbot CLI
```cmd
cd chatbot
pip install -r requirements.txt
python chatbot.py
```

---

## 5. Building the Standalone Executable (.EXE)

To package SkyGuard AI into a self-contained executable for offline deployment:
```cmd
build_exe.bat
```
This generates `dist\SkyGuardAI.exe`, which can run on any Windows machine without requiring Python or internet connectivity.

---

## 6. Judge Demonstration Guide (1-Minute Sequence)

1. **Launch App:** Run `python main.py` or open `SkyGuardAI.exe`.
2. **Select Station:** Choose any regional station profile from the dropdown (e.g., New Delhi, Leh, Mumbai).
3. **Start Monitoring:** Click **START MONITORING** to stream live telemetry.
4. **Inject Fault:** Click **Temperature Spike** or **Frozen Sensor** to trigger an anomaly.
5. **Inspect XAI:** Observe the 🔴 ANOMALY status, High Severity, Confidence (~95%), SHAP-style explanation, and Suggested Corrected Reading.
6. **Explore Tabs:** Check out the **Multi-Station Mesh Network Map** and **Privacy-First AI Assistant** tabs.
7. **Demo Mode:** Click **🎬 DEMO MODE** for an automated walk-through of the complete anomaly lifecycle.
