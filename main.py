"""
main.py — SkyGuard AI
Offline desktop demo for Smart India Hackathon 2026 (PS 26073)
Ministry of Earth Sciences / India Meteorological Department
AI/ML Anomaly Detection, Optical Fibre DAS, Natural Disaster XAI, Solar Microgrid, RUL, Mesh Network, and Webhooks.
"""
import tkinter as tk
from tkinter import ttk, scrolledtext, messagebox, filedialog
from collections import deque
from datetime import datetime
import threading
import atexit
import shutil
import hashlib
import json
from pathlib import Path

# Try importing ML/plotting
try:
    import numpy as np
    from sklearn.ensemble import IsolationForest
except Exception:
    np = None
    IsolationForest = None

try:
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib.pyplot import Figure
    from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
    HAS_MPL = True
except Exception:
    HAS_MPL = False
    Figure = None
    FigureCanvasTkAgg = None

from aws_simulator import AWSSimulator, STATION_PROFILES
from anomaly_detector import AnomalyDetector

# ------------------------------------------------------------------
# Style & Theme Constants
# ------------------------------------------------------------------
BG = "#000000"
CARD = "#0a0f1e"
CARD2 = "#1e293b"
ACCENT = "#3b82f6"
ACCENT2 = "#10b981"
ACCENT3 = "#f97316"
GOOD = "#22c55e"
BAD = "#ef4444"
CRITICAL = "#dc2626"
TEXT = "#e2e8f0"
TEXT2 = "#94a3b8"
PRIMARY = "#6366f1"
WARNING = "#eab308"
SUCCESS = "#16a34a"
DANGER = "#dc2626"


def set_dark_style():
    """Configures complete dark theme across all TTK widgets, tabs, frames, and dropdowns."""
    s = ttk.Style()
    s.theme_use("clam")

    # Global widget styling
    s.configure(".", background=BG, foreground=TEXT)
    s.configure("TFrame", background=BG)
    s.configure("TLabelframe", background=BG, foreground=TEXT, bordercolor="#1e293b", darkcolor="#1e293b", lightcolor="#1e293b")
    s.configure("TLabelframe.Label", background=BG, foreground=ACCENT, font=("Segoe UI Semibold", 9, "bold"))

    # Notebook tabs styling - eliminates white tab header strip and tab boundaries
    s.configure("TNotebook", background=BG, borderwidth=0, tabmargins=[0, 0, 0, 0], lightcolor=BG, darkcolor=BG, bordercolor=BG)
    s.configure("TNotebook.client", background=BG, bordercolor=BG, lightcolor=BG, darkcolor=BG)
    s.configure("TNotebook.Tab", background="#0a0f1e", foreground=TEXT2, font=("Segoe UI Semibold", 10), padding=[16, 7],
                bordercolor="#1e293b", lightcolor="#1e293b", darkcolor="#1e293b")
    s.map("TNotebook.Tab",
          background=[("selected", ACCENT), ("active", "#1e293b")],
          foreground=[("selected", "#ffffff"), ("active", "#ffffff")])

    # Checkbutton styling
    s.configure("TCheckbutton", background=BG, foreground=TEXT, font=("Calibri", 10), indicatorsize=12)
    s.map("TCheckbutton",
          background=[("active", BG), ("selected", BG), ("!disabled", BG)],
          foreground=[("active", TEXT), ("selected", TEXT), ("!disabled", TEXT)],
          indicatorcolor=[("selected", ACCENT), ("!selected", CARD2)])

    # Combobox styling
    s.configure("TCombobox", fieldbackground="#0a0f1e", background="#1e293b", foreground=TEXT,
                arrowcolor=TEXT, bordercolor="#1e293b", darkcolor="#1e293b", lightcolor="#1e293b")
    s.map("TCombobox",
          fieldbackground=[("readonly", "#0a0f1e"), ("active", "#0a0f1e")],
          selectbackground=[("readonly", "#1e293b")],
          selectforeground=[("readonly", TEXT)])


CHAT_DATA_DIR = Path(__file__).parent / ".chatbot_scratch"


class PrivacyDataManager:
    """Privacy-first local dataset manager with secure wipe on exit."""
    def __init__(self):
        self._datasets = {}
        self._ensure_dir()
        atexit.register(self.purge_all)

    def _ensure_dir(self):
        CHAT_DATA_DIR.mkdir(parents=True, exist_ok=True)

    def ingest_df(self, df, name="Live_AWS_Stream.csv"):
        file_hash = hashlib.sha256(name.encode() + str(len(df)).encode()).hexdigest()[:12]
        dest = CHAT_DATA_DIR / f"{file_hash}.csv"
        df.to_csv(dest, index=False)
        meta = {
            "id": file_hash,
            "name": name,
            "path": str(dest),
            "rows": len(df),
            "cols": len(df.columns),
            "columns": list(df.columns),
        }
        self._datasets[file_hash] = meta
        return meta

    def ingest_file(self, src_path):
        import pandas as pd
        src = Path(src_path)
        if not src.exists():
            raise FileNotFoundError(f"File not found: {src_path}")
        df = pd.read_csv(src)
        return self.ingest_df(df, src.name)

    def get_dataframe(self, dataset_id):
        import pandas as pd
        meta = self._datasets.get(dataset_id)
        if not meta:
            raise KeyError("Dataset not found")
        return pd.read_csv(meta["path"])

    def purge_all(self):
        if CHAT_DATA_DIR.exists():
            shutil.rmtree(CHAT_DATA_DIR, ignore_errors=True)
        self._datasets.clear()


class ChatAIEngine:
    """Offline rule-based assistant for querying weather and telemetry datasets."""
    def answer(self, question, df):
        if df is None or len(df) == 0:
            return "No data available. Please load a dataset or start monitoring."
        q = question.lower().strip()
        if any(w in q for w in ["row", "records", "entries", "how many"]):
            return f"The dataset contains **{len(df):,}** records across **{len(df.columns)}** metrics."
        if any(w in q for w in ["column", "fields", "metrics"]):
            return f"Available columns:\n`{', '.join(df.columns)}`"
        if any(w in q for w in ["summary", "describe", "stats"]):
            return f"📊 **Dataset Summary:**\n```\n{df.describe().to_string()}\n```"
        if any(w in q for w in ["missing", "null", "nan"]):
            missing = df.isnull().sum()
            if missing.sum() == 0:
                return "✅ No missing values found in this dataset."
            return f"**Missing Values:**\n{missing[missing > 0].to_string()}"
        if any(w in q for w in ["average", "mean", "avg"]):
            numeric = df.select_dtypes(include="number")
            return f"📈 **Averages:**\n```\n{numeric.mean().to_string()}\n```"
        if any(w in q for w in ["max", "highest", "peak"]):
            numeric = df.select_dtypes(include="number")
            return f"🔺 **Maximums:**\n```\n{numeric.max().to_string()}\n```"
        if any(w in q for w in ["anomaly", "disaster", "fault"]):
            if "status" in df.columns:
                anoms = df[df["status"] != "Normal"]
                if len(anoms) > 0:
                    return f"⚠️ Found **{len(anoms)}** anomalous/disaster records:\n```\n{anoms.head(10).to_string()}\n```"
            return "No explicit status anomaly column found."
        return "I am your offline AI assistant. Try asking about rows, summary stats, averages, missing values, or anomalies!"


class SkyGuardApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("SkyGuard AI — Ministry of Earth Sciences / IMD (SIH 2026)")
        self.geometry("1420x940")
        self.configure(bg=BG)
        self.resizable(True, True)
        set_dark_style()

        # Dark listbox for combobox popup
        self.option_add("*TCombobox*Listbox.background", "#0a0f1e")
        self.option_add("*TCombobox*Listbox.foreground", TEXT)
        self.option_add("*TCombobox*Listbox.selectBackground", ACCENT)
        self.option_add("*TCombobox*Listbox.selectForeground", "#ffffff")

        # Monitor state
        self.monitoring = False
        self.readings = deque(maxlen=30)
        self.frozen_counter = 0
        self.pending_fault = None
        self.event_lines = deque(maxlen=20)
        self.sim = AWSSimulator(seed=42)
        self.det = AnomalyDetector()
        self.current_reading = None
        self.is_anomaly = False
        self.severity = "NONE"
        self.confidence = 0.0
        self.graph_fig = None
        self.graph_canvas = None
        self.voice_enabled = tk.BooleanVar(value=True)
        self._was_anom = False

        # Webhook / NDMA dispatch log
        self.ndma_logs = deque(maxlen=15)

        # Chatbot state
        self.chat_dm = PrivacyDataManager()
        self.chat_ai = ChatAIEngine()
        self.active_dataset_id = None

        self._build_ui()
        self.after(100, self._first_tick)
        self.protocol("WM_DELETE_WINDOW", self._on_close)

    def _build_ui(self):
        container = tk.Frame(self, bg=BG)
        container.pack(fill="both", expand=True, padx=10, pady=10)

        # Header
        header = tk.Frame(container, bg=BG)
        header.pack(fill="x", pady=(0, 6))
        tk.Label(header, text="SkyGuard AI — National AWS Intelligence & Disaster Early Warning System", font=("Segoe UI", 20, "bold"),
                 bg=BG, fg=ACCENT, anchor="w").pack(fill="x")
        tk.Label(header, text="MoES / IMD • Optical Fibre DAS • Natural Disaster XAI • Microgrid Power • Mesh Network • Webhooks", font=("Segoe UI", 11, "bold"),
                 bg=BG, fg=PRIMARY, anchor="w").pack(fill="x")
        div = tk.Frame(header, bg=ACCENT3, height=3)
        div.pack(fill="x", pady=(4, 0))

        # Tabs Notebook
        self.notebook = ttk.Notebook(container)
        self.notebook.pack(fill="both", expand=True, pady=6)

        self.tab_monitor = tk.Frame(self.notebook, bg=BG)
        self.tab_mesh = tk.Frame(self.notebook, bg=BG)
        self.tab_chatbot = tk.Frame(self.notebook, bg=BG)

        self.notebook.add(self.tab_monitor, text="  📊 Real-Time Monitor & Disaster XAI  ")
        self.notebook.add(self.tab_mesh, text="  🗺️ Multi-Station Mesh Network Map  ")
        self.notebook.add(self.tab_chatbot, text="  💬 Privacy-First AI Assistant (Chat)  ")

        self._build_monitor_tab(self.tab_monitor)
        self._build_mesh_tab(self.tab_mesh)
        self._build_chatbot_tab(self.tab_chatbot)

    def _make_card(self, parent, title, unit, color):
        """Create a metric card with a title and value label; returns the value label."""
        frame = tk.Frame(parent, bg=CARD, highlightbackground=color,
                         highlightthickness=1, padx=10, pady=6)
        frame.pack(side="left", fill="x", expand=True, padx=2)
        tk.Label(frame, text=title, bg=CARD, fg=TEXT2,
                 font=("Segoe UI", 9)).pack(anchor="w")
        val = tk.Label(frame, text="—", bg=CARD, fg=color,
                       font=("Segoe UI", 20, "bold"))
        val.pack(anchor="w")
        return val

    # ==================================================================
    # TAB 1: Monitor & Disaster XAI
    # ==================================================================
    def _build_monitor_tab(self, parent):
        ctrl = tk.Frame(parent, bg=BG)
        ctrl.pack(fill="x", pady=6)

        self.btn_start = tk.Button(ctrl, text="▶ START", bg=SUCCESS, fg="white",
                                   font=("Calibri", 10, "bold"), padx=10, pady=5, command=self.start_monitor, bd=0)
        self.btn_start.pack(side="left", padx=3)
        self.btn_stop = tk.Button(ctrl, text="⏹ STOP", bg=DANGER, fg="white",
                                  font=("Calibri", 10, "bold"), padx=10, pady=5, command=self.stop_monitor, bd=0)
        self.btn_stop.pack(side="left", padx=3)
        self.btn_demo = tk.Button(ctrl, text="🎬 DEMO MODE", bg=ACCENT3, fg="white",
                                  font=("Calibri", 10, "bold"), padx=12, pady=5, command=self.demo_mode, bd=0)
        self.btn_demo.pack(side="left", padx=8)

        self.voice_checkbox = ttk.Checkbutton(ctrl, text="🔊 Voice Alert", variable=self.voice_enabled, style="TCheckbutton")
        self.voice_checkbox.pack(side="left", padx=6)

        # Export Report Button
        tk.Button(ctrl, text="📄 Export IMD Incident Report", bg=PRIMARY, fg="white",
                  font=("Calibri", 10, "bold"), padx=10, pady=5, command=self.export_incident_report, bd=0).pack(side="right", padx=4)

        # Fault & Disaster Buttons Row 1 (Sensors)
        fault_frame = tk.Frame(parent, bg=BG)
        fault_frame.pack(fill="x", pady=2)
        fault_labels = [
            ("✅ Normal", self.simulate_normal, "#166534", "#bbf7d0"),
            ("🌡 Temp Spike", self.simulate_temp_spike, "#7f1d1d", "#fca5a5"),
            ("🧊 Frozen", self.simulate_frozen, "#1e3a5f", "#93c5fd"),
            ("💧 Humidity", self.simulate_humidity_fault, "#581c87", "#d8b4fe"),
            ("📉 Pressure", self.simulate_pressure_fault, "#78350f", "#fcd34d"),
            ("📡 Comm", self.simulate_comm_error, "#374151", "#9ca3af"),
            ("📈 Drift", self.simulate_drift, "#4c0519", "#fda4af"),
        ]
        for text, cmd, bg, fg in fault_labels:
            tk.Button(fault_frame, text=text, bg=bg, fg=fg, font=("Calibri", 9, "bold"),
                      bd=0, padx=6, pady=3, command=cmd).pack(side="left", padx=2)

        # Fault & Disaster Buttons Row 2 (Natural Disasters & Fibre)
        disaster_frame = tk.Frame(parent, bg=BG)
        disaster_frame.pack(fill="x", pady=2)
        disaster_labels = [
            ("⚡ Earthquake", self.simulate_earthquake, "#581c87", "#e9d5ff"),
            ("🌀 Cyclone", self.simulate_cyclone, "#1e3a8a", "#bfdbfe"),
            ("⛈️ Cloudburst", self.simulate_weather_event, "#134e4a", "#5eead4"),
            ("🌐 Fibre Pre-Fault", self.simulate_fibre_prefault, "#78350f", "#fef08a"),
            ("✂️ Fibre Cut", self.simulate_fibre_cut, "#7f1d1d", "#fca5a5"),
        ]
        for text, cmd, bg, fg in disaster_labels:
            tk.Button(disaster_frame, text=text, bg=bg, fg=fg, font=("Calibri", 9, "bold"),
                      bd=0, padx=8, pady=3, command=cmd).pack(side="left", padx=3)

        # Main grid
        main = tk.Frame(parent, bg=BG)
        main.pack(fill="both", expand=True, pady=4)

        # Left column
        left = tk.Frame(main, bg=BG)
        left.pack(side="left", fill="both", expand=True, padx=(0, 4))

        cards = tk.Frame(left, bg=BG)
        cards.pack(fill="x", pady=2)
        self.card_t = self._make_card(cards, "Temperature", "°C", "#38bdf8")
        self.card_p = self._make_card(cards, "Pressure", "hPa", "#a78bfa")
        self.card_h = self._make_card(cards, "Humidity", "%", "#34d399")

        # Status panel
        status_p = ttk.LabelFrame(left, text=" INTELLIGENCE & DISASTER STATUS ")
        status_p.pack(fill="x", pady=4)
        self.status_label = tk.Label(status_p, text="🟢 SENSOR STREAM NOMINAL", font=("Segoe UI", 18, "bold"),
                                     bg=CARD, fg=GOOD)
        self.status_label.pack(fill="x", padx=6, pady=4)

        sub = tk.Frame(status_p, bg=CARD)
        sub.pack(fill="x", padx=6, pady=(0, 4))
        self.info_type = tk.Label(sub, text="Event Type: Nominal", bg=CARD, fg=PRIMARY, font=("Segoe UI", 10, "bold"), anchor="w")
        self.info_type.pack(fill="x", padx=6)

        info_row = tk.Frame(sub, bg=CARD)
        info_row.pack(fill="x", padx=6, pady=2)
        self.info_sev = tk.Label(info_row, text="Severity: NONE", bg=CARD, fg=TEXT2, font=("Calibri", 9), anchor="w", width=20)
        self.info_sev.pack(side="left")
        self.info_conf = tk.Label(info_row, text="AI Confidence: —", bg=CARD, fg=TEXT2, font=("Calibri", 9), anchor="w")
        self.info_conf.pack(side="left")

        self.info_wmo = tk.Label(sub, text="WMO Status: FLAG 1 (Valid & Verified)", bg=CARD, fg=TEXT, font=("Calibri", 9, "italic"), anchor="w")
        self.info_wmo.pack(fill="x", padx=6, pady=(2, 4))

        # Explainable AI & Disaster Explanation
        why_f = ttk.LabelFrame(left, text=" EXPLAINABLE AI & NATURAL DISASTER REASONING ")
        why_f.pack(fill="x", pady=4)
        self.why_text = tk.Label(why_f, text="AI analysis will appear here.", bg=CARD, fg=TEXT, wraplength=440,
                                 font=("Calibri", 10), justify="left", anchor="nw")
        self.why_text.pack(fill="both", expand=True, padx=6, pady=6)

        # Predictive Maintenance & RUL
        rul_f = ttk.LabelFrame(left, text=" PREDICTIVE MAINTENANCE & REMAINING USEFUL LIFE (RUL) ")
        rul_f.pack(fill="x", pady=4)
        rul_frame = tk.Frame(rul_f, bg=CARD)
        rul_frame.pack(fill="x", padx=6, pady=4)
        self.rul_sensor = tk.Label(rul_frame, text="Sensor RUL: 320 days", bg=CARD, fg=GOOD, font=("Consolas", 9, "bold"), width=22, anchor="w")
        self.rul_sensor.pack(side="left", padx=2)
        self.rul_fibre = tk.Label(rul_frame, text="Fibre RUL: 540 days", bg=CARD, fg=GOOD, font=("Consolas", 9, "bold"), width=22, anchor="w")
        self.rul_fibre.pack(side="left", padx=2)

        # Microgrid Power Telemetry
        power_f = ttk.LabelFrame(left, text=" SOLAR MICROGRID & POWER HEALTH ")
        power_f.pack(fill="x", pady=4)
        pwr_frame = tk.Frame(power_f, bg=CARD)
        pwr_frame.pack(fill="x", padx=6, pady=4)
        self.pwr_volt = tk.Label(pwr_frame, text="Battery: 13.6V", bg=CARD, fg="#38bdf8", font=("Consolas", 9, "bold"), width=16, anchor="w")
        self.pwr_volt.pack(side="left", padx=2)
        self.pwr_curr = tk.Label(pwr_frame, text="Solar: 2.5A", bg=CARD, fg="#34d399", font=("Consolas", 9, "bold"), width=15, anchor="w")
        self.pwr_curr.pack(side="left", padx=2)
        self.pwr_stat = tk.Label(power_f, text="Status: Nominal (Solar Charging Active)", bg=CARD, fg=GOOD, font=("Calibri", 9, "bold"), anchor="w")
        self.pwr_stat.pack(fill="x", padx=6, pady=(0, 4))

        # Optical Fibre Backhaul Telemetry
        fibre_f = ttk.LabelFrame(left, text=" OPTICAL FIBRE BACKHAUL (OTDR & DAS) ")
        fibre_f.pack(fill="x", pady=4)
        fibre_frame = tk.Frame(fibre_f, bg=CARD)
        fibre_frame.pack(fill="x", padx=6, pady=4)
        self.fibre_att = tk.Label(fibre_frame, text="Loss: 0.22 dB/km", bg=CARD, fg="#38bdf8", font=("Consolas", 9, "bold"), width=16, anchor="w")
        self.fibre_att.pack(side="left", padx=2)
        self.fibre_strain = tk.Label(fibre_frame, text="Strain: 12.0 με", bg=CARD, fg="#a855f7", font=("Consolas", 9, "bold"), width=15, anchor="w")
        self.fibre_strain.pack(side="left", padx=2)
        self.fibre_stat = tk.Label(fibre_f, text="Status: Nominal (OTDR Verified)", bg=CARD, fg=GOOD, font=("Calibri", 9, "bold"), anchor="w")
        self.fibre_stat.pack(fill="x", padx=6, pady=(0, 4))

        # Right column
        right = tk.Frame(main, bg=BG)
        right.pack(side="right", fill="both", expand=True, padx=(4, 0))

        station_f = tk.Frame(right, bg=BG)
        station_f.pack(fill="x", pady=(0, 2))
        tk.Label(station_f, text="📍 Station:", bg=BG, fg=TEXT2, font=("Segoe UI", 10, "bold")).pack(side="left")
        self.station_var = tk.StringVar(value="AWS-101 [New Delhi / Safdarjung]")
        self.station_dropdown = ttk.Combobox(station_f, textvariable=self.station_var,
                                             values=list(STATION_PROFILES.keys()),
                                             state="readonly", width=34, font=("Calibri", 10), style="TCombobox")
        self.station_dropdown.pack(side="left", padx=8)
        self.station_dropdown.bind("<<ComboboxSelected>>", self._on_station_change)

        # Graph
        graph_div = ttk.LabelFrame(right, text=" MULTIVARIATE SENSOR STREAM ")
        graph_div.pack(fill="both", expand=True, pady=2)
        self.graph_frame = tk.Frame(graph_div, bg="#070a14")
        self.graph_frame.pack(fill="both", expand=True, padx=4, pady=4)

        if HAS_MPL:
            plt_bg = "#070a14"
            self.graph_fig = Figure(figsize=(7, 3.5), dpi=100)
            self.graph_fig.patch.set_facecolor(plt_bg)
            self.ax_t = self.graph_fig.add_subplot(311)
            self.ax_p = self.graph_fig.add_subplot(312, sharex=self.ax_t)
            self.ax_h = self.graph_fig.add_subplot(313, sharex=self.ax_t)
            for ax in [self.ax_t, self.ax_p, self.ax_h]:
                ax.set_facecolor(plt_bg)
                ax.tick_params(colors="#e2e8f0", labelsize=7)
                for spine in ax.spines.values():
                    spine.set_color("#1e293b")
                ax.grid(color="#1e293b", linestyle='-', linewidth=0.5, alpha=0.5)
            self.ax_t.set_ylabel("Temp (°C)", color=ACCENT, fontsize=8)
            self.ax_p.set_ylabel("Press (hPa)", color="#a855f7", fontsize=8)
            self.ax_h.set_ylabel("Hum (%)", color="#10b981", fontsize=8)
            self.graph_fig.tight_layout(pad=1.0)
            self.canvas = FigureCanvasTkAgg(self.graph_fig, master=self.graph_frame)
            self.canvas.draw()
            canvas_widget = self.canvas.get_tk_widget()
            canvas_widget.configure(bg=plt_bg, highlightthickness=0)
            canvas_widget.pack(fill="both", expand=True)

        # NDMA / IMD Webhook Dispatch Log
        ndma_f = ttk.LabelFrame(right, text=" 📡 NDMA / IMD REST API WEBHOOK DISPATCH LOG ")
        ndma_f.pack(fill="x", pady=2)
        self.ndma_text = scrolledtext.ScrolledText(ndma_f, height=5, wrap=tk.WORD,
                                                   bg="#0b1220", fg=TEXT, font=("Consolas", 8),
                                                   insertbackground=TEXT, highlightthickness=0, bd=0)
        self.ndma_text.pack(fill="both", expand=True, padx=4, pady=4)
        self.ndma_text.config(state="disabled")

        # Event log
        log_f = ttk.LabelFrame(right, text=" EVENT LOG ")
        log_f.pack(fill="x", pady=2)
        self.event_text = scrolledtext.ScrolledText(log_f, height=5, wrap=tk.WORD,
                                                    bg="#0b1220", fg=TEXT, font=("Consolas", 8),
                                                    insertbackground=TEXT, highlightthickness=0, bd=0)
        self.event_text.pack(fill="both", expand=True, padx=4, pady=4)
        self.event_text.config(state="disabled")

        # Darken ScrolledText wrapper frames and scrollbars
        for st_w in (self.ndma_text, self.event_text):
            st_w.frame.configure(bg=BG)
            try:
                st_w.vbar.configure(bg=CARD, activebackground=CARD2, troughcolor="#070a14",
                                    highlightthickness=0, bd=0, relief="flat")
            except Exception:
                pass

    # ==================================================================
    # TAB 2: Multi-Station Mesh Network Map
    # ==================================================================
    def _build_mesh_tab(self, parent):
        top_f = tk.Frame(parent, bg=BG)
        top_f.pack(fill="x", pady=10)
        tk.Label(top_f, text="🗺️ National AWS Mesh Network Status (MoES / IMD)", font=("Segoe UI", 16, "bold"),
                 bg=BG, fg=ACCENT).pack(anchor="w", padx=10)
        tk.Label(top_f, text="Real-time status monitoring across all deployed high-altitude, coastal, arid, and subtropical stations.",
                 font=("Calibri", 10), bg=BG, fg=TEXT2).pack(anchor="w", padx=10)

        grid_f = tk.Frame(parent, bg=BG)
        grid_f.pack(fill="both", expand=True, padx=10, pady=10)

        self.mesh_cards = {}
        row, col = 0, 0
        for name, p in STATION_PROFILES.items():
            card = ttk.LabelFrame(grid_f, text=f" {name} ")
            card.grid(row=row, column=col, padx=10, pady=10, sticky="nsew")
            grid_f.grid_columnconfigure(col, weight=1)
            grid_f.grid_rowconfigure(row, weight=1)

            lbl_reg = tk.Label(card, text=p["region"], bg=CARD, fg=PRIMARY, font=("Segoe UI", 10, "bold"), anchor="w")
            lbl_reg.pack(fill="x", padx=8, pady=4)

            lbl_stat = tk.Label(card, text="🟢 Status: Nominal", bg=CARD, fg=GOOD, font=("Segoe UI", 12, "bold"), anchor="w")
            lbl_stat.pack(fill="x", padx=8, pady=2)

            lbl_details = tk.Label(card, text=f"Elevation: {p['elevation']}\nFibre Span: {p['fibre_km']} km\nTemp Baseline: {p['t_mean']}°C",
                                   bg=CARD, fg=TEXT, font=("Calibri", 10), justify="left", anchor="w")
            lbl_details.pack(fill="x", padx=8, pady=6)

            self.mesh_cards[name] = {"stat": lbl_stat, "details": lbl_details}

            col += 1
            if col > 1:
                col = 0
                row += 1

        refresh_btn = tk.Button(parent, text="🔄 REFRESH MESH NETWORK STATUS", bg=PRIMARY, fg="white",
                                font=("Calibri", 11, "bold"), padx=14, pady=8, command=self.refresh_mesh_network, bd=0)
        refresh_btn.pack(pady=10)

    def refresh_mesh_network(self):
        for name, d in self.mesh_cards.items():
            # simulate slight status variation
            if "Delhi" in name:
                d["stat"].config(text="🟢 Status: Nominal", fg=GOOD)
            elif "Leh" in name:
                d["stat"].config(text="⚠️ Status: High Altitude Wind Warning", fg=WARNING)
            elif "Mumbai" in name:
                d["stat"].config(text="🟢 Status: Coastal Nominal", fg=GOOD)
            elif "Jaisalmer" in name:
                d["stat"].config(text="🟢 Status: Arid Nominal", fg=GOOD)
        messagebox.showinfo("Mesh Network", "Mesh network sync complete. All stations reporting telemetry.")

    # ==================================================================
    # TAB 3: Privacy-First AI Chatbot
    # ==================================================================
    def _build_chatbot_tab(self, parent):
        banner = tk.Frame(parent, bg=CARD, bd=1, relief="solid", highlightbackground="#1e293b", highlightthickness=1)
        banner.pack(fill="x", pady=(0, 6), padx=4)
        tk.Label(banner, text="🛡️ SkyGuard AI Privacy-First Assistant", font=("Segoe UI", 14, "bold"), bg=CARD, fg=ACCENT2).pack(anchor="w", padx=10, pady=(6, 2))
        tk.Label(banner, text="100% Offline • Zero Cloud Transmission • Auto-Wiped on Exit", font=("Calibri", 10), bg=CARD, fg=TEXT2).pack(anchor="w", padx=10, pady=(0, 6))

        ds_bar = tk.Frame(parent, bg=BG)
        ds_bar.pack(fill="x", pady=4)
        tk.Button(ds_bar, text="📁 Load CSV Dataset", bg=PRIMARY, fg="white", font=("Calibri", 10, "bold"), padx=10, pady=5, command=self._chat_load_file, bd=0).pack(side="left", padx=4)
        tk.Button(ds_bar, text="🔄 Sync Live AWS Stream", bg=ACCENT, fg="white", font=("Calibri", 10, "bold"), padx=10, pady=5, command=self._chat_sync_stream, bd=0).pack(side="left", padx=4)
        self.chat_status_label = tk.Label(ds_bar, text="Status: Using Live Stream Buffer", bg=BG, fg=TEXT2, font=("Calibri", 10, "italic"))
        self.chat_status_label.pack(side="left", padx=10)

        chat_frame = ttk.LabelFrame(parent, text=" CONVERSATION ")
        chat_frame.pack(fill="both", expand=True, pady=4)
        self.chat_display = scrolledtext.ScrolledText(chat_frame, wrap=tk.WORD, bg="#070a14", fg=TEXT, font=("Calibri", 11),
                                                      insertbackground=TEXT, highlightthickness=0, bd=0)
        self.chat_display.pack(fill="both", expand=True, padx=6, pady=6)
        self.chat_display.config(state="disabled")
        self.chat_display.frame.configure(bg=BG)
        try:
            self.chat_display.vbar.configure(bg=CARD, activebackground=CARD2, troughcolor="#070a14",
                                             highlightthickness=0, bd=0, relief="flat")
        except Exception:
            pass

        input_bar = tk.Frame(parent, bg=BG)
        input_bar.pack(fill="x", pady=6)
        self.chat_input = tk.Entry(input_bar, font=("Calibri", 12), bg=CARD, fg=TEXT, insertbackground=TEXT,
                                   relief="flat", highlightbackground="#1e293b", highlightthickness=1)
        self.chat_input.pack(side="left", fill="x", expand=True, padx=(0, 6), ipady=6)
        self.chat_input.bind("<Return>", lambda event: self._send_chat())
        tk.Button(input_bar, text="SEND 💬", bg=SUCCESS, fg="white", font=("Calibri", 11, "bold"), padx=16, pady=6, command=self._send_chat, bd=0).pack(side="right")

        self._append_chat("🤖 SkyGuard AI", "Hello! I am your offline, privacy-first data assistant. Ask me anything about your loaded weather dataset or live AWS stream.")

    def _chat_sync_stream(self):
        import pandas as pd
        if not self.readings:
            messagebox.showinfo("Sync", "No readings in stream yet.")
            return
        data = [{"tick": i, "temperature_c": r[0], "pressure_hpa": r[1], "humidity_pct": r[2]} for i, r in enumerate(self.readings) if r and len(r) >= 3]
        if not data:
            return
        df = pd.DataFrame(data)
        meta = self.chat_dm.ingest_df(df, name="Live_AWS_Stream.csv")
        self.active_dataset_id = meta["id"]
        self.chat_status_label.config(text="Active: Live Stream Buffer")
        self._append_chat("System", "✅ Synced live AWS stream into chatbot.")

    def _chat_load_file(self):
        path = filedialog.askopenfilename(title="Select Dataset", filetypes=[("CSV files", "*.csv"), ("All files", "*.*")])
        if not path:
            return
        try:
            meta = self.chat_dm.ingest_file(path)
            self.active_dataset_id = meta["id"]
            self.chat_status_label.config(text=f"Active: {meta['name']}")
            self._append_chat("System", f"✅ Loaded {meta['name']} ({meta['rows']:,} rows).")
        except Exception as e:
            messagebox.showerror("Error", str(e))

    def _send_chat(self):
        query = self.chat_input.get().strip()
        if not query:
            return
        self.chat_input.delete(0, tk.END)
        self._append_chat("You", query)
        if not self.active_dataset_id:
            self._chat_sync_stream()
        try:
            df = self.chat_dm.get_dataframe(self.active_dataset_id)
            ans = self.chat_ai.answer(query, df)
            self._append_chat("🤖 SkyGuard AI", ans)
        except Exception as e:
            self._append_chat("System", f"Error: {e}")

    def _append_chat(self, sender, text):
        self.chat_display.config(state="normal")
        self.chat_display.insert("end", f"\n{sender}:\n{text}\n" + "─"*50 + "\n")
        self.chat_display.see("end")
        self.chat_display.config(state="disabled")

    # ==================================================================
    # Monitoring & Webhooks
    # ==================================================================
    def dispatch_ndma_webhook(self, alert_type, details):
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        payload = f"[{ts}] POST /api/v1/ndma/disaster-alert | Station: {self.station_var.get()} | Alert: {alert_type} | Details: {details}"
        self.ndma_logs.append(payload)
        self.ndma_text.config(state="normal")
        self.ndma_text.insert("end", payload + "\n")
        self.ndma_text.see("end")
        if len(self.ndma_logs) > 15:
            self.ndma_text.delete("1.0", "2.0")
        self.ndma_text.config(state="disabled")

    def export_incident_report(self):
        path = filedialog.asksaveasfilename(defaultextension=".csv", filetypes=[("CSV Files", "*.csv"), ("Text Files", "*.txt")])
        if not path:
            return
        try:
            with open(path, "w", encoding="utf-8") as f:
                f.write("SkyGuard AI — Official IMD / MoES Incident & Telemetry Report\n")
                f.write(f"Timestamp,{datetime.now().isoformat()}\n")
                f.write(f"Station,{self.station_var.get()}\n")
                f.write(f"Event Type,{self.info_type.cget('text')}\n")
                f.write(f"Severity,{self.info_sev.cget('text')}\n")
                f.write(f"WMO Status,{self.info_wmo.cget('text')}\n")
                f.write(f"Explanation,\"{self.why_text.cget('text')}\"\n")
            messagebox.showinfo("Export", f"Official incident report successfully saved to:\n{path}")
        except Exception as e:
            messagebox.showerror("Error", str(e))

    def speak_alert(self, text):
        if not self.voice_enabled.get():
            return
        def _run():
            try:
                import subprocess
                safe_text = text.replace("'", "''")
                cmd = f"Add-Type -AssemblyName System.Speech; (New-Object System.Speech.Synthesis.SpeechSynthesizer).Speak('{safe_text}')"
                subprocess.Popen(["powershell", "-Command", cmd], creationflags=0x08000000)
            except Exception:
                pass
        threading.Thread(target=_run, daemon=True).start()

    def start_monitor(self):
        if self.monitoring:
            return
        self.monitoring = True
        self.sim.reset_fault()
        self.event_log("AWS monitoring started")
        self.btn_start.config(state="disabled")
        self.btn_stop.config(state="normal")
        self.tick()

    def stop_monitor(self):
        self.monitoring = False
        self.event_log("Monitoring stopped")
        self.btn_start.config(state="normal")
        self.btn_stop.config(state="disabled")

    def tick(self):
        try:
            if not self.monitoring:
                return
            if self.pending_fault:
                reading = self.pending_fault
                self.pending_fault = None
            else:
                reading = self.sim.generate_normal()

            if reading is not None:
                if len(self.readings) > 0:
                    prev = self.readings[-1]
                    if abs(prev[0] - reading[0]) < 0.01:
                        self.frozen_counter += 1
                    else:
                        self.frozen_counter = 0
            else:
                self.frozen_counter = 0

            self.current_reading = reading
            self.readings.append(reading if reading else (None, None, None))

            if reading is not None and self.frozen_counter == 0:
                self.det.update_normal_history(reading)

            info = self.det.classify_anomaly(reading, frozen_counter=self.frozen_counter)
            is_anom = (info.get("severity", "NONE") != "NONE") or (info.get("type") != "Nominal")
            conf = float(info.get("confidence", 0.0))

            self.is_anomaly = is_anom
            self.severity = info.get("severity", "NONE")
            self.confidence = conf

            # Fibre & Power telemetry update
            f_state = self.sim.fibre.step(self.sim.active_disaster)
            self.fibre_att.config(text=f"Loss: {f_state['attenuation_db_km']:.2f} dB/km")
            self.fibre_strain.config(text=f"Strain: {f_state['strain_micro']:.1f} με")
            if f_state["warning"]:
                self.fibre_stat.config(text=f"Status: {f_state['status']}", fg=BAD if "Cut" in f_state['status'] else WARNING)
            else:
                self.fibre_stat.config(text=f"Status: {f_state['status']}", fg=GOOD)

            p_state = self.sim.power.step(bool(self.sim.active_disaster))
            self.pwr_volt.config(text=f"Battery: {p_state['battery_v']}V")
            self.pwr_curr.config(text=f"Solar: {p_state['solar_a']}A")
            self.pwr_stat.config(text=f"Status: {p_state['status']}", fg=GOOD if "Nominal" in p_state['status'] else WARNING)

            rul = self.sim.get_rul_estimation()
            self.rul_sensor.config(text=f"Sensor RUL: {rul['sensor_rul_days']} days", fg=GOOD if rul['sensor_rul_days'] > 30 else WARNING)
            self.rul_fibre.config(text=f"Fibre RUL: {rul['fibre_rul_days']} days", fg=GOOD if rul['fibre_rul_days'] > 30 else WARNING)

            if reading is not None:
                self.card_t.config(text=f"{reading[0]}°C" if reading[0] is not None else "—")
                self.card_p.config(text=f"{reading[1]} hPa" if reading[1] is not None else "—")
                self.card_h.config(text=f"{reading[2]}%" if reading[2] is not None else "—")
            else:
                self.card_t.config(text="—")
                self.card_p.config(text="—")
                self.card_h.config(text="—")

            if is_anom:
                if info.get("is_genuine_weather"):
                    self.status_label.config(text=f"🟢 {info['type'].upper()}", fg=ACCENT2)
                    if "Natural Disaster" in info['type'] or "Cyclone" in info['type'] or "Earthquake" in info['type'] or "Cloudburst" in info['type']:
                        self.dispatch_ndma_webhook(info['type'], info['explanation'])
                else:
                    self.status_label.config(text="🔴 ANOMALY DETECTED", fg=BAD)
                self.info_type.config(text=f"Event Type: {info['type']}")
                self.info_sev.config(text=f"Severity: {self.severity}")
                self.info_conf.config(text=f"AI Confidence: {self.confidence:.0f}%")
                self.info_wmo.config(text=f"WMO Status: {info.get('wmo_status', 'FLAG 2')}")
                self.why_text.config(text=info["explanation"])
                self.event_log(f"{info['type']} detected | Severity: {self.severity}")
            else:
                self.status_label.config(text="🟢 SENSOR STREAM NOMINAL", fg=GOOD)
                self.info_type.config(text="Event Type: Nominal")
                self.info_sev.config(text="Severity: NONE")
                self.info_conf.config(text="AI Confidence: —")
                self.info_wmo.config(text="WMO Status: FLAG 1 (Valid & Verified)")
                self.why_text.config(text="AI verified observation is nominal within WMO standard uncertainty limits.")

            if is_anom and not self._was_anom:
                self.speak_alert(f"Warning. {info.get('type', 'Anomaly')} detected.")
            elif not is_anom and self._was_anom:
                self.speak_alert("Sensor stream returned to normal.")
            self._was_anom = is_anom

            self._update_graph()

            if self.monitoring:
                self.after(1500, self.tick)
        except Exception as e:
            self.event_log(f"Error in tick: {e}")
            if self.monitoring:
                self.after(1500, self.tick)

    def _update_graph(self):
        if not HAS_MPL or self.graph_fig is None:
            return
        for ax in [self.ax_t, self.ax_p, self.ax_h]:
            ax.clear()
            ax.grid(color="#1e293b", linestyle='-', linewidth=0.5, alpha=0.5)
        xs = np.arange(len(self.readings))
        if len(self.readings) == 0:
            return
        t_vals = [r[0] if r is not None and r[0] is not None else np.nan for r in self.readings]
        p_vals = [r[1] if r is not None and r[1] is not None else np.nan for r in self.readings]
        h_vals = [r[2] if r is not None and r[2] is not None else np.nan for r in self.readings]
        self.ax_t.plot(xs, t_vals, color=ACCENT, marker="o", markersize=3, linewidth=1.5)
        self.ax_p.plot(xs, p_vals, color="#a855f7", marker="s", markersize=3, linewidth=1.5)
        self.ax_h.plot(xs, h_vals, color="#10b981", marker="^", markersize=3, linewidth=1.5)
        if self.is_anomaly and len(self.readings) > 0:
            last = len(self.readings) - 1
            for ax, vals in zip([self.ax_t, self.ax_p, self.ax_h], [t_vals, p_vals, h_vals]):
                if not np.isnan(vals[-1]):
                    ax.scatter([last], [vals[-1]], color=CRITICAL, s=100, zorder=5)
                ax.axvspan(last - 0.5, last + 0.5, color=CRITICAL, alpha=0.15)
        self.ax_t.set_ylabel("Temp (°C)", color=ACCENT, fontsize=8)
        self.ax_p.set_ylabel("Press (hPa)", color="#a855f7", fontsize=8)
        self.ax_h.set_ylabel("Hum (%)", color="#10b981", fontsize=8)
        for ax in [self.ax_t, self.ax_p, self.ax_h]:
             ax.tick_params(colors="#e2e8f0", labelsize=7)
             ax.set_xlim(max(0, len(xs) - 30), max(10, len(xs) - 1))
        self.graph_fig.tight_layout(pad=1.0)
        self.canvas.draw_idle()

    def _on_station_change(self, event):
        station = self.station_var.get()
        self.sim.set_station(station)
        p = self.sim.t_mean, self.sim.p_mean, self.sim.h_mean
        self.det.retrain_station(*p)
        self.readings.clear()
        self.event_log(f"Station switched to {station}. Model retrained.")
        self.current_reading = self.sim.generate_normal()
        self.readings.append(self.current_reading)
        self._update_graph()

    def simulate_normal(self):
        self.pending_fault = self.sim.generate_normal()
        self.event_log("Simulated normal reading")

    def simulate_temp_spike(self):
        self.pending_fault = self.sim.inject_temperature_spike()
        self.event_log("Simulated Temperature Spike")

    def simulate_frozen(self):
        self.pending_fault = self.sim.inject_frozen_sensor()
        self.event_log("Simulated Frozen Sensor")

    def simulate_humidity_fault(self):
        self.pending_fault = self.sim.inject_humidity_fault()
        self.event_log("Simulated Humidity Fault")

    def simulate_pressure_fault(self):
        self.pending_fault = self.sim.inject_pressure_fault()
        self.event_log("Simulated Pressure Fault")

    def simulate_comm_error(self):
        self.pending_fault = self.sim.inject_communication_error()
        self.event_log("Simulated Communication Error")

    def simulate_drift(self):
        self.pending_fault = self.sim.inject_drift()
        self.event_log("Simulated Sensor Calibration Drift")

    def simulate_earthquake(self):
        self.pending_fault = self.sim.inject_earthquake_event()
        self.event_log("🚨 Natural Disaster: Earthquake (Seismic Shockwave)")

    def simulate_cyclone(self):
        self.pending_fault = self.sim.inject_cyclone_event()
        self.event_log("🌀 Natural Disaster: Tropical Cyclone Landfall")

    def simulate_weather_event(self):
        self.pending_fault = self.sim.inject_genuine_weather_event()
        self.event_log("⛈️ Natural Disaster: Cloudburst / Flash Flood")

    def simulate_fibre_prefault(self):
        self.pending_fault = self.sim.inject_drift()
        self.sim.fibre.step("fibre_degradation")
        self.event_log("🌐 Optical Fibre Pre-Fault Warning (Attenuation Buildup)")

    def simulate_fibre_cut(self):
        self.pending_fault = self.sim.inject_communication_error()
        self.sim.fibre.step("fibre_cut")
        self.event_log("✂️ Optical Fibre Cut / Link Down")

    def event_log(self, msg):
        ts = datetime.now().strftime("%H:%M:%S")
        line = f"[{ts}] {msg}"
        self.event_lines.append(line)
        self.event_text.config(state="normal")
        self.event_text.insert("end", line + "\n")
        self.event_text.see("end")
        if len(self.event_lines) > 20:
            self.event_text.delete("1.0", "2.0")
        self.event_text.config(state="disabled")

    def demo_mode(self):
        if self.monitoring:
            return
        self.event_log("🎬 DEMO MODE started")
        self.start_monitor()
        self.after(1500, self.simulate_temp_spike)
        self.after(3500, self.simulate_earthquake)
        self.after(5500, self.simulate_cyclone)
        self.after(7500, self.simulate_fibre_prefault)
        self.after(9500, self.simulate_normal)
        self.after(11500, self.stop_monitor)

    def _first_tick(self):
        r = self.sim.generate_normal()
        self.current_reading = r
        self.card_t.config(text=f"{r[0]}°C")
        self.card_p.config(text=f"{r[1]} hPa")
        self.card_h.config(text=f"{r[2]}%")
        self.readings.append(r)
        self._update_graph()
        self.event_log("SkyGuard AI fully initialized with all features enabled.")

    def _on_close(self):
        self.stop_monitor()
        self.chat_dm.purge_all()
        self.destroy()


if __name__ == "__main__":
    app = SkyGuardApp()
    app.mainloop()
