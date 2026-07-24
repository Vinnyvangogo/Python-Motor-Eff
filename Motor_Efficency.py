"""
Three-Phase Motor Controller Efficiency Analyser
=================================================
NI cDAQ-9189 branch script — Modules 2 & 3, Channels 0-5 only.

Hardware:
  Module 2  CH0-5  NI 9320  Voltage inputs  (Phase A/B/C for input & output)
  Module 3  CH0-5  NI 9320  Current inputs  (Phase A/B/C for input & output)

Signal pairing (3-phase motor controller efficiency):
  Input  side: Mod2/CH0 × Mod3/CH0  (Phase A in)
               Mod2/CH1 × Mod3/CH1  (Phase B in)
               Mod2/CH2 × Mod3/CH2  (Phase C in)
  Output side: Mod2/CH3 × Mod3/CH3  (Phase A out)
               Mod2/CH4 × Mod3/CH4  (Phase B out)
               Mod2/CH5 × Mod3/CH5  (Phase C out)

Calculations:
  Per-phase apparent power  S  = Vrms × Irms  (VA)
  Three-phase total input   Pin  = sum of input  phase powers  (W, assuming
                                   unity power factor; adjust PF if known)
  Three-phase total output  Pout = sum of output phase powers
  Efficiency                η   = (Pout / Pin) × 100  (%)

Configuration:
  Reads chassis name, IP, sample rates, channel names, and scale/offset
  calibration from the same cdaq_calibration.json used by the main script.
  Only the Mod2/Mod3 CH0-5 entries are consumed — all other sections are
  ignored.

Dependencies:
    pip install nidaqmx numpy
"""

import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import threading
import time
import csv
import math
import queue
import json
import os
from datetime import datetime
from typing import Optional

import numpy as np

try:
    import nidaqmx
    from nidaqmx.constants import (
        AcquisitionType, TerminalConfiguration, ReadRelativeTo
    )
    SIMULATION_MODE = False
except ImportError:
    SIMULATION_MODE = True
    print("[WARNING] nidaqmx not found - running in SIMULATION mode")


# ══════════════════════════════════════════════════════════════════════════
#  Configuration: which channels this script uses
# ══════════════════════════════════════════════════════════════════════════
N_CH        = 6          # channels per module (0-5)
MOD_VOLTAGE = 2          # Mod2 → voltage inputs
MOD_CURRENT = 3          # Mod3 → current inputs

# Descriptive labels shown in the UI — overwritten from JSON on startup
V_NAMES = [f"V_PH{c}" for c in ["A_IN","B_IN","C_IN","A_OUT","B_OUT","C_OUT"]]
I_NAMES = [f"I_PH{c}" for c in ["A_IN","B_IN","C_IN","A_OUT","B_OUT","C_OUT"]]

# Phase pairing for power calculation: (v_idx, i_idx)  — indices 0-5
PHASE_PAIRS = [(i, i) for i in range(N_CH)]   # V0×I0 … V5×I5

# Input phases (indices into 0-5 range) and output phases
INPUT_PHASES  = [0, 1, 2]   # CH0-2: input  side A/B/C
OUTPUT_PHASES = [3, 4, 5]   # CH3-5: output side A/B/C


# ══════════════════════════════════════════════════════════════════════════
#  JSON calibration file path (shared with main script)
# ══════════════════════════════════════════════════════════════════════════
def _find_cal_file() -> str:
    import sys
    candidates = []
    try:
        script_dir = os.path.dirname(os.path.abspath(sys.argv[0]))
        if script_dir and os.path.isdir(script_dir):
            candidates.append(os.path.join(script_dir, "cdaq_calibration.json"))
    except Exception:
        pass
    try:
        file_dir = os.path.dirname(os.path.abspath(__file__))
        p = os.path.join(file_dir, "cdaq_calibration.json")
        if p not in candidates:
            candidates.append(p)
    except NameError:
        pass
    cwd_p = os.path.join(os.getcwd(), "cdaq_calibration.json")
    if cwd_p not in candidates:
        candidates.append(cwd_p)

    print("[Motor Efficiency] Searching for cdaq_calibration.json:")
    for p in candidates:
        exists = os.path.exists(p)
        print(f"  {'FOUND' if exists else 'not found':10s}  {p}")
        if exists:
            return p
    default = candidates[0]
    print(f"[Motor Efficiency] JSON not found -- will create at: {default}")
    return default

CAL_FILE = _find_cal_file()

# Colours
C_BG     = "#0d1117"
C_PANEL  = "#161b22"
C_BORDER = "#30363d"
C_ACCENT = "#00b4d8"
C_GREEN  = "#39d353"
C_RED    = "#f85149"
C_YELLOW = "#e3b341"
C_ORANGE = "#f0883e"
C_TEXT   = "#e6edf3"
C_MUTED  = "#8b949e"
C_INPUT  = "#21262d"
C_BLUE   = "#58a6ff"

FONT_MONO   = ("Courier New", 9)
FONT_MONO_S = ("Courier New", 8)
FONT_MONO_L = ("Courier New", 14, "bold")
FONT_SMALL  = ("Segoe UI", 9)
FONT_TINY   = ("Segoe UI", 8)
FONT_MED    = ("Segoe UI", 10)
FONT_BOLD   = ("Segoe UI", 10, "bold")
FONT_HEAD   = ("Segoe UI", 12, "bold")
FONT_TITLE  = ("Segoe UI", 14, "bold")


# ══════════════════════════════════════════════════════════════════════════
#  DAQ Manager (Mod2/Mod3 CH0-5 only)
# ══════════════════════════════════════════════════════════════════════════
class MotorDAQManager:
    """Acquires voltage (Mod2) and current (Mod3) CH0-5 from the cDAQ-9189.

    Uses the same JSON calibration file as the main script.  Only the
    NI_9320_modules_2_to_6 entries for module 2 and 3, channels 0-5 are
    consumed; all other JSON sections are ignored.
    """

    def __init__(self, chassis: str, ip: str, error_queue: "queue.Queue"):
        self.chassis = chassis or "cDAQ1"
        self.ip      = ip
        self.errors  = error_queue

        self.dev_v = f"{self.chassis}Mod{MOD_VOLTAGE}"   # e.g. cDAQ9189-...Mod2
        self.dev_i = f"{self.chassis}Mod{MOD_CURRENT}"   # e.g. cDAQ9189-...Mod3

        # Live RMS data — index 0-5 for each module
        self.v_rms = [0.0] * N_CH   # Vrms per phase
        self.i_rms = [0.0] * N_CH   # Irms per phase

        # Calibration: (scale, offset) per channel, applied to RMS
        self.v_cal = [(1.0, 0.0)] * N_CH
        self.i_cal = [(1.0, 0.0)] * N_CH

        # Channel enable
        self.v_enabled = [True] * N_CH
        self.i_enabled = [True] * N_CH

        # Acquisition config (set from JSON before start)
        self.hw_rate   = 200_000
        self.ac_freq   = 60.0
        self.n_cycles  = 3

        # State
        self.running = False
        self.logging = False
        self.log_queue: queue.Queue = queue.Queue()

        # Error throttle
        self._last_error_time:   dict = {}
        self._error_repeat_count: dict = {}

        self._sim_t = 0.0

    def report_error(self, source: str, message: str):
        key  = (source, message)
        now  = time.monotonic()
        last = self._last_error_time.get(key, 0.0)
        count = self._error_repeat_count.get(key, 0) + 1
        self._error_repeat_count[key] = count
        if now - last < 0.5:
            return
        self._last_error_time[key] = now
        suffix = f"  (x{count})" if count > 1 else ""
        self._error_repeat_count[key] = 0
        self.errors.put((datetime.now(), source, message + suffix))

    def test_connection(self) -> tuple[bool, str]:
        if SIMULATION_MODE:
            return True, "Simulation mode — no hardware required."

        if self.ip:
            import socket
            for port in (80, 502):
                try:
                    s = socket.create_connection((self.ip, port), timeout=2.0)
                    s.close()
                    break
                except OSError as e:
                    if port == 502:
                        return False, (f"Cannot reach {self.ip} ({e}). "
                                       f"Check IP, cabling, chassis power.")

        try:
            sys_obj   = nidaqmx.system.System.local()
            dev_names = [d.name for d in sys_obj.devices]
            missing   = [d for d in (self.dev_v, self.dev_i) if d not in dev_names]
            if missing:
                return False, (f"Module device(s) not found: {missing}. "
                               f"Available: {dev_names or 'none'}.")
            nidaqmx.system.Device(self.dev_v).self_test_device()
        except Exception as e:
            return False, f"NI-DAQmx device check failed: {e}"

        return True, f"Connected to '{self.chassis}' ({self.ip})."

    def start(self):
        if self.running:
            return
        self.running = True
        threading.Thread(target=self._acq_loop, daemon=True).start()

    def stop(self):
        self.running = False

    def _acq_loop(self):
        """Single continuous task spanning both modules (channel expansion).

        Voltage channels (Mod2 CH0-5) and current channels (Mod3 CH0-5)
        are added to ONE task so they are sampled synchronously — this is
        essential for correct power calculations since V×I must be in phase.
        """
        ac_freq   = max(1.0, self.ac_freq)
        n_cycles  = max(1, self.n_cycles)
        block_sec = n_cycles / ac_freq
        interval  = block_sec
        n_samp    = max(1, int(round(self.hw_rate * block_sec)))

        # Channel ordering in the task: V0-V5 then I0-I5
        task = None
        if not SIMULATION_MODE:
            try:
                task = nidaqmx.Task()
                for ch in range(N_CH):
                    task.ai_channels.add_ai_voltage_chan(
                        f"{self.dev_v}/ai{ch}",
                        min_val=-10.0, max_val=10.0,
                        terminal_config=TerminalConfiguration.DIFF
                    )
                for ch in range(N_CH):
                    task.ai_channels.add_ai_voltage_chan(
                        f"{self.dev_i}/ai{ch}",
                        min_val=-10.0, max_val=10.0,
                        terminal_config=TerminalConfiguration.DIFF
                    )
                task.timing.cfg_samp_clk_timing(
                    rate=float(self.hw_rate),
                    sample_mode=AcquisitionType.CONTINUOUS,
                    samps_per_chan=self.hw_rate * 2   # 2 s host buffer
                )
                task.start()
            except Exception as e:
                self.report_error("Acq", f"Failed to start task: {e}")
                self.running = False
                if task:
                    try: task.close()
                    except Exception: pass
                return

        try:
            while self.running:
                t0 = time.perf_counter()

                if SIMULATION_MODE:
                    self._sim_t += interval
                    t_arr = np.linspace(self._sim_t - interval,
                                        self._sim_t, n_samp, endpoint=False)
                    # Simulate realistic 3-phase: input ~240Vrms, output ~230Vrms
                    # Input voltages V0-V2, output voltages V3-V5
                    v_raw = [
                        5.43 * np.sin(2*np.pi*400*t_arr + i*2*np.pi/3) for i in range(3)
                    ] + [
                        5.20 * np.sin(2*np.pi*400*t_arr + i*2*np.pi/3) for i in range(3)
                    ]
                    i_raw = [
                        0.50 * np.sin(2*np.pi*400*t_arr + i*2*np.pi/3) for i in range(3)
                    ] + [
                        0.52 * np.sin(2*np.pi*400*t_arr + i*2*np.pi/3) for i in range(3)
                    ]
                    raw_samples = v_raw + i_raw   # 12 channels total

                else:
                    try:
                        data = task.read(
                            number_of_samples_per_channel=nidaqmx.constants.READ_ALL_AVAILABLE,
                            timeout=2.0)
                        first = np.atleast_1d(data[0]) if data else np.array([])
                        if len(first) == 0:
                            time.sleep(max(0.0, interval - (time.perf_counter() - t0)))
                            continue
                        raw_samples = data   # 12 arrays: V0-5 then I0-5
                    except Exception as e:
                        self.report_error("Acq", str(e))
                        try:
                            task.in_stream.relative_to = ReadRelativeTo.MOST_RECENT_SAMPLE
                            task.in_stream.offset = 0
                        except Exception:
                            pass
                        time.sleep(0.2)
                        continue

                # Compute calibrated RMS for each channel
                for ch in range(N_CH):
                    arr = np.atleast_1d(raw_samples[ch])
                    rms_raw = float(np.sqrt(np.nanmean(arr**2)))
                    if self.v_enabled[ch]:
                        s, o = self.v_cal[ch]
                        self.v_rms[ch] = max(0.0, rms_raw * s + o)

                for ch in range(N_CH):
                    arr = np.atleast_1d(raw_samples[N_CH + ch])
                    rms_raw = float(np.sqrt(np.nanmean(arr**2)))
                    if self.i_enabled[ch]:
                        s, o = self.i_cal[ch]
                        self.i_rms[ch] = max(0.0, rms_raw * s + o)

                elapsed = time.perf_counter() - t0
                time.sleep(max(0.0, interval - elapsed))

        finally:
            if task:
                try:
                    task.stop()
                    task.close()
                except Exception:
                    pass


# ══════════════════════════════════════════════════════════════════════════
#  Power & Efficiency Calculator
# ══════════════════════════════════════════════════════════════════════════
def calc_power_efficiency(v_rms, i_rms, pf=1.0):
    """
    Calculate per-phase apparent power, total input/output power, and
    efficiency for a three-phase motor controller.

    Parameters
    ----------
    v_rms : list[float]  length 6 — [V_A_in, V_B_in, V_C_in,
                                      V_A_out, V_B_out, V_C_out]
    i_rms : list[float]  length 6 — [I_A_in, I_B_in, I_C_in,
                                      I_A_out, I_B_out, I_C_out]
    pf    : float        power factor (default 1.0 = unity / apparent power)

    Returns
    -------
    dict with keys:
      phase_va  : list[float]  — apparent power per phase (VA)
      phase_w   : list[float]  — real power per phase (W)
      p_in_va, p_in_w          — total 3-phase input  power
      p_out_va, p_out_w        — total 3-phase output power
      efficiency               — η % (None if p_in_w == 0)
      losses_w                 — input - output (W)
    """
    phase_va = [v_rms[i] * i_rms[i] for i in range(N_CH)]
    phase_w  = [va * pf              for va in phase_va]

    p_in_va  = sum(phase_va[i] for i in INPUT_PHASES)
    p_in_w   = sum(phase_w[i]  for i in INPUT_PHASES)
    p_out_va = sum(phase_va[i] for i in OUTPUT_PHASES)
    p_out_w  = sum(phase_w[i]  for i in OUTPUT_PHASES)

    efficiency = (p_out_w / p_in_w * 100) if p_in_w > 0.001 else None
    losses_w   = p_in_w - p_out_w

    return dict(
        phase_va=phase_va, phase_w=phase_w,
        p_in_va=p_in_va,   p_in_w=p_in_w,
        p_out_va=p_out_va, p_out_w=p_out_w,
        efficiency=efficiency, losses_w=losses_w
    )


# ══════════════════════════════════════════════════════════════════════════
#  GUI Application
# ══════════════════════════════════════════════════════════════════════════
class MotorEffApp(tk.Tk):

    def __init__(self):
        super().__init__()
        self.title("Motor Controller Efficiency — cDAQ-9189  Mod2/Mod3")
        self.configure(bg=C_BG)
        self.geometry("1300x860")
        self.minsize(1100, 760)

        self.error_queue: queue.Queue = queue.Queue()
        self.daq: Optional[MotorDAQManager] = None
        self._connected_ok = False

        # Recording state
        self._log_file: Optional[str] = None
        self._log_fh   = None
        self._log_writer = None
        self._rec_start_time: Optional[float] = None

        # JSON-loaded config
        self._json_chassis        = "cDAQ9189-XXXXXXX"
        self._json_ip             = "169.254.32.5"
        self._json_auto_start     = False
        self._rec_auto_start      = False
        self._rec_prefix          = "motor_efficiency"
        self._rec_timed           = False
        self._rec_duration_sec    = 300.0
        self._rec_start_delay_sec = 2.0
        self._hw_rate             = 200_000
        self._ac_freq             = 60.0
        self._n_cycles            = 3
        self._pf                  = 1.0   # power factor

        # Per-channel calibration loaded from JSON
        self._v_cal = [(1.0, 0.0)] * N_CH
        self._i_cal = [(1.0, 0.0)] * N_CH
        self._v_enabled = [True] * N_CH
        self._i_enabled = [True] * N_CH

        self._load_from_json()
        self._build_style()
        self._build_ui()

        if SIMULATION_MODE:
            self._auto_connect()

        self._poll()
        self._poll_errors()

    # ── JSON loading ──────────────────────────────────────────────────────
    def _load_from_json(self):
        """Load chassis, IP, rates, channel names, and calibration from JSON."""
        if not os.path.exists(CAL_FILE):
            self.error_queue.put((datetime.now(), "JSON",
                                  f"Not found: {CAL_FILE}"))
            return
        try:
            with open(CAL_FILE) as f:
                data = json.load(f)

            self._json_chassis = data.get("chassis_name", self._json_chassis)
            self._json_ip      = data.get("ip_address",   self._json_ip)
            self._json_auto_start = bool(data.get("auto_start_on_connect", False))

            rec = data.get("recording") or {}
            self._rec_auto_start      = bool(rec.get("auto_record_on_start",  False))
            self._rec_prefix          = str(rec.get("log_filename_prefix",    "motor_efficiency"))
            self._rec_timed           = bool(rec.get("timed_recording",        False))
            self._rec_duration_sec    = float(rec.get("record_duration_sec",   300))
            self._rec_start_delay_sec = float(rec.get("record_start_delay_sec", 2.0))

            mc  = data.get("module_config", {})
            aic = mc.get("modules_2_to_6_AI_9320", {})
            self._ac_freq  = float(aic.get("ac_frequency_hz",  60.0))
            self._n_cycles = int(aic.get("ac_cycles_per_block", 3))

            ht = aic.get("high_rate_task", {})
            self._hw_rate = int(ht.get("hw_sample_rate_hz", 200_000))

            # Load channel names and calibration for Mod2/Mod3 CH0-5
            for rec_ch in data.get("NI_9320_modules_2_to_6", []):
                mod = rec_ch.get("module")
                ch  = rec_ch.get("channel")
                if ch is None or ch >= N_CH:
                    continue
                name    = rec_ch.get("name", "")
                scale   = float(rec_ch.get("scale",  1.0))
                offset  = float(rec_ch.get("offset", 0.0))
                enabled = bool(rec_ch.get("enabled", True))

                if mod == MOD_VOLTAGE:
                    if name: V_NAMES[ch] = name
                    self._v_cal[ch]     = (scale, offset)
                    self._v_enabled[ch] = enabled
                elif mod == MOD_CURRENT:
                    if name: I_NAMES[ch] = name
                    self._i_cal[ch]     = (scale, offset)
                    self._i_enabled[ch] = enabled

        except Exception as e:
            self.error_queue.put((datetime.now(), "JSON",
                                  f"Failed to load: {e}"))

    # ── Style ─────────────────────────────────────────────────────────────
    def _build_style(self):
        s = ttk.Style(self)
        s.theme_use("clam")
        s.configure(".", background=C_BG, foreground=C_TEXT,
                    fieldbackground=C_INPUT, troughcolor=C_BORDER,
                    selectbackground=C_ACCENT, selectforeground="#000")
        s.configure("TNotebook", background=C_BG, tabmargins=[2,4,2,0])
        s.configure("TNotebook.Tab", background=C_PANEL, foreground=C_MUTED,
                    padding=[12,5], font=FONT_MED)
        s.map("TNotebook.Tab",
              background=[("selected", C_BG)],
              foreground=[("selected", C_ACCENT)])
        s.configure("TLabelframe", background=C_BG, foreground=C_ACCENT,
                    relief="flat", borderwidth=1)
        s.configure("TLabelframe.Label", background=C_BG, foreground=C_ACCENT,
                    font=FONT_BOLD)
        for name, fg in [("G.TButton",C_GREEN),("R.TButton",C_RED),
                          ("A.TButton",C_ACCENT),("Y.TButton",C_YELLOW)]:
            s.configure(name, background=C_PANEL, foreground=fg,
                        relief="flat", padding=[8,4], font=FONT_SMALL)
            s.map(name, background=[("active", C_BORDER)])
        s.configure("TCheckbutton", background=C_BG, foreground=C_TEXT,
                    font=FONT_TINY)
        s.configure("Panel.TCheckbutton", background=C_PANEL,
                    foreground=C_TEXT, font=FONT_TINY)
        s.configure("TEntry", fieldbackground=C_INPUT, foreground=C_TEXT,
                    insertcolor=C_TEXT)
        s.configure("TScrollbar", background=C_BORDER, troughcolor=C_PANEL)

    # ── UI ────────────────────────────────────────────────────────────────
    def _build_ui(self):
        # Top bar
        top = tk.Frame(self, bg=C_PANEL, pady=6, padx=14)
        top.pack(fill="x")

        tk.Label(top, text="Motor Efficiency", font=FONT_TITLE,
                 bg=C_PANEL, fg=C_ACCENT).pack(side="left")
        mode_txt = "SIMULATION" if SIMULATION_MODE else "HARDWARE"
        mode_fg  = C_YELLOW if SIMULATION_MODE else C_GREEN
        tk.Label(top, text=f" [{mode_txt}]", font=FONT_SMALL,
                 bg=C_PANEL, fg=mode_fg).pack(side="left", padx=4)

        tk.Label(top, text="  Chassis:", font=FONT_SMALL,
                 bg=C_PANEL, fg=C_MUTED).pack(side="left")
        self._chassis_var = tk.StringVar(value=self._json_chassis)
        tk.Entry(top, textvariable=self._chassis_var, width=18,
                 bg=C_INPUT, fg=C_TEXT, relief="flat",
                 font=FONT_MONO, insertbackground=C_TEXT
                 ).pack(side="left", padx=4)

        tk.Label(top, text="IP:", font=FONT_SMALL,
                 bg=C_PANEL, fg=C_MUTED).pack(side="left")
        self._ip_var = tk.StringVar(value=self._json_ip)
        tk.Entry(top, textvariable=self._ip_var, width=14,
                 bg=C_INPUT, fg=C_TEXT, relief="flat",
                 font=FONT_MONO, insertbackground=C_TEXT
                 ).pack(side="left", padx=4)

        self._conn_btn = ttk.Button(top, text="Connect",
                                    style="G.TButton",
                                    command=self._connect)
        self._conn_btn.pack(side="left", padx=6)

        self._start_btn = ttk.Button(top, text="▶ Start",
                                     style="G.TButton",
                                     command=self._start_acq)
        self._start_btn.pack(side="left", padx=2)

        self._stop_btn = ttk.Button(top, text="■ Stop",
                                    style="R.TButton",
                                    command=self._stop_acq)
        self._stop_btn.pack(side="left", padx=2)

        self._log_btn = ttk.Button(top, text="▶ Start CSV Capture",
                                   style="Y.TButton",
                                   command=self._toggle_log)
        self._log_btn.pack(side="right")

        self._status_lbl = tk.Label(top, text="● Disconnected",
                                    font=FONT_SMALL, bg=C_PANEL, fg=C_RED)
        self._status_lbl.pack(side="left", padx=8)

        # Power factor entry
        tk.Label(top, text="  PF:", font=FONT_SMALL,
                 bg=C_PANEL, fg=C_MUTED).pack(side="left")
        self._pf_var = tk.StringVar(value=str(self._pf))
        tk.Entry(top, textvariable=self._pf_var, width=5,
                 bg=C_INPUT, fg=C_TEXT, relief="flat",
                 font=FONT_MONO, insertbackground=C_TEXT
                 ).pack(side="left", padx=2)

        # Notebook
        self._nb = ttk.Notebook(self)
        self._nb.pack(fill="both", expand=True, padx=6, pady=6)

        self._build_efficiency_tab()
        self._build_channel_tab()
        self._build_cal_tab()

        # Error bar
        self._error_log: list = []
        bottom_wrap = tk.Frame(self, bg="#1c1106")
        bottom_wrap.pack(fill="x", side="bottom")

        self._err_log_frame = tk.Frame(bottom_wrap, bg="#0d0701")
        self._err_log_text  = tk.Text(
            self._err_log_frame, height=8, bg="#0d0701",
            fg=C_RED, font=FONT_MONO_S, wrap="none",
            relief="flat", state="disabled")
        err_vsb = ttk.Scrollbar(self._err_log_frame, orient="vertical",
                                 command=self._err_log_text.yview)
        self._err_log_text.configure(yscrollcommand=err_vsb.set)
        self._err_log_text.pack(side="left", fill="both",
                                expand=True, padx=(10,0), pady=4)
        err_vsb.pack(side="right", fill="y", pady=4)
        self._err_log_visible = False

        bottom = tk.Frame(bottom_wrap, bg="#1c1106", height=28)
        bottom.pack(fill="x", side="bottom")
        bottom.pack_propagate(False)

        tk.Label(bottom, text="Status:", font=FONT_TINY,
                 bg="#1c1106", fg=C_MUTED).pack(side="left", padx=(10,4))
        self._err_lbl = tk.Label(bottom, text="No errors",
                                  font=FONT_TINY, bg="#1c1106",
                                  fg=C_MUTED, anchor="w")
        self._err_lbl.pack(side="left", fill="x", expand=True)

        self._err_toggle_btn = ttk.Button(
            bottom, text="Show Log (0)", style="A.TButton",
            command=self._toggle_error_log)
        self._err_toggle_btn.pack(side="right", padx=4, pady=2)
        ttk.Button(bottom, text="Clear", style="R.TButton",
                   command=self._clear_error).pack(side="right", padx=4, pady=2)

        # Auto-connect if JSON flag set
        if self._json_auto_start and not SIMULATION_MODE:
            self.after(200, self._connect)

    # ── Efficiency tab ────────────────────────────────────────────────────
    def _build_efficiency_tab(self):
        tab = tk.Frame(self._nb, bg=C_BG)
        self._nb.add(tab, text="  Efficiency  ")

        # Big efficiency display
        eff_frame = tk.Frame(tab, bg=C_PANEL, padx=16, pady=12,
                             highlightbackground=C_BORDER, highlightthickness=1)
        eff_frame.pack(fill="x", padx=10, pady=8)

        tk.Label(eff_frame, text="EFFICIENCY", font=FONT_BOLD,
                 bg=C_PANEL, fg=C_MUTED).grid(row=0, column=0,
                 columnspan=3, pady=(0,4))

        self._eff_var = tk.StringVar(value="---")
        tk.Label(eff_frame, textvariable=self._eff_var,
                 font=("Courier New", 48, "bold"),
                 bg=C_PANEL, fg=C_GREEN).grid(row=1, column=0,
                 columnspan=3, padx=20)

        # Input / Output / Losses summary
        summary = tk.Frame(eff_frame, bg=C_PANEL)
        summary.grid(row=2, column=0, columnspan=3, pady=8)

        def _big_lbl(parent, label, col, fg):
            f = tk.Frame(parent, bg=C_PANEL)
            f.grid(row=0, column=col, padx=20)
            tk.Label(f, text=label, font=FONT_SMALL,
                     bg=C_PANEL, fg=C_MUTED).pack()
            var = tk.StringVar(value="---")
            tk.Label(f, textvariable=var, font=FONT_MONO_L,
                     bg=C_PANEL, fg=fg).pack()
            return var

        self._pin_var    = _big_lbl(summary, "INPUT  (W)",   0, C_BLUE)
        self._pout_var   = _big_lbl(summary, "OUTPUT (W)",   1, C_ACCENT)
        self._loss_var   = _big_lbl(summary, "LOSSES (W)",   2, C_ORANGE)
        self._pin_va_var = _big_lbl(summary, "INPUT  (VA)",  3, C_MUTED)
        self._pout_va_var= _big_lbl(summary, "OUTPUT (VA)", 4, C_MUTED)

        # Per-phase power table
        phase_frame = ttk.LabelFrame(tab,
            text=" Per-Phase Power  (Apparent VA  |  Real W  |  Vrms  |  Irms )",
            padding=8)
        phase_frame.pack(fill="x", padx=10, pady=4)

        ph_hdrs = ["Phase", "Signal", "Vrms (V)", "Irms (A)",
                   "S (VA)", "P (W)", "Side"]
        for c, h in enumerate(ph_hdrs):
            tk.Label(phase_frame, text=h, font=FONT_BOLD,
                     bg=C_BG, fg=C_MUTED, width=12, anchor="w"
                     ).grid(row=0, column=c, padx=4, pady=2, sticky="w")

        self._phase_vars = []   # list of dict per phase
        phase_labels = ["A-IN","B-IN","C-IN","A-OUT","B-OUT","C-OUT"]
        sides        = ["Input","Input","Input","Output","Output","Output"]

        for i in range(N_CH):
            row_f = tk.Frame(phase_frame, bg=C_PANEL if i%2==0 else C_BG)
            row_f.grid(row=i+1, column=0, columnspan=7,
                       sticky="ew", padx=2, pady=1)
            phase_frame.rowconfigure(i+1, weight=1)

            d = {}
            tk.Label(row_f, text=phase_labels[i], font=FONT_BOLD,
                     bg=row_f.cget("bg"), fg=C_TEXT, width=7,
                     anchor="w").pack(side="left", padx=4)

            # Signal name (from JSON)
            d["name_lbl"] = tk.Label(row_f, text=V_NAMES[i],
                                      font=FONT_TINY, bg=row_f.cget("bg"),
                                      fg=C_MUTED, width=24, anchor="w")
            d["name_lbl"].pack(side="left", padx=4)

            for key, fg in [("vrms",C_ACCENT),("irms",C_BLUE),
                             ("va",C_TEXT),("w",C_GREEN)]:
                var = tk.StringVar(value="---")
                d[key] = var
                tk.Label(row_f, textvariable=var, width=12,
                         font=FONT_MONO, bg=row_f.cget("bg"),
                         fg=fg, anchor="e").pack(side="left", padx=4)

            side_fg = C_BLUE if i in INPUT_PHASES else C_ACCENT
            tk.Label(row_f, text=sides[i], font=FONT_TINY,
                     bg=row_f.cget("bg"), fg=side_fg,
                     width=8).pack(side="left", padx=4)

            self._phase_vars.append(d)

        return tab

    # ── Channel tab ───────────────────────────────────────────────────────
    def _build_channel_tab(self):
        tab = tk.Frame(self._nb, bg=C_BG)
        self._nb.add(tab, text="  Channels  ")

        body = tk.Frame(tab, bg=C_BG)
        body.pack(fill="both", expand=True, padx=10, pady=8)

        # Voltage (left) and Current (right) side by side
        self._v_vars    = []
        self._i_vars    = []
        self._v_checks  = []
        self._i_checks  = []

        for side, label, names, checks, vars_, col in [
            ("Voltage", "Module 2  ·  Voltage Inputs  [Vrms]",
             V_NAMES, self._v_checks, self._v_vars, 0),
            ("Current", "Module 3  ·  Current Inputs  [Arms]",
             I_NAMES, self._i_checks, self._i_vars, 1),
        ]:
            fg = C_ACCENT if side == "Voltage" else C_BLUE
            sec = ttk.LabelFrame(body, text=f" {label} ", padding=8)
            sec.grid(row=0, column=col, padx=6, pady=4, sticky="nsew")
            body.columnconfigure(col, weight=1)

            hdr = tk.Frame(sec, bg=C_BG)
            hdr.pack(fill="x")
            for txt, w in [("En","3"),("Ch","4"),("Signal Name","24"),
                           ("RMS Value","12")]:
                tk.Label(hdr, text=txt, font=FONT_BOLD,
                         bg=C_BG, fg=C_MUTED, width=int(w),
                         anchor="w").pack(side="left", padx=2)

            for ch in range(N_CH):
                row = tk.Frame(sec, bg=C_BG)
                row.pack(fill="x", pady=2)

                chk = tk.BooleanVar(value=(self._v_enabled[ch]
                                           if side=="Voltage"
                                           else self._i_enabled[ch]))
                checks.append(chk)
                ttk.Checkbutton(row, variable=chk, width=3,
                                style="TCheckbutton",
                                command=lambda s=side, c=ch:
                                    self._ch_toggle(s, c)
                                ).pack(side="left")

                tk.Label(row, text=f"{ch}", font=FONT_MONO_S,
                         bg=C_BG, fg=C_TEXT, width=4,
                         anchor="w").pack(side="left", padx=2)
                tk.Label(row, text=names[ch], font=FONT_TINY,
                         bg=C_BG, fg=C_MUTED, width=24,
                         anchor="w").pack(side="left", padx=2)

                var = tk.StringVar(value="---")
                vars_.append(var)
                tk.Label(row, textvariable=var, width=12,
                         font=FONT_MONO, bg=C_INPUT, fg=fg,
                         anchor="e", padx=4).pack(side="left", padx=2)

        return tab

    # ── Calibration tab ───────────────────────────────────────────────────
    def _build_cal_tab(self):
        tab = tk.Frame(self._nb, bg=C_BG)
        self._nb.add(tab, text="  Calibration  ")

        hdr = tk.Frame(tab, bg=C_BG)
        hdr.pack(fill="x", padx=10, pady=4)
        tk.Label(hdr, text="Scale & Offset — output = (raw × scale) + offset",
                 font=FONT_HEAD, bg=C_BG, fg=C_ACCENT).pack(side="left")
        ttk.Button(hdr, text="Apply", style="G.TButton",
                   command=self._apply_cal).pack(side="right")

        body = tk.Frame(tab, bg=C_BG)
        body.pack(fill="both", expand=True, padx=10, pady=4)

        self._v_cal_scale:  list[tk.StringVar] = []
        self._v_cal_offset: list[tk.StringVar] = []
        self._i_cal_scale:  list[tk.StringVar] = []
        self._i_cal_offset: list[tk.StringVar] = []

        for side, label, names, slist, olist, cal_data, col in [
            ("Voltage", "Module 2  ·  Voltage",
             V_NAMES, self._v_cal_scale, self._v_cal_offset, self._v_cal, 0),
            ("Current", "Module 3  ·  Current",
             I_NAMES, self._i_cal_scale, self._i_cal_offset, self._i_cal, 1),
        ]:
            sec = ttk.LabelFrame(body, text=f" {label} ", padding=8)
            sec.grid(row=0, column=col, padx=6, pady=4, sticky="nsew")
            body.columnconfigure(col, weight=1)

            for txt, w in [("Ch","4"),("Signal Name","24"),
                           ("Scale","10"),("Offset","10")]:
                tk.Label(sec, text=txt, font=FONT_BOLD, bg=C_BG,
                         fg=C_MUTED, width=int(w), anchor="w"
                         ).grid(row=0,
                                column=["Ch","Signal Name",
                                        "Scale","Offset"].index(txt),
                                padx=4, sticky="w")

            for ch in range(N_CH):
                s_var = tk.StringVar(value=str(cal_data[ch][0]))
                o_var = tk.StringVar(value=str(cal_data[ch][1]))
                slist.append(s_var)
                olist.append(o_var)

                tk.Label(sec, text=str(ch), font=FONT_MONO_S,
                         bg=C_BG, fg=C_TEXT, width=4, anchor="w"
                         ).grid(row=ch+1, column=0, padx=4, pady=1, sticky="w")
                tk.Label(sec, text=names[ch], font=FONT_TINY,
                         bg=C_BG, fg=C_MUTED, width=24, anchor="w"
                         ).grid(row=ch+1, column=1, padx=4, pady=1, sticky="w")
                tk.Entry(sec, textvariable=s_var, width=10,
                         bg=C_INPUT, fg=C_TEXT, font=FONT_MONO_S,
                         insertbackground=C_TEXT, relief="flat"
                         ).grid(row=ch+1, column=2, padx=4, pady=1)
                tk.Entry(sec, textvariable=o_var, width=10,
                         bg=C_INPUT, fg=C_TEXT, font=FONT_MONO_S,
                         insertbackground=C_TEXT, relief="flat"
                         ).grid(row=ch+1, column=3, padx=4, pady=1)

        return tab

    # ── Connection ────────────────────────────────────────────────────────
    def _connect(self):
        chassis = self._chassis_var.get().strip()
        ip      = self._ip_var.get().strip()
        self.daq = MotorDAQManager(chassis, ip, self.error_queue)
        self._apply_cal_to_daq()

        if SIMULATION_MODE:
            self._auto_connect()
            return

        self._conn_btn.config(state="disabled", text="Testing...")
        self._status_lbl.config(text="Testing connection...", fg=C_YELLOW)

        def worker():
            ok, msg = self.daq.test_connection()
            self.after(0, lambda: self._on_connect_result(ok, msg))

        threading.Thread(target=worker, daemon=True).start()

    def _auto_connect(self):
        """Used by simulation mode and auto_start_on_connect."""
        chassis = self._chassis_var.get().strip()
        ip      = self._ip_var.get().strip()
        self.daq = MotorDAQManager(chassis, ip, self.error_queue)
        self._connected_ok = True
        self._apply_cal_to_daq()
        self._conn_btn.config(text="Reconnect")
        mode = "SIMULATION" if SIMULATION_MODE else "HARDWARE"
        self._status_lbl.config(
            text=f"● Connected ({mode})", fg=C_GREEN)
        if self._json_auto_start:
            self.after(500, self._start_acq)

    def _on_connect_result(self, ok: bool, msg: str):
        self._conn_btn.config(state="normal", text="Reconnect")
        self._connected_ok = ok
        if ok:
            self._status_lbl.config(text=f"● Connected — {msg}", fg=C_GREEN)
            if self._json_auto_start:
                self.after(500, self._start_acq)
        else:
            self._status_lbl.config(text="● Connection failed", fg=C_RED)
            self.error_queue.put((datetime.now(), "Connection", msg))
            messagebox.showerror("Connection Failed", msg)

    # ── Acquisition ───────────────────────────────────────────────────────
    def _ensure_connected(self) -> bool:
        if not self.daq or (not SIMULATION_MODE and not self._connected_ok):
            messagebox.showwarning("Not Connected",
                                   "Connect to the chassis first.")
            return False
        return True

    def _start_acq(self):
        if not self._ensure_connected():
            return
        self.daq.hw_rate  = self._hw_rate
        self.daq.ac_freq  = self._ac_freq
        self.daq.n_cycles = self._n_cycles
        self.daq.start()
        self._status_lbl.config(
            text=f"● Running  {self._hw_rate:,} S/s  |  "
                 f"{self._n_cycles} cycles @ {self._ac_freq:.0f} Hz",
            fg=C_GREEN)
        if self._rec_auto_start:
            delay_ms = max(100, int(self._rec_start_delay_sec * 1000))
            self.after(delay_ms, self._start_auto_recording)

    def _stop_acq(self):
        if self.daq:
            self.daq.stop()
        self._stop_recording()
        self._status_lbl.config(text="● Stopped", fg=C_MUTED)

    # ── Channel enable ────────────────────────────────────────────────────
    def _ch_toggle(self, side: str, ch: int):
        if not self.daq:
            return
        if side == "Voltage":
            self.daq.v_enabled[ch] = self._v_checks[ch].get()
        else:
            self.daq.i_enabled[ch] = self._i_checks[ch].get()

    # ── Calibration ───────────────────────────────────────────────────────
    def _apply_cal(self):
        errors = []
        for ch in range(N_CH):
            try:
                s = float(self._v_cal_scale[ch].get())
                o = float(self._v_cal_offset[ch].get())
                self._v_cal[ch] = (s, o)
            except ValueError:
                errors.append(f"V CH{ch}")
            try:
                s = float(self._i_cal_scale[ch].get())
                o = float(self._i_cal_offset[ch].get())
                self._i_cal[ch] = (s, o)
            except ValueError:
                errors.append(f"I CH{ch}")

        if errors:
            messagebox.showwarning("Invalid values",
                                   "Cannot parse:\n" + "\n".join(errors))
            return
        self._apply_cal_to_daq()
        messagebox.showinfo("Calibration Applied",
                            "Scale/offset values applied.")

    def _apply_cal_to_daq(self):
        if not self.daq:
            return
        for ch in range(N_CH):
            self.daq.v_cal[ch] = self._v_cal[ch]
            self.daq.i_cal[ch] = self._i_cal[ch]

    # ── CSV recording ─────────────────────────────────────────────────────
    def _make_log_path(self) -> str:
        import sys
        try:
            script_dir = os.path.dirname(os.path.abspath(sys.argv[0]))
        except Exception:
            script_dir = os.getcwd()
        ts   = datetime.now().strftime("%Y%m%d_%H%M%S")
        name = f"{self._rec_prefix}_{ts}.csv"
        return os.path.join(script_dir, name)

    def _start_recording(self, path: str):
        self._log_file = path
        hdrs = (["Timestamp"]
                + [f"V{c}_{V_NAMES[c]}_Vrms" for c in range(N_CH)]
                + [f"I{c}_{I_NAMES[c]}_Arms"  for c in range(N_CH)]
                + [f"S{c}_VA" for c in range(N_CH)]
                + [f"P{c}_W"  for c in range(N_CH)]
                + ["P_in_VA","P_in_W","P_out_VA","P_out_W",
                   "Losses_W","Efficiency_pct"])
        self._log_fh     = open(path, "w", newline="")
        self._log_writer = csv.writer(self._log_fh)
        self._log_writer.writerow(hdrs)
        self._log_fh.flush()
        self._rec_start_time = time.perf_counter()
        self.daq.logging = True
        self._log_btn.config(text="■ Stop CSV Capture")
        threading.Thread(target=self._csv_writer_loop, daemon=True).start()

    def _stop_recording(self):
        if self.daq:
            self.daq.logging = False
        self._rec_start_time = None
        self._log_btn.config(text="▶ Start CSV Capture")

    def _start_auto_recording(self):
        if not self.daq:
            return
        path = self._make_log_path()
        self._start_recording(path)
        self.error_queue.put((datetime.now(), "Auto-Record",
                              f"Recording: {os.path.basename(path)}"
                              + (f" [{self._rec_duration_sec:.0f}s]"
                                 if self._rec_timed else "")))

    def _toggle_log(self):
        if not self._ensure_connected():
            return
        if not self.daq.logging:
            import sys
            try:
                init_dir = os.path.dirname(os.path.abspath(sys.argv[0]))
            except Exception:
                init_dir = None
            path = filedialog.asksaveasfilename(
                defaultextension=".csv",
                filetypes=[("CSV", "*.csv"), ("All", "*.*")],
                title="Save Motor Efficiency CSV",
                initialdir=init_dir,
                initialfile=os.path.basename(self._make_log_path()),
            )
            if not path:
                return
            self._start_recording(path)
        else:
            self._stop_recording()

    def _csv_writer_loop(self):
        """Write one row per acquisition block with V, I, power, and efficiency."""
        if not self.daq:
            return
        interval = self._n_cycles / max(1.0, self._ac_freq)

        def fmt(v):
            try:
                return "" if not np.isfinite(v) else f"{v:.6f}"
            except Exception:
                return ""

        while self.daq and self.daq.logging:
            t0     = time.perf_counter()
            ts_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")

            v = list(self.daq.v_rms)
            i = list(self.daq.i_rms)
            try:
                pf  = float(self._pf_var.get())
            except Exception:
                pf  = 1.0
            res = calc_power_efficiency(v, i, pf)

            row = ([ts_str]
                   + [fmt(x) for x in v]
                   + [fmt(x) for x in i]
                   + [fmt(x) for x in res["phase_va"]]
                   + [fmt(x) for x in res["phase_w"]]
                   + [fmt(res["p_in_va"]),  fmt(res["p_in_w"]),
                      fmt(res["p_out_va"]), fmt(res["p_out_w"]),
                      fmt(res["losses_w"]),
                      fmt(res["efficiency"]) if res["efficiency"] is not None else ""])

            if self._log_writer:
                self._log_writer.writerow(row)
                self._log_fh.flush()

            elapsed = time.perf_counter() - t0
            time.sleep(max(0.0, interval - elapsed))

        if self._log_fh:
            self._log_fh.close()
            self._log_fh    = None
            self._log_writer = None

    def _timed_recording_complete(self):
        self._stop_acq()
        saved = os.path.basename(self._log_file or "unknown")
        self.error_queue.put((datetime.now(), "Timed Recording",
                              f"Complete after {self._rec_duration_sec:.0f}s "
                              f"— saved: {saved}"))
        if self._rec_auto_start and self._rec_timed:
            self._status_lbl.config(
                text=f"● Recording complete — closing in 1s...",
                fg=C_YELLOW)
            self.after(1000, self.destroy)
        else:
            self._status_lbl.config(
                text=f"● Timed recording complete ({self._rec_duration_sec:.0f}s)",
                fg=C_YELLOW)

    # ── Error log ─────────────────────────────────────────────────────────
    _MAX_ERROR_LOG = 500

    def _toggle_error_log(self):
        self._err_log_visible = not self._err_log_visible
        if self._err_log_visible:
            self._err_log_frame.pack(fill="both", side="bottom")
        else:
            self._err_log_frame.pack_forget()
        self._err_toggle_btn.config(
            text=("Hide" if self._err_log_visible else "Show Log")
            + f" ({len(self._error_log)})")

    def _clear_error(self):
        self._err_lbl.config(text="No errors", fg=C_MUTED)
        self._error_log.clear()
        self._err_log_text.config(state="normal")
        self._err_log_text.delete("1.0", "end")
        self._err_log_text.config(state="disabled")
        self._err_toggle_btn.config(
            text=("Hide" if self._err_log_visible else "Show Log") + " (0)")

    # ── Poll ──────────────────────────────────────────────────────────────
    def _poll(self):
        if self.daq:
            v = list(self.daq.v_rms)
            i = list(self.daq.i_rms)

            for ch in range(N_CH):
                self._v_vars[ch].set(f"{v[ch]:9.4f}")
                self._i_vars[ch].set(f"{i[ch]:9.4f}")

            try:
                pf = float(self._pf_var.get())
            except Exception:
                pf = 1.0

            res = calc_power_efficiency(v, i, pf)

            for ch in range(N_CH):
                d = self._phase_vars[ch]
                d["vrms"].set(f"{v[ch]:9.3f} V")
                d["irms"].set(f"{i[ch]:9.3f} A")
                d["va"].set(f"{res['phase_va'][ch]:9.2f} VA")
                d["w"].set(f"{res['phase_w'][ch]:9.2f} W")
                d["name_lbl"].config(text=V_NAMES[ch])

            if res["efficiency"] is not None:
                eff = res["efficiency"]
                fg  = C_GREEN if eff >= 90 else C_YELLOW if eff >= 70 else C_RED
                self._eff_var.set(f"{eff:.2f}%")
                # recolour the big label
                for w in self.winfo_children():
                    pass  # colour applied at creation; update via config below
            else:
                self._eff_var.set("---")

            self._pin_var.set(f"{res['p_in_w']:8.2f}")
            self._pout_var.set(f"{res['p_out_w']:8.2f}")
            self._loss_var.set(f"{res['losses_w']:8.2f}")
            self._pin_va_var.set(f"{res['p_in_va']:8.2f}")
            self._pout_va_var.set(f"{res['p_out_va']:8.2f}")

            # Timed recording check
            if (self._rec_timed and self._rec_start_time is not None
                    and self.daq.logging):
                elapsed   = time.perf_counter() - self._rec_start_time
                remaining = max(0.0, self._rec_duration_sec - elapsed)
                m, s = divmod(int(elapsed), 60)
                self._log_btn.config(
                    text=f"■ Recording  {remaining:.0f}s left")
                if elapsed >= self._rec_duration_sec:
                    self._timed_recording_complete()
            elif (self.daq.logging and not self._rec_timed
                  and self._rec_start_time is not None):
                elapsed = time.perf_counter() - self._rec_start_time
                m, s = divmod(int(elapsed), 60)
                self._log_btn.config(text=f"■ Recording  {m:02d}:{s:02d}")

        self.after(100, self._poll)   # 10 Hz — efficiency calc is the bottleneck

    def _poll_errors(self):
        new_items = []
        try:
            while True:
                new_items.append(self.error_queue.get_nowait())
        except queue.Empty:
            pass

        if new_items:
            self._error_log.extend(new_items)
            if len(self._error_log) > self._MAX_ERROR_LOG:
                self._error_log = self._error_log[-self._MAX_ERROR_LOG:]

            ts, source, message = new_items[-1]
            self._err_lbl.config(
                text=f"[{ts.strftime('%H:%M:%S')}] {source}: {message}"
                + (f"  (+{len(new_items)-1} more)" if len(new_items)>1 else ""),
                fg=C_RED)

            lines = []
            for ts, source, msg in new_items[-50:]:
                lines.append(f"[{ts.strftime('%H:%M:%S')}] {source}: {msg}")

            self._err_log_text.config(state="normal")
            self._err_log_text.insert("end", "\n".join(lines) + "\n")
            self._err_log_text.see("end")
            self._err_log_text.config(state="disabled")

            self._err_toggle_btn.config(
                text=("Hide" if self._err_log_visible else "Show Log")
                + f" ({len(self._error_log)})")

        self.after(300, self._poll_errors)

    # ── Close ─────────────────────────────────────────────────────────────
    def destroy(self):
        if self.daq:
            self.daq.stop()
            self.daq.logging = False
        if self._log_fh:
            try:
                self._log_fh.close()
            except Exception:
                pass
        super().destroy()


# ══════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    app = MotorEffApp()
    app.mainloop()

