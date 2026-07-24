"""
Three-Phase Motor Controller Efficiency Analyser
=================================================
NI cDAQ-9189 branch — Modules 2 & 3, Channels 0-5 only.

ACQUISITION
  Records raw voltage and current waveforms at 200 kS/s to a binary
  NumPy (.npy) file.  No processing happens during capture — the
  acquisition loop only accumulates samples and flushes blocks to disk.
  This keeps the loop lean and leaves all signal processing to the
  post-processing step.

  File layout:  float32 array  shape (N_samples, 12)
    Columns 0-5 : Voltage  CH0-5  (raw sensor volts, pre-calibration)
    Columns 6-11: Current  CH0-5  (raw sensor amps,  pre-calibration)

POST-PROCESSING (separate tab)
  Load any .npy file captured by this script and compute:
    • Vrms, Irms per phase
    • Real power P (W)  = mean(v × i) per phase
    • Apparent power S (VA) = Vrms × Irms
    • Power factor PF = P / S
    • Total 3-phase input  power (CH0-2)
    • Total 3-phase output power (CH3-5)
    • Efficiency η = Pout / Pin × 100 %
    • Losses = Pin − Pout

  Scale/offset calibration from the JSON is applied during
  post-processing, so you can recalibrate and re-run without
  re-capturing data.

Hardware:
  Module 2  CH0-5  NI 9320  Voltage inputs
    CH0-2 : input  side Phase A/B/C voltage
    CH3-5 : output side Phase A/B/C voltage
  Module 3  CH0-5  NI 9320  Current inputs
    CH0-2 : input  side Phase A/B/C current
    CH3-5 : output side Phase A/B/C current

Dependencies:  pip install nidaqmx numpy
Reads:         cdaq_calibration.json  (same folder as this script)
"""

import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import threading
import time
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
    print("[WARNING] nidaqmx not found — running in SIMULATION mode")


# ══════════════════════════════════════════════════════════════════════════
#  Constants
# ══════════════════════════════════════════════════════════════════════════
N_CH        = 6           # channels per module (0-5)
N_RAW       = N_CH * 2   # total raw columns (6V + 6I)
MOD_V       = 2           # Mod2 → voltage
MOD_I       = 3           # Mod3 → current
INPUT_PH    = [0, 1, 2]  # CH0-2 → input side
OUTPUT_PH   = [3, 4, 5]  # CH3-5 → output side

# Channel names — overwritten from JSON on startup
V_NAMES = [f"V_PH{s}" for s in ["A_IN","B_IN","C_IN","A_OUT","B_OUT","C_OUT"]]
I_NAMES = [f"I_PH{s}" for s in ["A_IN","B_IN","C_IN","A_OUT","B_OUT","C_OUT"]]

# ── colours & fonts ───────────────────────────────────────────────────────
C_BG     = "#0d1117"
C_PANEL  = "#161b22"
C_BORDER = "#30363d"
C_ACCENT = "#00b4d8"
C_GREEN  = "#39d353"
C_RED    = "#f85149"
C_YELLOW = "#e3b341"
C_ORANGE = "#f0883e"
C_BLUE   = "#58a6ff"
C_TEXT   = "#e6edf3"
C_MUTED  = "#8b949e"
C_INPUT  = "#21262d"

FONT_MONO   = ("Courier New", 9)
FONT_MONO_S = ("Courier New", 8)
FONT_MONO_L = ("Courier New", 16, "bold")
FONT_MONO_XL= ("Courier New", 36, "bold")
FONT_SMALL  = ("Segoe UI", 9)
FONT_TINY   = ("Segoe UI", 8)
FONT_BOLD   = ("Segoe UI", 10, "bold")
FONT_HEAD   = ("Segoe UI", 12, "bold")
FONT_TITLE  = ("Segoe UI", 14, "bold")


# ══════════════════════════════════════════════════════════════════════════
#  JSON calibration file (shared with main script)
# ══════════════════════════════════════════════════════════════════════════
def _find_cal_file() -> str:
    import sys
    candidates = []
    try:
        d = os.path.dirname(os.path.abspath(sys.argv[0]))
        if os.path.isdir(d):
            candidates.append(os.path.join(d, "cdaq_calibration.json"))
    except Exception:
        pass
    try:
        d = os.path.dirname(os.path.abspath(__file__))
        p = os.path.join(d, "cdaq_calibration.json")
        if p not in candidates:
            candidates.append(p)
    except NameError:
        pass
    cwd = os.path.join(os.getcwd(), "cdaq_calibration.json")
    if cwd not in candidates:
        candidates.append(cwd)

    print("[Motor Efficiency] Searching for cdaq_calibration.json:")
    for p in candidates:
        exists = os.path.exists(p)
        print(f"  {'FOUND' if exists else 'not found':10s}  {p}")
        if exists:
            return p
    print(f"[Motor Efficiency] JSON not found — default: {candidates[0]}")
    return candidates[0]

CAL_FILE = _find_cal_file()


# ══════════════════════════════════════════════════════════════════════════
#  Post-processing: all signal math lives here
# ══════════════════════════════════════════════════════════════════════════
def process_block(raw: np.ndarray,
                  v_cal: list, i_cal: list,
                  ac_freq: float, fs: float) -> dict:
    """
    Compute Vrms, Irms, P, S, PF, efficiency from a raw sample block.

    Parameters
    ----------
    raw    : float32 array  shape (N, 12) — columns 0-5 voltage, 6-11 current
    v_cal  : list of (scale, offset) per voltage channel
    i_cal  : list of (scale, offset) per current channel
    ac_freq: mains/signal frequency  (Hz)
    fs     : hardware sample rate    (S/s)

    Returns
    -------
    dict with keys: vrms, irms, P, S, pf, Q,
                    p_in, p_out, s_in, s_out,
                    pf_in, pf_out, losses, efficiency
    """
    if raw.ndim == 1:
        raw = raw.reshape(-1, N_RAW)

    # Align to complete cycles to eliminate spectral leakage in mean(v*i)
    samples_per_cycle = fs / ac_freq
    n_complete = max(1, int(raw.shape[0] // samples_per_cycle))
    n_use      = int(round(n_complete * samples_per_cycle))
    raw        = raw[:n_use]

    # Apply calibration
    V = np.empty((raw.shape[0], N_CH), dtype=np.float64)
    I = np.empty((raw.shape[0], N_CH), dtype=np.float64)
    for ch in range(N_CH):
        s, o = v_cal[ch]
        V[:, ch] = raw[:, ch] * s + o
        s, o = i_cal[ch]
        I[:, ch] = raw[:, N_CH + ch] * s + o

    # Per-phase quantities
    vrms = np.sqrt(np.nanmean(V**2, axis=0))       # Vrms per channel
    irms = np.sqrt(np.nanmean(I**2, axis=0))       # Irms per channel
    P    = np.nanmean(V * I, axis=0)               # real power   W
    S    = vrms * irms                              # apparent pwr VA
    Q    = np.sqrt(np.maximum(0, S**2 - P**2))     # reactive pwr VAR
    pf   = np.where(S > 0.001, P / S, 0.0)         # power factor

    # Three-phase totals
    p_in   = float(P[INPUT_PH].sum())
    p_out  = float(P[OUTPUT_PH].sum())
    s_in   = float(S[INPUT_PH].sum())
    s_out  = float(S[OUTPUT_PH].sum())
    pf_in  = p_in  / s_in  if s_in  > 0.001 else 0.0
    pf_out = p_out / s_out if s_out > 0.001 else 0.0
    losses = p_in  - p_out
    eff    = (p_out / p_in * 100) if p_in > 0.001 else None

    return dict(
        vrms=vrms, irms=irms, P=P, S=S, Q=Q, pf=pf,
        p_in=p_in, p_out=p_out, s_in=s_in, s_out=s_out,
        pf_in=pf_in, pf_out=pf_out,
        losses=losses, efficiency=eff,
        n_samples=raw.shape[0], n_cycles=n_complete
    )


def process_file(path: str, v_cal: list, i_cal: list,
                 ac_freq: float, fs: float,
                 progress_cb=None) -> dict:
    """
    Load a raw .npy file and return per-cycle statistics.

    The file is processed in chunks (one second of data each) to keep
    memory usage bounded regardless of file size.  progress_cb(pct) is
    called with 0-100 as chunks are processed.
    """
    data = np.load(path, mmap_mode='r')   # memory-mapped — no full load
    if data.ndim == 1:
        data = data.reshape(-1, N_RAW)
    N          = data.shape[0]
    chunk_size = int(fs)    # 1 second per chunk

    # accumulators
    all_vrms  = []
    all_irms  = []
    all_P     = []
    all_S     = []
    all_pf    = []
    all_p_in  = []
    all_p_out = []
    all_eff   = []

    n_chunks = max(1, N // chunk_size)
    for idx in range(n_chunks):
        start = idx * chunk_size
        end   = min(start + chunk_size, N)
        chunk = np.array(data[start:end], dtype=np.float64)
        if chunk.shape[0] < 2:
            break
        res = process_block(chunk, v_cal, i_cal, ac_freq, fs)
        all_vrms.append(res["vrms"])
        all_irms.append(res["irms"])
        all_P.append(res["P"])
        all_S.append(res["S"])
        all_pf.append(res["pf"])
        all_p_in.append(res["p_in"])
        all_p_out.append(res["p_out"])
        all_eff.append(res["efficiency"] if res["efficiency"] is not None else np.nan)
        if progress_cb:
            progress_cb(int((idx + 1) / n_chunks * 100))

    if not all_vrms:
        return {}

    # Stack into time-series arrays (one row per second)
    ts_vrms  = np.array(all_vrms)   # (n_chunks, 6)
    ts_irms  = np.array(all_irms)
    ts_P     = np.array(all_P)
    ts_S     = np.array(all_S)
    ts_pf    = np.array(all_pf)
    ts_p_in  = np.array(all_p_in)
    ts_p_out = np.array(all_p_out)
    ts_eff   = np.array(all_eff)

    # Summary statistics
    def safe_nanmean(a): return float(np.nanmean(a)) if len(a) else 0.0

    return dict(
        # time-series (one value per second chunk)
        ts_vrms=ts_vrms, ts_irms=ts_irms,
        ts_P=ts_P, ts_S=ts_S, ts_pf=ts_pf,
        ts_p_in=ts_p_in, ts_p_out=ts_p_out, ts_eff=ts_eff,
        n_chunks=n_chunks, fs=fs, ac_freq=ac_freq,
        # averages over full file
        avg_vrms  = ts_vrms.mean(axis=0),
        avg_irms  = ts_irms.mean(axis=0),
        avg_P     = ts_P.mean(axis=0),
        avg_S     = ts_S.mean(axis=0),
        avg_pf    = ts_pf.mean(axis=0),
        avg_p_in  = safe_nanmean(ts_p_in),
        avg_p_out = safe_nanmean(ts_p_out),
        avg_eff   = float(np.nanmean(ts_eff)),
        avg_losses= safe_nanmean(ts_p_in - ts_p_out),
        min_eff   = float(np.nanmin(ts_eff)),
        max_eff   = float(np.nanmax(ts_eff)),
    )


# ══════════════════════════════════════════════════════════════════════════
#  DAQ Manager — raw capture only
# ══════════════════════════════════════════════════════════════════════════
class MotorDAQManager:
    """Acquires Mod2 CH0-5 (voltage) and Mod3 CH0-5 (current) as raw
    float32 samples at hw_rate S/s and streams them to a .npy file.
    No signal processing is done in this class.
    """

    def __init__(self, chassis: str, ip: str, error_queue: queue.Queue):
        self.chassis = chassis or "cDAQ1"
        self.ip      = ip
        self.errors  = error_queue
        self.dev_v   = f"{self.chassis}Mod{MOD_V}"
        self.dev_i   = f"{self.chassis}Mod{MOD_I}"

        self.hw_rate  = 200_000
        self.ac_freq  = 400.0
        self.n_cycles = 3

        self.running  = False

        # raw capture state
        self.capturing    = False
        self.capture_path: Optional[str] = None
        self._cap_lock    = threading.Lock()
        self._cap_fh      = None   # open file handle during capture
        self._cap_count   = 0      # samples written so far

        # Live preview data — updated every block for the UI
        self.latest_block: Optional[np.ndarray] = None

        self._last_error_time:   dict = {}
        self._error_repeat_count: dict = {}

    def report_error(self, source: str, msg: str):
        key   = (source, msg)
        now   = time.monotonic()
        last  = self._last_error_time.get(key, 0.0)
        count = self._error_repeat_count.get(key, 0) + 1
        self._error_repeat_count[key] = count
        if now - last < 0.5:
            return
        self._last_error_time[key] = now
        sfx = f"  (x{count})" if count > 1 else ""
        self._error_repeat_count[key] = 0
        self.errors.put((datetime.now(), source, msg + sfx))

    def test_connection(self) -> tuple[bool, str]:
        if SIMULATION_MODE:
            return True, "Simulation mode — no hardware required."
        if self.ip:
            import socket
            for port in (80, 502):
                try:
                    s = socket.create_connection((self.ip, port), timeout=2.0)
                    s.close(); break
                except OSError as e:
                    if port == 502:
                        return False, f"Cannot reach {self.ip}: {e}"
        try:
            sys_ = nidaqmx.system.System.local()
            devs = [d.name for d in sys_.devices]
            miss = [d for d in (self.dev_v, self.dev_i) if d not in devs]
            if miss:
                return False, f"Devices not found: {miss}.  Available: {devs}"
            nidaqmx.system.Device(self.dev_v).self_test_device()
        except Exception as e:
            return False, f"NI-DAQmx check failed: {e}"
        return True, f"Connected to '{self.chassis}' ({self.ip})."

    def start(self):
        if self.running:
            return
        self.running = True
        threading.Thread(target=self._acq_loop, daemon=True).start()

    def stop(self):
        self.running = False
        self.capturing = False

    def start_capture(self, path: str):
        """Open a new .npy-format binary file and begin streaming raw data."""
        with self._cap_lock:
            if self._cap_fh:
                self._cap_fh.close()
            # Write a temporary header; we'll write the real shape on close.
            # Format: raw binary float32, we record shape separately.
            self._cap_fh     = open(path, 'wb')
            self._cap_count  = 0
            self.capture_path = path
            # Write a placeholder header (will be re-written on close)
            # We use a simple format: raw float32 rows, shape stored as
            # a companion .json sidecar file for simplicity.
            self.capturing = True

    def stop_capture(self) -> int:
        """Flush and close the capture file. Returns samples written."""
        with self._cap_lock:
            self.capturing = False
            if self._cap_fh:
                self._cap_fh.close()
                self._cap_fh = None
                # Save companion sidecar with shape and metadata
                if self.capture_path:
                    meta = {
                        "n_samples":  self._cap_count,
                        "n_columns":  N_RAW,
                        "dtype":      "float32",
                        "hw_rate_hz": self.hw_rate,
                        "ac_freq_hz": self.ac_freq,
                        "channels": {
                            "0-5":  "Voltage CH0-5 (raw sensor V, pre-cal)",
                            "6-11": "Current CH0-5 (raw sensor A, pre-cal)"
                        },
                        "column_names": V_NAMES + I_NAMES,
                        "captured_utc": datetime.utcnow().isoformat()
                    }
                    sidecar = self.capture_path.replace(".bin", "_meta.json")
                    with open(sidecar, "w") as f:
                        json.dump(meta, f, indent=2)
        return self._cap_count

    def _acq_loop(self):
        """Read raw samples from both modules in one synchronised task.
        Accumulate blocks in RAM; flush to disk when capturing."""
        block_sec = max(0.01, self.n_cycles / max(1.0, self.ac_freq))
        n_samp    = max(1, int(round(self.hw_rate * block_sec)))

        task = None
        if not SIMULATION_MODE:
            try:
                task = nidaqmx.Task()
                for ch in range(N_CH):
                    task.ai_channels.add_ai_voltage_chan(
                        f"{self.dev_v}/ai{ch}",
                        min_val=-10.0, max_val=10.0,
                        terminal_config=TerminalConfiguration.DIFF)
                for ch in range(N_CH):
                    task.ai_channels.add_ai_voltage_chan(
                        f"{self.dev_i}/ai{ch}",
                        min_val=-10.0, max_val=10.0,
                        terminal_config=TerminalConfiguration.DIFF)
                task.timing.cfg_samp_clk_timing(
                    rate=float(self.hw_rate),
                    sample_mode=AcquisitionType.CONTINUOUS,
                    samps_per_chan=self.hw_rate * 2)
                task.start()
            except Exception as e:
                self.report_error("Acq", f"Task start failed: {e}")
                self.running = False
                if task:
                    try: task.close()
                    except Exception: pass
                return

        try:
            while self.running:
                t0 = time.perf_counter()

                if SIMULATION_MODE:
                    t_arr = np.linspace(
                        time.perf_counter() - block_sec,
                        time.perf_counter(), n_samp, endpoint=False)
                    freq = self.ac_freq
                    v_sigs = np.column_stack([
                        5.43 * np.sin(2*np.pi*freq*t_arr + i*2*np.pi/3 + np.random.randn()*0.01)
                        for i in range(N_CH)])
                    i_sigs = np.column_stack([
                        0.50 * np.sin(2*np.pi*freq*t_arr + i*2*np.pi/3 + np.random.randn()*0.01)
                        for i in range(N_CH)])
                    block = np.hstack([v_sigs, i_sigs]).astype(np.float32)

                else:
                    try:
                        data = task.read(
                            number_of_samples_per_channel=nidaqmx.constants.READ_ALL_AVAILABLE,
                            timeout=2.0)
                        first = np.atleast_1d(data[0]) if data else np.array([])
                        if len(first) == 0:
                            time.sleep(max(0.0, block_sec - (time.perf_counter()-t0)))
                            continue
                        # data is list of 12 arrays (V0..V5, I0..I5)
                        block = np.column_stack(
                            [np.atleast_1d(ch) for ch in data]
                        ).astype(np.float32)
                    except Exception as e:
                        self.report_error("Acq", str(e))
                        try:
                            task.in_stream.relative_to = ReadRelativeTo.MOST_RECENT_SAMPLE
                            task.in_stream.offset = 0
                        except Exception: pass
                        time.sleep(0.2)
                        continue

                # Update live preview (latest block for UI display)
                self.latest_block = block

                # Write raw samples to file if capturing
                if self.capturing:
                    with self._cap_lock:
                        if self._cap_fh and self.capturing:
                            self._cap_fh.write(block.tobytes())
                            self._cap_count += block.shape[0]

                elapsed = time.perf_counter() - t0
                time.sleep(max(0.0, block_sec - elapsed))
        finally:
            if task:
                try: task.stop(); task.close()
                except Exception: pass


# ══════════════════════════════════════════════════════════════════════════
#  GUI
# ══════════════════════════════════════════════════════════════════════════
class MotorEffApp(tk.Tk):

    def __init__(self):
        super().__init__()
        self.title("Motor Controller Efficiency  —  cDAQ-9189  Mod2/Mod3")
        self.configure(bg=C_BG)
        self.geometry("1300x900")
        self.minsize(1100, 780)

        self.error_queue: queue.Queue = queue.Queue()
        self.daq: Optional[MotorDAQManager] = None
        self._connected_ok = False

        # Config defaults (overwritten from JSON)
        self._json_chassis     = "cDAQ9189-XXXXXXX"
        self._json_ip          = "169.254.32.5"
        self._json_auto_start  = False
        self._hw_rate          = 200_000
        self._ac_freq          = 400.0
        self._n_cycles         = 3
        self._rec_prefix       = "motor_raw"
        self._rec_auto_start   = False
        self._rec_timed        = False
        self._rec_duration_sec = 300.0
        self._rec_start_delay  = 2.0

        # Calibration loaded from JSON
        self._v_cal = [(1.0, 0.0)] * N_CH
        self._i_cal = [(1.0, 0.0)] * N_CH

        # Capture state
        self._cap_path: Optional[str] = None
        self._cap_start: Optional[float] = None

        self._load_from_json()
        self._build_style()
        self._build_ui()

        if SIMULATION_MODE:
            self._do_connect()

        elif self._json_auto_start:
            self.after(200, self._do_connect)

        self._poll()
        self._poll_errors()

    # ── JSON ──────────────────────────────────────────────────────────────
    def _load_from_json(self):
        if not os.path.exists(CAL_FILE):
            self.error_queue.put((datetime.now(), "JSON",
                                  f"Not found: {CAL_FILE}"))
            return
        try:
            with open(CAL_FILE) as f:
                data = json.load(f)

            self._json_chassis    = data.get("chassis_name",  self._json_chassis)
            self._json_ip         = data.get("ip_address",    self._json_ip)
            self._json_auto_start = bool(data.get("auto_start_on_connect", False))

            rec = data.get("recording") or {}
            self._rec_auto_start   = bool(rec.get("auto_record_on_start",    False))
            self._rec_prefix       = str(rec.get("log_filename_prefix",      "motor_raw"))
            self._rec_timed        = bool(rec.get("timed_recording",          False))
            self._rec_duration_sec = float(rec.get("record_duration_sec",     300))
            self._rec_start_delay  = float(rec.get("record_start_delay_sec",  2.0))

            mc  = data.get("module_config", {})
            aic = mc.get("modules_2_to_6_AI_9320", {})
            self._ac_freq  = float(aic.get("ac_frequency_hz",    400.0))
            self._n_cycles = int(aic.get("ac_cycles_per_block",    3))
            ht = aic.get("high_rate_task", {})
            self._hw_rate  = int(ht.get("hw_sample_rate_hz", 200_000))

            for rec_ch in data.get("NI_9320_modules_2_to_6", []):
                mod = rec_ch.get("module")
                ch  = rec_ch.get("channel")
                if ch is None or ch >= N_CH:
                    continue
                name  = rec_ch.get("name", "")
                scale = float(rec_ch.get("scale",  1.0))
                off   = float(rec_ch.get("offset", 0.0))
                if mod == MOD_V:
                    if name: V_NAMES[ch] = name
                    self._v_cal[ch] = (scale, off)
                elif mod == MOD_I:
                    if name: I_NAMES[ch] = name
                    self._i_cal[ch] = (scale, off)

        except Exception as e:
            self.error_queue.put((datetime.now(), "JSON", f"Load failed: {e}"))

    # ── Style ─────────────────────────────────────────────────────────────
    def _build_style(self):
        s = ttk.Style(self)
        s.theme_use("clam")
        s.configure(".", background=C_BG, foreground=C_TEXT,
                    fieldbackground=C_INPUT, troughcolor=C_BORDER)
        s.configure("TNotebook", background=C_BG, tabmargins=[2,4,2,0])
        s.configure("TNotebook.Tab", background=C_PANEL, foreground=C_MUTED,
                    padding=[12,5], font=("Segoe UI",10))
        s.map("TNotebook.Tab", background=[("selected",C_BG)],
              foreground=[("selected",C_ACCENT)])
        s.configure("TLabelframe", background=C_BG, foreground=C_ACCENT,
                    relief="flat", borderwidth=1)
        s.configure("TLabelframe.Label", background=C_BG,
                    foreground=C_ACCENT, font=("Segoe UI",10,"bold"))
        for nm, fg in [("G.TButton",C_GREEN),("R.TButton",C_RED),
                       ("A.TButton",C_ACCENT),("Y.TButton",C_YELLOW)]:
            s.configure(nm, background=C_PANEL, foreground=fg,
                        relief="flat", padding=[8,4])
            s.map(nm, background=[("active",C_BORDER)])
        s.configure("TEntry", fieldbackground=C_INPUT, foreground=C_TEXT)
        s.configure("TProgressbar", troughcolor=C_BORDER,
                    background=C_ACCENT)

    # ── UI ────────────────────────────────────────────────────────────────
    def _build_ui(self):
        # ── Top bar ──────────────────────────────────────────────────────
        top = tk.Frame(self, bg=C_PANEL, pady=6, padx=14)
        top.pack(fill="x")

        tk.Label(top, text="Motor Efficiency", font=("Segoe UI",14,"bold"),
                 bg=C_PANEL, fg=C_ACCENT).pack(side="left")
        mode_txt = "SIMULATION" if SIMULATION_MODE else "HARDWARE"
        tk.Label(top, text=f" [{mode_txt}]", font=("Segoe UI",9),
                 bg=C_PANEL,
                 fg=C_YELLOW if SIMULATION_MODE else C_GREEN
                 ).pack(side="left", padx=4)

        for lbl, attr, width in [("  Chassis:", "_json_chassis", 18),
                                   ("IP:", "_json_ip", 14)]:
            tk.Label(top, text=lbl, font=("Segoe UI",9),
                     bg=C_PANEL, fg=C_MUTED).pack(side="left")
            var = tk.StringVar(value=getattr(self, attr))
            setattr(self, f"_{attr.strip('_')}_var", var)
            tk.Entry(top, textvariable=var, width=width,
                     bg=C_INPUT, fg=C_TEXT, relief="flat",
                     font=("Courier New",9), insertbackground=C_TEXT
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

        self._cap_btn = ttk.Button(top, text="⏺ Start Capture",
                                   style="Y.TButton",
                                   command=self._toggle_capture)
        self._cap_btn.pack(side="right")

        self._status_lbl = tk.Label(top, text="● Disconnected",
                                    font=("Segoe UI",9),
                                    bg=C_PANEL, fg=C_RED)
        self._status_lbl.pack(side="left", padx=8)

        # ── Notebook ─────────────────────────────────────────────────────
        nb = ttk.Notebook(self)
        nb.pack(fill="both", expand=True, padx=6, pady=4)

        nb.add(self._build_live_tab(),    text="  Live Preview  ")
        nb.add(self._build_capture_tab(), text="  Capture  ")
        nb.add(self._build_postproc_tab(),text="  Post-Process  ")
        nb.add(self._build_cal_tab(),     text="  Calibration  ")

        # ── Error bar ────────────────────────────────────────────────────
        self._error_log: list = []
        bwrap = tk.Frame(self, bg="#1c1106")
        bwrap.pack(fill="x", side="bottom")

        self._err_log_frame = tk.Frame(bwrap, bg="#0d0701")
        self._err_log_text  = tk.Text(self._err_log_frame, height=7,
                                      bg="#0d0701", fg=C_RED,
                                      font=FONT_MONO_S, wrap="none",
                                      relief="flat", state="disabled")
        esb = ttk.Scrollbar(self._err_log_frame, orient="vertical",
                             command=self._err_log_text.yview)
        self._err_log_text.configure(yscrollcommand=esb.set)
        self._err_log_text.pack(side="left", fill="both",
                                expand=True, padx=(10,0), pady=4)
        esb.pack(side="right", fill="y", pady=4)
        self._err_log_visible = False

        bot = tk.Frame(bwrap, bg="#1c1106", height=28)
        bot.pack(fill="x", side="bottom")
        bot.pack_propagate(False)
        tk.Label(bot, text="Status:", font=FONT_TINY,
                 bg="#1c1106", fg=C_MUTED).pack(side="left", padx=(10,4))
        self._err_lbl = tk.Label(bot, text="No errors", font=FONT_TINY,
                                  bg="#1c1106", fg=C_MUTED, anchor="w")
        self._err_lbl.pack(side="left", fill="x", expand=True)
        self._err_toggle_btn = ttk.Button(bot, text="Show Log (0)",
                                           style="A.TButton",
                                           command=self._toggle_err_log)
        self._err_toggle_btn.pack(side="right", padx=4, pady=2)
        ttk.Button(bot, text="Clear", style="R.TButton",
                   command=self._clear_error).pack(side="right", padx=4, pady=2)

    # ── Live Preview tab ──────────────────────────────────────────────────
    def _build_live_tab(self):
        tab = tk.Frame(self._nb if hasattr(self,'_nb') else self, bg=C_BG)

        # Quick-look efficiency (computed from latest block)
        eff_f = tk.Frame(tab, bg=C_PANEL, padx=14, pady=10,
                         highlightbackground=C_BORDER, highlightthickness=1)
        eff_f.pack(fill="x", padx=10, pady=8)

        tk.Label(eff_f, text="LIVE EFFICIENCY  (latest block)",
                 font=FONT_BOLD, bg=C_PANEL, fg=C_MUTED).pack()

        self._live_eff_var = tk.StringVar(value="---")
        tk.Label(eff_f, textvariable=self._live_eff_var,
                 font=FONT_MONO_XL, bg=C_PANEL, fg=C_GREEN).pack()

        row_f = tk.Frame(eff_f, bg=C_PANEL)
        row_f.pack()
        self._live_vars: dict[str, tk.StringVar] = {}
        for key, label, fg in [
            ("pin",    "Pin (W)",    C_BLUE),
            ("pout",   "Pout (W)",   C_ACCENT),
            ("losses", "Losses (W)", C_ORANGE),
            ("pf_in",  "PF in",      C_TEXT),
            ("pf_out", "PF out",     C_TEXT),
        ]:
            f = tk.Frame(row_f, bg=C_PANEL)
            f.pack(side="left", padx=16)
            tk.Label(f, text=label, font=FONT_TINY,
                     bg=C_PANEL, fg=C_MUTED).pack()
            v = tk.StringVar(value="---")
            self._live_vars[key] = v
            tk.Label(f, textvariable=v, font=FONT_MONO_L,
                     bg=C_PANEL, fg=fg).pack()

        # Per-phase table
        ph_f = ttk.LabelFrame(tab,
            text=" Per-Phase  (Vrms · Irms · P · S · PF) ", padding=6)
        ph_f.pack(fill="x", padx=10, pady=4)

        hdrs = ["Phase","Signal","Vrms","Irms","P (W)","S (VA)","PF"]
        widths= [7,26,10,10,10,10,7]
        for c,(h,w) in enumerate(zip(hdrs,widths)):
            tk.Label(ph_f, text=h, font=FONT_BOLD,
                     bg=C_BG, fg=C_MUTED, width=w,
                     anchor="w").grid(row=0,column=c,padx=3,pady=2)

        ph_labels = ["A-IN","B-IN","C-IN","A-OUT","B-OUT","C-OUT"]
        sides_fg  = [C_BLUE]*3 + [C_ACCENT]*3
        self._ph_vars: list[dict] = []

        for i in range(N_CH):
            row_bg = C_PANEL if i%2==0 else C_BG
            for c,(h,w) in enumerate(zip(hdrs,widths)):
                if c == 0:
                    tk.Label(ph_f, text=ph_labels[i], font=FONT_BOLD,
                             bg=row_bg, fg=sides_fg[i], width=w,
                             anchor="w").grid(row=i+1,column=c,padx=3,pady=1,sticky="w")
                elif c == 1:
                    nm = tk.Label(ph_f, text=V_NAMES[i], font=FONT_TINY,
                                  bg=row_bg, fg=C_MUTED, width=w, anchor="w")
                    nm.grid(row=i+1,column=c,padx=3,pady=1,sticky="w")
                else:
                    pass  # filled by StringVar labels below

            d = {"name": None}
            col_map = {2:"vrms",3:"irms",4:"P",5:"S",6:"pf"}
            col_fg  = {2:C_ACCENT,3:C_BLUE,4:C_GREEN,5:C_TEXT,6:C_YELLOW}
            for c_idx,(c_key,c_fg) in zip(col_map.keys(),
                                           zip(col_map.values(),col_fg.values())):
                var = tk.StringVar(value="---")
                d[c_key] = var
                tk.Label(ph_f, textvariable=var, width=widths[c_idx],
                         font=FONT_MONO_S, bg=C_PANEL if i%2==0 else C_BG,
                         fg=c_fg, anchor="e").grid(row=i+1,column=c_idx,
                         padx=3, pady=1, sticky="e")
            self._ph_vars.append(d)

        return tab

    # ── Capture tab ───────────────────────────────────────────────────────
    def _build_capture_tab(self):
        tab = tk.Frame(self._nb if hasattr(self,'_nb') else self, bg=C_BG)

        info = ttk.LabelFrame(tab, text=" Raw Capture Settings ", padding=12)
        info.pack(fill="x", padx=10, pady=10)

        # Sample rate info
        self._rate_lbl = tk.Label(info,
            text=f"Rate: {self._hw_rate:,} S/s  ·  "
                 f"Channels: 12 (6V + 6I)  ·  "
                 f"Data rate: {self._hw_rate*12*4/1e6:.1f} MB/s",
            font=FONT_SMALL, bg=C_BG, fg=C_ACCENT)
        self._rate_lbl.pack(anchor="w", pady=4)

        # Duration selector
        dur_f = tk.Frame(info, bg=C_BG)
        dur_f.pack(anchor="w", pady=4)
        tk.Label(dur_f, text="Capture duration (0 = unlimited):",
                 font=FONT_SMALL, bg=C_BG, fg=C_MUTED).pack(side="left")
        self._dur_var = tk.StringVar(value=str(int(self._rec_duration_sec)))
        tk.Entry(dur_f, textvariable=self._dur_var, width=8,
                 bg=C_INPUT, fg=C_TEXT, font=FONT_MONO,
                 insertbackground=C_TEXT, relief="flat").pack(side="left", padx=6)
        tk.Label(dur_f, text="seconds", font=FONT_SMALL,
                 bg=C_BG, fg=C_MUTED).pack(side="left")

        # File prefix
        pfx_f = tk.Frame(info, bg=C_BG)
        pfx_f.pack(anchor="w", pady=4)
        tk.Label(pfx_f, text="File prefix:", font=FONT_SMALL,
                 bg=C_BG, fg=C_MUTED).pack(side="left")
        self._pfx_var = tk.StringVar(value=self._rec_prefix)
        tk.Entry(pfx_f, textvariable=self._pfx_var, width=24,
                 bg=C_INPUT, fg=C_TEXT, font=FONT_MONO,
                 insertbackground=C_TEXT, relief="flat").pack(side="left", padx=6)

        # Status / size indicator
        self._cap_status_var = tk.StringVar(value="Not capturing")
        tk.Label(info, textvariable=self._cap_status_var,
                 font=FONT_MONO, bg=C_BG, fg=C_YELLOW).pack(anchor="w", pady=4)

        # Progress bar (for timed capture)
        self._cap_prog = ttk.Progressbar(info, orient="horizontal",
                                          length=400, mode="determinate")
        self._cap_prog.pack(anchor="w", pady=4)

        # File size estimate
        sizes = ttk.LabelFrame(tab, text=" File Size Estimate ", padding=8)
        sizes.pack(fill="x", padx=10, pady=4)

        rate_mb = self._hw_rate * 12 * 4 / 1e6
        for dur, label in [(10,"10s"),(30,"30s"),(60,"1 min"),
                            (300,"5 min"),(600,"10 min")]:
            sz = rate_mb * dur
            unit = "MB" if sz < 1000 else "GB"
            val  = sz if sz < 1000 else sz/1000
            tk.Label(sizes, text=f"{label}: {val:.1f} {unit}",
                     font=FONT_SMALL, bg=C_BG, fg=C_MUTED
                     ).pack(side="left", padx=16)

        return tab

    # ── Post-process tab ──────────────────────────────────────────────────
    def _build_postproc_tab(self):
        tab = tk.Frame(self._nb if hasattr(self,'_nb') else self, bg=C_BG)

        ctrl = tk.Frame(tab, bg=C_BG)
        ctrl.pack(fill="x", padx=10, pady=8)

        ttk.Button(ctrl, text="📂 Load .bin File",
                   style="A.TButton",
                   command=self._load_bin_file).pack(side="left", padx=4)

        self._pp_freq_var = tk.StringVar(value=str(int(self._ac_freq)))
        tk.Label(ctrl, text="  Signal freq (Hz):",
                 font=FONT_SMALL, bg=C_BG, fg=C_MUTED).pack(side="left")
        tk.Entry(ctrl, textvariable=self._pp_freq_var, width=6,
                 bg=C_INPUT, fg=C_TEXT, font=FONT_MONO,
                 insertbackground=C_TEXT, relief="flat").pack(side="left", padx=4)

        ttk.Button(ctrl, text="⚙ Process",
                   style="G.TButton",
                   command=self._run_postproc).pack(side="left", padx=8)

        ttk.Button(ctrl, text="💾 Export CSV",
                   style="Y.TButton",
                   command=self._export_csv).pack(side="left", padx=4)

        self._pp_file_lbl = tk.Label(ctrl, text="No file loaded",
                                      font=FONT_SMALL, bg=C_BG, fg=C_MUTED)
        self._pp_file_lbl.pack(side="left", padx=14)

        self._pp_prog = ttk.Progressbar(tab, orient="horizontal",
                                         length=600, mode="determinate")
        self._pp_prog.pack(padx=10, pady=4, anchor="w")

        # Results summary
        sum_f = ttk.LabelFrame(tab, text=" Results Summary (average over file) ",
                                padding=10)
        sum_f.pack(fill="x", padx=10, pady=6)

        self._pp_vars: dict[str, tk.StringVar] = {}
        cells = [
            [("avg_efficiency",  "Avg Efficiency",  C_GREEN),
             ("min_eff",         "Min Efficiency",  C_RED),
             ("max_eff",         "Max Efficiency",  C_GREEN)],
            [("avg_p_in",        "Avg Pin (W)",     C_BLUE),
             ("avg_p_out",       "Avg Pout (W)",    C_ACCENT),
             ("avg_losses",      "Avg Losses (W)",  C_ORANGE)],
        ]
        for r, row in enumerate(cells):
            for c, (key, label, fg) in enumerate(row):
                f = tk.Frame(sum_f, bg=C_BG)
                f.grid(row=r, column=c, padx=20, pady=6, sticky="w")
                tk.Label(f, text=label, font=FONT_TINY,
                         bg=C_BG, fg=C_MUTED).pack(anchor="w")
                v = tk.StringVar(value="---")
                self._pp_vars[key] = v
                tk.Label(f, textvariable=v, font=FONT_MONO_L,
                         bg=C_BG, fg=fg).pack(anchor="w")

        # Per-channel averages
        ph2_f = ttk.LabelFrame(tab, text=" Per-Phase Averages ", padding=6)
        ph2_f.pack(fill="x", padx=10, pady=4)

        hdrs2 = ["Phase","Signal","Vrms","Irms","P (W)","S (VA)","PF"]
        w2    = [7,26,10,10,12,12,8]
        for c,(h,w) in enumerate(zip(hdrs2,w2)):
            tk.Label(ph2_f, text=h, font=FONT_BOLD, bg=C_BG, fg=C_MUTED,
                     width=w, anchor="w").grid(row=0,column=c,padx=3,pady=2)

        ph_labels2 = ["A-IN","B-IN","C-IN","A-OUT","B-OUT","C-OUT"]
        sides_fg2  = [C_BLUE]*3 + [C_ACCENT]*3
        self._pp_ph_vars: list[dict] = []

        for i in range(N_CH):
            bg = C_PANEL if i%2==0 else C_BG
            tk.Label(ph2_f, text=ph_labels2[i], font=FONT_BOLD,
                     bg=bg, fg=sides_fg2[i], width=w2[0],
                     anchor="w").grid(row=i+1,column=0,padx=3,pady=1,sticky="w")
            nm = tk.Label(ph2_f, text=V_NAMES[i], font=FONT_TINY,
                          bg=bg, fg=C_MUTED, width=w2[1], anchor="w")
            nm.grid(row=i+1,column=1,padx=3,pady=1,sticky="w")
            d = {"name_lbl": nm}
            for c_idx, (key, fg) in enumerate(
                    zip(["vrms","irms","P","S","pf"],
                        [C_ACCENT,C_BLUE,C_GREEN,C_TEXT,C_YELLOW]),
                    start=2):
                var = tk.StringVar(value="---")
                d[key] = var
                tk.Label(ph2_f, textvariable=var, width=w2[c_idx],
                         font=FONT_MONO_S, bg=bg, fg=fg,
                         anchor="e").grid(row=i+1,column=c_idx,
                         padx=3,pady=1,sticky="e")
            self._pp_ph_vars.append(d)

        self._pp_result: Optional[dict] = None
        self._pp_bin_path: Optional[str] = None
        return tab

    # ── Calibration tab ───────────────────────────────────────────────────
    def _build_cal_tab(self):
        tab = tk.Frame(self._nb if hasattr(self,'_nb') else self, bg=C_BG)

        hdr = tk.Frame(tab, bg=C_BG)
        hdr.pack(fill="x", padx=10, pady=4)
        tk.Label(hdr, text="Scale & Offset  —  applied during post-processing",
                 font=FONT_HEAD, bg=C_BG, fg=C_ACCENT).pack(side="left")
        ttk.Button(hdr, text="Apply", style="G.TButton",
                   command=self._apply_cal).pack(side="right")

        body = tk.Frame(tab, bg=C_BG)
        body.pack(fill="both", expand=True, padx=10, pady=4)

        self._vcal_s: list[tk.StringVar] = []
        self._vcal_o: list[tk.StringVar] = []
        self._ical_s: list[tk.StringVar] = []
        self._ical_o: list[tk.StringVar] = []

        for col,(side,names,sl,ol,cal) in enumerate([
            ("Voltage (Mod2)", V_NAMES, self._vcal_s, self._vcal_o, self._v_cal),
            ("Current (Mod3)", I_NAMES, self._ical_s, self._ical_o, self._i_cal),
        ]):
            sec = ttk.LabelFrame(body, text=f" {side} ", padding=8)
            sec.grid(row=0, column=col, padx=6, pady=4, sticky="nsew")
            body.columnconfigure(col, weight=1)

            for c,(h,w) in enumerate(zip(["Ch","Signal Name","Scale","Offset"],
                                          [4,26,12,12])):
                tk.Label(sec, text=h, font=FONT_BOLD, bg=C_BG,
                         fg=C_MUTED, width=w, anchor="w"
                         ).grid(row=0,column=c,padx=4,sticky="w")

            for ch in range(N_CH):
                s_v = tk.StringVar(value=str(cal[ch][0]))
                o_v = tk.StringVar(value=str(cal[ch][1]))
                sl.append(s_v); ol.append(o_v)
                tk.Label(sec, text=str(ch), font=FONT_MONO_S, bg=C_BG,
                         fg=C_TEXT, width=4, anchor="w"
                         ).grid(row=ch+1,column=0,padx=4,pady=1,sticky="w")
                tk.Label(sec, text=names[ch], font=FONT_TINY, bg=C_BG,
                         fg=C_MUTED, width=26, anchor="w"
                         ).grid(row=ch+1,column=1,padx=4,pady=1,sticky="w")
                tk.Entry(sec, textvariable=s_v, width=12, bg=C_INPUT,
                         fg=C_TEXT, font=FONT_MONO_S,
                         insertbackground=C_TEXT, relief="flat"
                         ).grid(row=ch+1,column=2,padx=4,pady=1)
                tk.Entry(sec, textvariable=o_v, width=12, bg=C_INPUT,
                         fg=C_TEXT, font=FONT_MONO_S,
                         insertbackground=C_TEXT, relief="flat"
                         ).grid(row=ch+1,column=3,padx=4,pady=1)
        return tab

    # ── Connection ────────────────────────────────────────────────────────
    def _connect(self):
        chassis = self._json_chassis_var.get().strip()
        ip      = self._json_ip_var.get().strip()
        self.daq = MotorDAQManager(chassis, ip, self.error_queue)

        if SIMULATION_MODE:
            self._do_connect()
            return

        self._conn_btn.config(state="disabled", text="Testing...")
        self._status_lbl.config(text="Testing...", fg=C_YELLOW)

        def worker():
            ok, msg = self.daq.test_connection()
            self.after(0, lambda: self._on_connect_result(ok, msg))
        threading.Thread(target=worker, daemon=True).start()

    def _do_connect(self):
        """Complete connection without a test (simulation or auto-start)."""
        chassis = self._json_chassis_var.get().strip()
        ip      = self._json_ip_var.get().strip()
        if not self.daq:
            self.daq = MotorDAQManager(chassis, ip, self.error_queue)
        self._connected_ok = True
        self._conn_btn.config(text="Reconnect")
        mode = "SIMULATION" if SIMULATION_MODE else "HARDWARE"
        self._status_lbl.config(text=f"● Connected ({mode})", fg=C_GREEN)

    def _on_connect_result(self, ok: bool, msg: str):
        self._conn_btn.config(state="normal", text="Reconnect")
        self._connected_ok = ok
        if ok:
            self._status_lbl.config(text=f"● {msg}", fg=C_GREEN)
            if self._json_auto_start:
                self.after(500, self._start_acq)
        else:
            self._status_lbl.config(text="● Connection failed", fg=C_RED)
            self.error_queue.put((datetime.now(), "Connection", msg))
            messagebox.showerror("Connection Failed", msg)

    def _ensure_connected(self) -> bool:
        if not self.daq or (not SIMULATION_MODE and not self._connected_ok):
            messagebox.showwarning("Not Connected",
                                   "Connect to the chassis first.")
            return False
        return True

    # ── Acquisition ───────────────────────────────────────────────────────
    def _start_acq(self):
        if not self._ensure_connected():
            return
        self.daq.hw_rate  = self._hw_rate
        self.daq.ac_freq  = self._ac_freq
        self.daq.n_cycles = self._n_cycles
        self.daq.start()
        self._status_lbl.config(
            text=f"● Running  {self._hw_rate:,} S/s",
            fg=C_GREEN)
        if self._rec_auto_start:
            delay = max(100, int(self._rec_start_delay * 1000))
            self.after(delay, self._start_capture_auto)

    def _stop_acq(self):
        if self.daq:
            if self.daq.capturing:
                self._do_stop_capture()
            self.daq.stop()
        self._status_lbl.config(text="● Stopped", fg=C_MUTED)

    # ── Capture ───────────────────────────────────────────────────────────
    def _make_bin_path(self) -> str:
        import sys
        try:
            d = os.path.dirname(os.path.abspath(sys.argv[0]))
        except Exception:
            d = os.getcwd()
        ts   = datetime.now().strftime("%Y%m%d_%H%M%S")
        name = f"{self._pfx_var.get().strip() or 'motor_raw'}_{ts}.bin"
        return os.path.join(d, name)

    def _toggle_capture(self):
        if not self._ensure_connected():
            return
        if not self.daq or not self.daq.running:
            messagebox.showwarning("Not Running",
                                   "Start acquisition first.")
            return
        if self.daq.capturing:
            self._do_stop_capture()
        else:
            import sys
            try:
                d = os.path.dirname(os.path.abspath(sys.argv[0]))
            except Exception:
                d = os.getcwd()
            path = filedialog.asksaveasfilename(
                defaultextension=".bin",
                filetypes=[("Raw binary", "*.bin"), ("All", "*.*")],
                title="Save Raw Capture",
                initialdir=d,
                initialfile=os.path.basename(self._make_bin_path()),
            )
            if not path:
                return
            self._do_start_capture(path)

    def _do_start_capture(self, path: str):
        self._cap_path  = path
        self._cap_start = time.perf_counter()
        self.daq.start_capture(path)
        self._cap_btn.config(text="⏹ Stop Capture")
        self._cap_status_var.set(
            f"Capturing → {os.path.basename(path)}")
        self.error_queue.put((datetime.now(), "Capture",
                              f"Started: {os.path.basename(path)}"))

    def _do_stop_capture(self):
        n = self.daq.stop_capture()
        secs = n / max(1, self._hw_rate)
        self._cap_btn.config(text="⏺ Start Capture")
        self._cap_status_var.set(
            f"Saved {n:,} samples ({secs:.1f}s)  →  "
            f"{os.path.basename(self._cap_path or '')}")
        self._cap_prog["value"] = 100
        self.error_queue.put((datetime.now(), "Capture",
                              f"Saved {n:,} samples ({secs:.1f}s): "
                              f"{os.path.basename(self._cap_path or '')}"))

    def _start_capture_auto(self):
        if self.daq and self.daq.running:
            self._do_start_capture(self._make_bin_path())

    # ── Post-processing ───────────────────────────────────────────────────
    def _apply_cal(self):
        errors = []
        for ch in range(N_CH):
            try:
                self._v_cal[ch] = (float(self._vcal_s[ch].get()),
                                   float(self._vcal_o[ch].get()))
            except ValueError:
                errors.append(f"V CH{ch}")
            try:
                self._i_cal[ch] = (float(self._ical_s[ch].get()),
                                   float(self._ical_o[ch].get()))
            except ValueError:
                errors.append(f"I CH{ch}")
        if errors:
            messagebox.showwarning("Invalid", "\n".join(errors))
        else:
            messagebox.showinfo("Calibration", "Applied.")

    def _load_bin_file(self):
        path = filedialog.askopenfilename(
            filetypes=[("Raw binary", "*.bin"), ("All", "*.*")],
            title="Load Raw Capture File")
        if not path:
            return
        # Try to load sidecar metadata
        meta_path = path.replace(".bin", "_meta.json")
        if os.path.exists(meta_path):
            with open(meta_path) as f:
                meta = json.load(f)
            self._pp_freq_var.set(str(meta.get("ac_freq_hz", self._ac_freq)))
            self._hw_rate = int(meta.get("hw_rate_hz", self._hw_rate))
        self._pp_bin_path = path
        n_bytes = os.path.getsize(path)
        n_samps = n_bytes // (N_RAW * 4)
        secs    = n_samps / self._hw_rate
        self._pp_file_lbl.config(
            text=f"{os.path.basename(path)}  "
                 f"({n_samps:,} samples  ·  {secs:.1f}s  ·  "
                 f"{n_bytes/1e6:.1f} MB)",
            fg=C_ACCENT)
        self._pp_result = None
        self._pp_prog["value"] = 0

    def _run_postproc(self):
        if not self._pp_bin_path or not os.path.exists(self._pp_bin_path):
            messagebox.showwarning("No file", "Load a .bin file first.")
            return
        try:
            ac_freq = float(self._pp_freq_var.get())
        except ValueError:
            messagebox.showerror("Error", "Invalid frequency.")
            return

        # Run in background thread so UI stays responsive
        self._pp_prog["value"] = 0
        self._pp_file_lbl.config(text="Processing...", fg=C_YELLOW)

        def worker():
            try:
                # Load raw binary
                raw = np.fromfile(self._pp_bin_path, dtype=np.float32)
                raw = raw.reshape(-1, N_RAW)
                result = process_file(self._pp_bin_path,
                                      self._v_cal, self._i_cal,
                                      ac_freq, self._hw_rate,
                                      progress_cb=lambda p: self.after(
                                          0, lambda: self._pp_prog.__setitem__(
                                              "value", p)))
                self.after(0, lambda: self._show_pp_result(result))
            except Exception as e:
                self.after(0, lambda: messagebox.showerror(
                    "Processing Error", str(e)))

        threading.Thread(target=worker, daemon=True).start()

    def _show_pp_result(self, res: dict):
        self._pp_result = res
        if not res:
            self._pp_file_lbl.config(text="No result", fg=C_RED)
            return

        def fmt(v, sfx=""):
            try:
                return f"{v:.3f}{sfx}" if np.isfinite(v) else "---"
            except Exception:
                return "---"

        self._pp_vars["avg_efficiency"].set(
            fmt(res.get("avg_eff", float("nan")), "%"))
        self._pp_vars["min_eff"].set(
            fmt(res.get("min_eff", float("nan")), "%"))
        self._pp_vars["max_eff"].set(
            fmt(res.get("max_eff", float("nan")), "%"))
        self._pp_vars["avg_p_in"].set(fmt(res.get("avg_p_in", 0)))
        self._pp_vars["avg_p_out"].set(fmt(res.get("avg_p_out", 0)))
        self._pp_vars["avg_losses"].set(fmt(res.get("avg_losses", 0)))

        avg_v = res.get("avg_vrms", np.zeros(N_CH))
        avg_i = res.get("avg_irms", np.zeros(N_CH))
        avg_P = res.get("avg_P",    np.zeros(N_CH))
        avg_S = res.get("avg_S",    np.zeros(N_CH))
        avg_pf= res.get("avg_pf",   np.zeros(N_CH))

        for i in range(N_CH):
            d = self._pp_ph_vars[i]
            d["name_lbl"].config(text=V_NAMES[i])
            d["vrms"].set(fmt(avg_v[i]))
            d["irms"].set(fmt(avg_i[i]))
            d["P"].set(fmt(avg_P[i]))
            d["S"].set(fmt(avg_S[i]))
            d["pf"].set(fmt(avg_pf[i]))

        n  = res.get("n_chunks", 0)
        fs = res.get("fs", self._hw_rate)
        self._pp_file_lbl.config(
            text=f"Done — {n} second(s) processed  ·  "
                 f"{n*fs:,} total samples",
            fg=C_GREEN)
        self._pp_prog["value"] = 100

    def _export_csv(self):
        if not self._pp_result:
            messagebox.showwarning("No results",
                                   "Run post-processing first.")
            return
        import sys
        try:
            d = os.path.dirname(os.path.abspath(sys.argv[0]))
        except Exception:
            d = os.getcwd()
        ts   = datetime.now().strftime("%Y%m%d_%H%M%S")
        path = filedialog.asksaveasfilename(
            defaultextension=".csv",
            filetypes=[("CSV", "*.csv"), ("All", "*.*")],
            title="Export Results CSV",
            initialdir=d,
            initialfile=f"motor_results_{ts}.csv")
        if not path:
            return

        res  = self._pp_result
        rows = []
        n    = res["n_chunks"]

        # Header
        hdr = (["Second"]
               + [f"Vrms_CH{i}_{V_NAMES[i]}" for i in range(N_CH)]
               + [f"Irms_CH{i}_{I_NAMES[i]}"  for i in range(N_CH)]
               + [f"P_CH{i}_W"    for i in range(N_CH)]
               + [f"S_CH{i}_VA"   for i in range(N_CH)]
               + [f"PF_CH{i}"     for i in range(N_CH)]
               + ["Pin_W","Pout_W","Losses_W","Efficiency_pct"])
        rows.append(hdr)

        for s in range(n):
            def g(arr, i): return f"{float(arr[s, i]):.6f}" if arr.ndim==2 else f"{float(arr[s]):.6f}"
            eff = res["ts_eff"][s]
            row = ([str(s+1)]
                   + [g(res["ts_vrms"], i) for i in range(N_CH)]
                   + [g(res["ts_irms"], i) for i in range(N_CH)]
                   + [g(res["ts_P"],    i) for i in range(N_CH)]
                   + [g(res["ts_S"],    i) for i in range(N_CH)]
                   + [g(res["ts_pf"],   i) for i in range(N_CH)]
                   + [f"{float(res['ts_p_in'][s]):.4f}",
                      f"{float(res['ts_p_out'][s]):.4f}",
                      f"{float(res['ts_p_in'][s]-res['ts_p_out'][s]):.4f}",
                      f"{float(eff):.4f}" if np.isfinite(eff) else ""])
            rows.append(row)

        import csv
        with open(path, "w", newline="") as f:
            csv.writer(f).writerows(rows)
        messagebox.showinfo("Exported",
                            f"Results saved to:\n{os.path.basename(path)}")

    # ── Error log ─────────────────────────────────────────────────────────
    _MAX_ERROR_LOG = 500

    def _toggle_err_log(self):
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
        if self.daq and self.daq.latest_block is not None:
            blk = self.daq.latest_block
            try:
                res = process_block(blk, self._v_cal, self._i_cal,
                                    self._ac_freq, self._hw_rate)
                eff = res.get("efficiency")

                if eff is not None:
                    fg = C_GREEN if eff >= 90 else (C_YELLOW if eff >= 70
                                                    else C_RED)
                    self._live_eff_var.set(f"{eff:.2f}%")
                else:
                    self._live_eff_var.set("---")

                def f2(v, u=""): return f"{v:.2f}{u}" if np.isfinite(v) else "---"

                self._live_vars["pin"].set(f2(res["p_in"]))
                self._live_vars["pout"].set(f2(res["p_out"]))
                self._live_vars["losses"].set(f2(res["losses"]))
                self._live_vars["pf_in"].set(f2(res["pf_in"]))
                self._live_vars["pf_out"].set(f2(res["pf_out"]))

                for i in range(N_CH):
                    d = self._ph_vars[i]
                    d["vrms"].set(f"{res['vrms'][i]:.3f}")
                    d["irms"].set(f"{res['irms'][i]:.4f}")
                    d["P"].set(f"{res['P'][i]:.2f}")
                    d["S"].set(f"{res['S'][i]:.2f}")
                    d["pf"].set(f"{res['pf'][i]:.4f}")
            except Exception:
                pass

        # Capture countdown / elapsed
        if self.daq and self.daq.capturing and self._cap_start is not None:
            elapsed = time.perf_counter() - self._cap_start
            n       = self.daq._cap_count
            mb      = n * N_RAW * 4 / 1e6
            m, s    = divmod(int(elapsed), 60)
            try:
                dur = float(self._dur_var.get())
            except Exception:
                dur = 0

            if dur > 0:
                pct = min(100, elapsed / dur * 100)
                self._cap_prog["value"] = pct
                rem = max(0, dur - elapsed)
                self._cap_status_var.set(
                    f"⏺ Recording  {m:02d}:{s:02d}  ·  "
                    f"{mb:.1f} MB  ·  {n:,} samples  ·  "
                    f"{rem:.0f}s remaining")
                if elapsed >= dur:
                    self._do_stop_capture()
            else:
                self._cap_prog["value"] = 0
                self._cap_status_var.set(
                    f"⏺ Recording  {m:02d}:{s:02d}  ·  "
                    f"{mb:.1f} MB  ·  {n:,} samples")

        self.after(200, self._poll)   # 5 Hz — post-proc is the bottleneck

    def _poll_errors(self):
        new = []
        try:
            while True: new.append(self.error_queue.get_nowait())
        except queue.Empty: pass
        if new:
            self._error_log.extend(new)
            if len(self._error_log) > self._MAX_ERROR_LOG:
                self._error_log = self._error_log[-self._MAX_ERROR_LOG:]
            ts, src, msg = new[-1]
            self._err_lbl.config(
                text=f"[{ts.strftime('%H:%M:%S')}] {src}: {msg}"
                + (f"  (+{len(new)-1})" if len(new)>1 else ""),
                fg=C_RED)
            lines = [f"[{t.strftime('%H:%M:%S')}] {s}: {m}"
                     for t,s,m in new[-50:]]
            self._err_log_text.config(state="normal")
            self._err_log_text.insert("end", "\n".join(lines)+"\n")
            self._err_log_text.see("end")
            self._err_log_text.config(state="disabled")
            self._err_toggle_btn.config(
                text=("Hide" if self._err_log_visible else "Show Log")
                + f" ({len(self._error_log)})")
        self.after(300, self._poll_errors)

    # ── Close ─────────────────────────────────────────────────────────────
    def destroy(self):
        if self.daq:
            if self.daq.capturing:
                self.daq.stop_capture()
            self.daq.stop()
        super().destroy()


# ══════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    app = MotorEffApp()
    app.mainloop()

