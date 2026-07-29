"""
Three-Phase Motor Controller Analyser
======================================
Combined acquisition, waveform viewer, and efficiency post-processor.
Reads the same cdaq_calibration.json used by the main cDAQ script.

Tabs
----
  Channel Config   — choose which channels on Mod2/Mod3 to acquire,
                     assign each to a role (V-in / V-out / I-in / I-out)
  Live Preview     — real-time Vrms/Irms/PF/efficiency from the DAQ
  Capture          — record raw float32 waveforms to a .bin file
  Waveform Viewer  — inspect any .bin file before or after capture;
                     auto-loads the file when a capture finishes
  Post-Process     — compute Vrms/Irms/P/S/PF/efficiency over the file
  Calibration      — scale & offset per channel (applied in post-processing)

Hardware
--------
  Module 2  — NI 9320  Voltage inputs   (configurable channels 0-15)
  Module 3  — NI 9320  Current inputs   (configurable channels 0-15)

File format
-----------
  Raw float32 binary, shape (N_samples, N_active_channels * 2)
    First  N_active_ch columns : voltage channels in selected order
    Second N_active_ch columns : current channels in selected order
  Companion _meta.json sidecar stores channel mapping and sample rate.

Dependencies: pip install nidaqmx numpy matplotlib
"""

import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import threading
import time
import queue
import json
import csv
import os
import sys
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

try:
    import matplotlib
    matplotlib.use("TkAgg")
    from matplotlib.figure import Figure
    from matplotlib.backends.backend_tkagg import (
        FigureCanvasTkAgg, NavigationToolbar2Tk)
    from matplotlib.ticker import FuncFormatter
    HAS_MPL = True
except ImportError:
    HAS_MPL = False
    print("[WARNING] matplotlib not installed — waveform viewer disabled")


# ══════════════════════════════════════════════════════════════════════════
#  Module / channel constants
# ══════════════════════════════════════════════════════════════════════════
MOD_V       = 2          # NI 9320 — voltage module
MOD_I       = 3          # NI 9320 — current module
MAX_CH      = 16         # channels per 9320 module

# Role constants
ROLE_V_IN   = "V-In"
ROLE_V_OUT  = "V-Out"
ROLE_I_IN   = "I-In"
ROLE_I_OUT  = "I-Out"
ROLE_UNUSED = "Unused"
ROLES       = [ROLE_V_IN, ROLE_V_OUT, ROLE_I_IN, ROLE_I_OUT, ROLE_UNUSED]

# ── Colours ───────────────────────────────────────────────────────────────
C_BG    = "#0d1117"
C_PANEL = "#161b22"
C_BORD  = "#30363d"
C_ACC   = "#00b4d8"
C_GREEN = "#39d353"
C_RED   = "#f85149"
C_YEL   = "#e3b341"
C_ORG   = "#f0883e"
C_BLUE  = "#58a6ff"
C_PURP  = "#d2a8ff"
C_TEXT  = "#e6edf3"
C_MUTED = "#8b949e"
C_INPUT = "#21262d"

# Colour per role for waveform plot
ROLE_COLORS = {
    ROLE_V_IN:   "#58a6ff",
    ROLE_V_OUT:  "#79c0ff",
    ROLE_I_IN:   "#f85149",
    ROLE_I_OUT:  "#ffa657",
    ROLE_UNUSED: "#8b949e",
}

FONT_MONO   = ("Courier New", 9)
FONT_MONOS  = ("Courier New", 8)
FONT_MONOL  = ("Courier New", 16, "bold")
FONT_MONOXL = ("Courier New", 36, "bold")
FONT_SMALL  = ("Segoe UI", 9)
FONT_TINY   = ("Segoe UI", 8)
FONT_BOLD   = ("Segoe UI", 10, "bold")
FONT_HEAD   = ("Segoe UI", 12, "bold")
FONT_TITLE  = ("Segoe UI", 14, "bold")


# ══════════════════════════════════════════════════════════════════════════
#  JSON calibration file
# ══════════════════════════════════════════════════════════════════════════
def _find_cal_file() -> str:
    candidates = []
    try:
        d = os.path.dirname(os.path.abspath(sys.argv[0]))
        if os.path.isdir(d):
            candidates.append(os.path.join(d, "cdaq_calibration.json"))
    except Exception: pass
    try:
        d = os.path.dirname(os.path.abspath(__file__))
        p = os.path.join(d, "cdaq_calibration.json")
        if p not in candidates: candidates.append(p)
    except NameError: pass
    cwd = os.path.join(os.getcwd(), "cdaq_calibration.json")
    if cwd not in candidates: candidates.append(cwd)
    print("[Motor Analyser] Searching for cdaq_calibration.json:")
    for p in candidates:
        ex = os.path.exists(p)
        print(f"  {'FOUND' if ex else 'not found':10s}  {p}")
        if ex: return p
    return candidates[0] if candidates else "cdaq_calibration.json"

CAL_FILE = _find_cal_file()


# ══════════════════════════════════════════════════════════════════════════
#  Channel configuration model
# ══════════════════════════════════════════════════════════════════════════
class ChannelConfig:
    """
    Holds the per-channel assignment for Mod2 (voltage) and Mod3 (current).
    Each of the 16 channels on each module can be assigned a role:
      V-In, V-Out, I-In, I-Out, or Unused.

    Active channels are those whose role is not Unused.
    The .bin file columns are ordered: V-channels first, then I-channels,
    sorted by channel number within each group.
    """

    def __init__(self):
        # Default: CH0-2 V-In, CH3-5 V-Out on Mod2;
        #          CH0-2 I-In, CH3-5 I-Out on Mod3;
        #          all others Unused
        self.v_roles = [ROLE_UNUSED] * MAX_CH
        self.i_roles = [ROLE_UNUSED] * MAX_CH
        self.v_names = [f"V_CH{c:02d}" for c in range(MAX_CH)]
        self.i_names = [f"I_CH{c:02d}" for c in range(MAX_CH)]
        self.v_cal   = [(1.0, 0.0)] * MAX_CH
        self.i_cal   = [(1.0, 0.0)] * MAX_CH

        for c in range(3):
            self.v_roles[c]   = ROLE_V_IN
            self.v_roles[c+3] = ROLE_V_OUT
            self.i_roles[c]   = ROLE_I_IN
            self.i_roles[c+3] = ROLE_I_OUT

    # ── Derived lists ────────────────────────────────────────────────────
    @property
    def v_active(self) -> list[int]:
        return [c for c in range(MAX_CH) if self.v_roles[c] != ROLE_UNUSED]

    @property
    def i_active(self) -> list[int]:
        return [c for c in range(MAX_CH) if self.i_roles[c] != ROLE_UNUSED]

    @property
    def n_v(self) -> int: return len(self.v_active)

    @property
    def n_i(self) -> int: return len(self.i_active)

    @property
    def n_cols(self) -> int: return self.n_v + self.n_i

    def col_index(self, side: str, ch: int) -> Optional[int]:
        """Return column index in .bin file for (side='V'|'I', ch=0..15)."""
        if side == 'V':
            act = self.v_active
            return act.index(ch) if ch in act else None
        else:
            act = self.i_active
            return self.n_v + act.index(ch) if ch in act else None

    def all_names(self) -> list[str]:
        return ([self.v_names[c] for c in self.v_active]
                + [self.i_names[c] for c in self.i_active])

    def all_roles(self) -> list[str]:
        return ([self.v_roles[c] for c in self.v_active]
                + [self.i_roles[c] for c in self.i_active])

    def all_cals(self) -> list[tuple]:
        return ([self.v_cal[c] for c in self.v_active]
                + [self.i_cal[c] for c in self.i_active])

    def paired_phases(self) -> list[tuple]:
        """Return list of (v_col, i_col, role_v, role_i) for matched pairs."""
        pairs = []
        for role_v, role_i, side_label in [
            (ROLE_V_IN,  ROLE_I_IN,  "Input"),
            (ROLE_V_OUT, ROLE_I_OUT, "Output"),
        ]:
            vch = [c for c in self.v_active if self.v_roles[c] == role_v]
            ich = [c for c in self.i_active if self.i_roles[c] == role_i]
            for v, i in zip(vch, ich):
                vc_idx = self.col_index('V', v)
                ic_idx = self.col_index('I', i)
                pairs.append((vc_idx, ic_idx, side_label, v, i))
        return pairs
        return pairs

    def to_meta(self, hw_rate: float, ac_freq: float) -> dict:
        return {
            "n_samples":    0,
            "n_columns":    self.n_cols,
            "dtype":        "float32",
            "hw_rate_hz":   hw_rate,
            "ac_freq_hz":   ac_freq,
            "v_active_chs": self.v_active,
            "i_active_chs": self.i_active,
            "v_names":      [self.v_names[c] for c in self.v_active],
            "i_names":      [self.i_names[c] for c in self.i_active],
            "v_roles":      [self.v_roles[c] for c in self.v_active],
            "i_roles":      [self.i_roles[c] for c in self.i_active],
            "column_names": self.all_names(),
            "captured_utc": datetime.utcnow().isoformat(),
        }


# ══════════════════════════════════════════════════════════════════════════
#  Signal processing
# ══════════════════════════════════════════════════════════════════════════
def channel_stats(samples: np.ndarray, fs: float) -> dict:
    if len(samples) < 2:
        return {}
    rms   = float(np.sqrt(np.nanmean(samples**2)))
    peak  = float(np.nanmax(np.abs(samples)))
    mean  = float(np.nanmean(samples))
    crest = peak / rms if rms > 1e-9 else float("nan")
    zc    = np.where(np.diff(np.sign(samples - mean)))[0]
    freq  = (len(zc)/2)/(len(samples)/fs) if len(zc)>2 else float("nan")
    return dict(rms=rms, peak=peak, mean=mean, crest=crest, freq=freq)


def process_window(cal_win: np.ndarray, cfg: "ChannelConfig",
                   ac_freq: float, fs: float) -> dict:
    """
    Compute power/PF/efficiency from a calibrated window array.
    cal_win: (N, n_cols) — already calibrated.
    Returns per-pair and totals dict.
    """
    if cal_win.shape[0] < 2:
        return {}

    # Align to complete cycles
    spc   = fs / max(1.0, ac_freq)
    n_cyc = max(1, int(cal_win.shape[0] // spc))
    n_use = int(round(n_cyc * spc))
    w     = cal_win[:n_use]

    pairs = cfg.paired_phases()
    if not pairs:
        return {}

    results = []
    p_in = p_out = s_in = s_out = 0.0

    for vc_idx, ic_idx, side, vch, ich in pairs:
        if vc_idx is None or ic_idx is None: continue
        if vc_idx >= w.shape[1] or ic_idx >= w.shape[1]: continue
        v = w[:, vc_idx]; i = w[:, ic_idx]
        vrms = float(np.sqrt(np.nanmean(v**2)))
        irms = float(np.sqrt(np.nanmean(i**2)))
        P    = float(np.nanmean(v * i))
        S    = vrms * irms
        Q    = float(np.sqrt(max(0.0, S**2 - P**2)))
        pf   = P / S if S > 1e-6 else 0.0
        results.append(dict(side=side, vch=vch, ich=ich,
                            vrms=vrms, irms=irms, P=P, S=S, Q=Q, pf=pf))
        if side == "Input":
            p_in += P; s_in += S
        else:
            p_out += P; s_out += S

    eff    = (p_out / p_in * 100) if p_in > 1e-3 else None
    losses = p_in - p_out
    return dict(pairs=results, p_in=p_in, p_out=p_out,
                s_in=s_in, s_out=s_out,
                pf_in=p_in/s_in  if s_in  > 1e-6 else 0.0,
                pf_out=p_out/s_out if s_out > 1e-6 else 0.0,
                efficiency=eff, losses=losses, n_cycles=n_cyc)


def process_file_chunked(path: str, cfg: "ChannelConfig",
                         ac_freq: float, fs: float,
                         progress_cb=None) -> dict:
    """Process a .bin file in 1-second chunks and return time-series."""
    raw   = np.fromfile(path, dtype=np.float32)
    n_col = cfg.n_cols
    if raw.size % n_col != 0:
        raise ValueError(f"File size {raw.size} not divisible by {n_col} columns")
    raw   = raw.reshape(-1, n_col)
    N     = raw.shape[0]
    chunk = int(fs)

    # Apply calibration once per chunk
    cals = cfg.all_cals()

    ts_p_in=[]; ts_p_out=[]; ts_eff=[]; ts_pf_in=[]; ts_pf_out=[]
    ts_vrms=[]; ts_irms=[]; ts_P=[]; ts_S=[]; ts_pf=[]
    n_chunks = max(1, N // chunk)

    for idx in range(n_chunks):
        s = idx * chunk
        e = min(s + chunk, N)
        blk = raw[s:e].astype(np.float64)
        # apply calibration
        for col, (sc, off) in enumerate(cals):
            blk[:, col] = blk[:, col] * sc + off
        res = process_window(blk, cfg, ac_freq, fs)
        if not res:
            break
        ts_p_in.append(res["p_in"]); ts_p_out.append(res["p_out"])
        ts_pf_in.append(res["pf_in"]); ts_pf_out.append(res["pf_out"])
        eff = res["efficiency"]
        ts_eff.append(eff if eff is not None else float("nan"))
        if progress_cb:
            progress_cb(int((idx+1)/n_chunks*100))

    if not ts_p_in:
        return {}

    ts_p_in  = np.array(ts_p_in)
    ts_p_out = np.array(ts_p_out)
    ts_eff   = np.array(ts_eff)

    return dict(
        n_chunks=n_chunks, fs=fs, ac_freq=ac_freq,
        ts_p_in=ts_p_in, ts_p_out=ts_p_out,
        ts_pf_in=np.array(ts_pf_in), ts_pf_out=np.array(ts_pf_out),
        ts_eff=ts_eff,
        avg_p_in  = float(np.nanmean(ts_p_in)),
        avg_p_out = float(np.nanmean(ts_p_out)),
        avg_eff   = float(np.nanmean(ts_eff)),
        min_eff   = float(np.nanmin(ts_eff)),
        max_eff   = float(np.nanmax(ts_eff)),
        avg_losses= float(np.nanmean(ts_p_in - ts_p_out)),
        avg_pf_in = float(np.nanmean(ts_pf_in)),
        avg_pf_out= float(np.nanmean(ts_pf_out)),
    )


# ══════════════════════════════════════════════════════════════════════════
#  DAQ Manager
# ══════════════════════════════════════════════════════════════════════════
class MotorDAQManager:

    def __init__(self, chassis: str, ip: str,
                 cfg: ChannelConfig, error_queue: queue.Queue):
        self.chassis = chassis or "cDAQ1"
        self.ip      = ip
        self.cfg     = cfg
        self.errors  = error_queue
        self.dev_v   = f"{self.chassis}Mod{MOD_V}"
        self.dev_i   = f"{self.chassis}Mod{MOD_I}"

        self.hw_rate  = 200_000
        self.ac_freq  = 400.0
        self.n_cycles = 3
        self.running  = False

        self.capturing    = False
        self._cap_lock    = threading.Lock()
        self._cap_fh      = None
        self._cap_count   = 0
        self.capture_path: Optional[str] = None

        self.latest_block: Optional[np.ndarray] = None

        self._last_err_t: dict = {}
        self._err_cnt:    dict = {}

    def report_error(self, src: str, msg: str):
        key = (src, msg); now = time.monotonic()
        cnt = self._err_cnt.get(key, 0) + 1
        self._err_cnt[key] = cnt
        if now - self._last_err_t.get(key, 0.0) < 0.5: return
        self._last_err_t[key] = now
        sfx = f"  (x{cnt})" if cnt > 1 else ""
        self._err_cnt[key] = 0
        self.errors.put((datetime.now(), src, msg + sfx))

    def test_connection(self) -> tuple[bool, str]:
        if SIMULATION_MODE:
            return True, "Simulation mode."
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
            devs = [d.name for d in nidaqmx.system.System.local().devices]
            miss = [d for d in (self.dev_v, self.dev_i) if d not in devs]
            if miss:
                return False, f"Not found: {miss}. Available: {devs}"
            nidaqmx.system.Device(self.dev_v).self_test_device()
        except Exception as e:
            return False, f"NI-DAQmx failed: {e}"
        return True, f"Connected to '{self.chassis}' ({self.ip})."

    def start(self):
        if self.running: return
        self.running = True
        threading.Thread(target=self._acq_loop, daemon=True).start()

    def stop(self):
        self.running = False
        self.capturing = False

    def start_capture(self, path: str):
        with self._cap_lock:
            if self._cap_fh: self._cap_fh.close()
            self._cap_fh    = open(path, 'wb')
            self._cap_count = 0
            self.capture_path = path
            self.capturing  = True

    def stop_capture(self) -> int:
        with self._cap_lock:
            self.capturing = False
            if self._cap_fh:
                self._cap_fh.close()
                self._cap_fh = None
                if self.capture_path:
                    meta = self.cfg.to_meta(self.hw_rate, self.ac_freq)
                    meta["n_samples"] = self._cap_count
                    sc = self.capture_path.replace(".bin", "_meta.json")
                    with open(sc, "w") as f:
                        json.dump(meta, f, indent=2)
        return self._cap_count

    def _acq_loop(self):
        v_chs = self.cfg.v_active
        i_chs = self.cfg.i_active
        if not v_chs and not i_chs:
            self.report_error("Acq", "No channels selected.")
            self.running = False
            return

        block_sec = max(0.01, self.n_cycles / max(1.0, self.ac_freq))
        n_samp    = max(1, int(round(self.hw_rate * block_sec)))

        task = None
        if not SIMULATION_MODE:
            try:
                task = nidaqmx.Task()
                for ch in v_chs:
                    task.ai_channels.add_ai_voltage_chan(
                        f"{self.dev_v}/ai{ch}",
                        min_val=-10.0, max_val=10.0,
                        terminal_config=TerminalConfiguration.DIFF)
                for ch in i_chs:
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
                    t_arr = np.linspace(time.perf_counter()-block_sec,
                                        time.perf_counter(), n_samp, endpoint=False)
                    freq  = self.ac_freq
                    vsigs = np.column_stack([
                        5.43 * np.sin(2*np.pi*freq*t_arr + j*2*np.pi/3)
                        for j in range(len(v_chs))])
                    isigs = np.column_stack([
                        0.50 * np.sin(2*np.pi*freq*t_arr + j*2*np.pi/3)
                        for j in range(len(i_chs))])
                    block = np.hstack([vsigs, isigs]).astype(np.float32)
                else:
                    try:
                        data  = task.read(
                            number_of_samples_per_channel=nidaqmx.constants.READ_ALL_AVAILABLE,
                            timeout=2.0)
                        first = np.atleast_1d(data[0]) if data else np.array([])
                        if len(first) == 0:
                            time.sleep(max(0.0, block_sec-(time.perf_counter()-t0)))
                            continue
                        block = np.column_stack(
                            [np.atleast_1d(c) for c in data]).astype(np.float32)
                    except Exception as e:
                        self.report_error("Acq", str(e))
                        try:
                            task.in_stream.relative_to = ReadRelativeTo.MOST_RECENT_SAMPLE
                            task.in_stream.offset = 0
                        except Exception: pass
                        time.sleep(0.2)
                        continue

                self.latest_block = block
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
#  Main Application
# ══════════════════════════════════════════════════════════════════════════
class MotorAnalyserApp(tk.Tk):

    def __init__(self):
        super().__init__()
        self.title("Motor Controller Analyser  —  cDAQ-9189")
        self.configure(bg=C_BG)
        self.geometry("1500x950")
        self.minsize(1200, 780)

        self.error_queue: queue.Queue = queue.Queue()
        self.cfg  = ChannelConfig()
        self.daq: Optional[MotorDAQManager] = None
        self._connected_ok = False

        # Config from JSON
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

        # Capture state
        self._cap_path:  Optional[str]   = None
        self._cap_start: Optional[float] = None

        # Viewer state
        self._viewer_raw:      Optional[np.ndarray] = None
        self._viewer_fs:       float = 200_000.0
        self._viewer_ac_freq:  float = 400.0
        self._viewer_n_samp:   int   = 0
        self._viewer_win_start: int  = 0
        self._viewer_win_len:  int   = 0
        self._viewer_ch_visible: list[bool] = []
        self._viewer_cfg:      Optional[ChannelConfig] = None
        self._viewer_file_path: Optional[str] = None

        # Post-process result
        self._pp_result:   Optional[dict] = None
        self._pp_bin_path: Optional[str]  = None

        self._load_from_json()
        self._build_style()
        self._build_ui()

        if SIMULATION_MODE:
            self._do_connect()
        elif self._json_auto_start:
            self.after(200, self._do_connect)

        self._poll()
        self._poll_errors()

    # ── JSON loading ──────────────────────────────────────────────────────
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

            # Capture config comes from the motor_analyser section.
            # Fall back to the recording section for shared fields
            # (auto_record_on_start, record_start_delay_sec) if not overridden.
            rec = data.get("recording") or {}
            ma  = data.get("motor_analyser") or {}

            self._rec_auto_start   = bool(rec.get("auto_record_on_start",   False))
            self._rec_prefix       = str(ma.get("capture_filename_prefix",
                                          rec.get("log_filename_prefix", "motor_raw")))
            self._rec_start_delay  = float(ma.get("capture_start_delay_sec",
                                            rec.get("record_start_delay_sec", 2.0)))
            # capture_duration_sec: 0 means unlimited (no timed stop)
            cap_dur = float(ma.get("capture_duration_sec", 0))
            self._rec_timed        = cap_dur > 0
            self._rec_duration_sec = cap_dur if cap_dur > 0 else 300.0

            mc  = data.get("module_config", {})
            aic = mc.get("modules_2_to_6_AI_9320", {})
            self._ac_freq  = float(aic.get("ac_frequency_hz",   400.0))
            self._n_cycles = int(aic.get("ac_cycles_per_block",   3))
            ht = aic.get("high_rate_task", {})
            self._hw_rate  = int(ht.get("hw_sample_rate_hz", 200_000))

            # Channel roles from motor_analyser section (16 entries per module)
            valid_roles = set(ROLES)
            v_role_list = ma.get("module_2_voltage_channel_roles", [])
            i_role_list = ma.get("module_3_current_channel_roles", [])
            for ch in range(MAX_CH):
                if ch < len(v_role_list):
                    r = v_role_list[ch]
                    if r in valid_roles:
                        self.cfg.v_roles[ch] = r
                if ch < len(i_role_list):
                    r = i_role_list[ch]
                    if r in valid_roles:
                        self.cfg.i_roles[ch] = r

            # Channel names and calibration from NI_9320_modules_2_to_6
            for rec_ch in data.get("NI_9320_modules_2_to_6", []):
                mod  = rec_ch.get("module")
                ch   = rec_ch.get("channel")
                if ch is None or ch >= MAX_CH: continue
                name  = rec_ch.get("name",   "")
                scale = float(rec_ch.get("scale",  1.0))
                off   = float(rec_ch.get("offset", 0.0))
                if mod == MOD_V:
                    if name: self.cfg.v_names[ch] = name
                    self.cfg.v_cal[ch] = (scale, off)
                elif mod == MOD_I:
                    if name: self.cfg.i_names[ch] = name
                    self.cfg.i_cal[ch] = (scale, off)

        except Exception as e:
            self.error_queue.put((datetime.now(), "JSON", f"Load failed: {e}"))

    def _save_motor_analyser_config(self):
        """Write capture settings and channel roles back to the
        motor_analyser section of cdaq_calibration.json, using the
        **existing-merge** pattern so all other sections are preserved.
        Called when the user clicks 'Apply Config' on the Channel Config tab.
        """
        if not os.path.exists(CAL_FILE):
            return
        try:
            with open(CAL_FILE) as f:
                existing = json.load(f)

            # Capture duration: 0 = unlimited, otherwise the number of seconds
            try:
                cap_dur = float(self._dur_var.get())
            except Exception:
                cap_dur = 0.0

            existing["motor_analyser"] = {
                **existing.get("motor_analyser", {}),   # preserve unknown keys
                "capture_filename_prefix":          self._pfx_var.get().strip() or "motor_raw",
                "capture_start_delay_sec":          self._rec_start_delay,
                "capture_duration_sec":             cap_dur,
                "module_2_voltage_channel_roles":   [self.cfg.v_roles[c]
                                                     for c in range(MAX_CH)],
                "module_3_current_channel_roles":   [self.cfg.i_roles[c]
                                                     for c in range(MAX_CH)],
                "_note": (
                    "capture_filename_prefix: base name for .bin files "
                    "(timestamp appended). "
                    "capture_start_delay_sec: seconds to wait after "
                    "acquisition starts before auto-recording begins. "
                    "capture_duration_sec: seconds to record "
                    "(0 = unlimited). "
                    "Valid roles: V-In, V-Out, I-In, I-Out, Unused."
                ),
            }

            with open(CAL_FILE, "w") as f:
                json.dump(existing, f, indent=2)

            self.error_queue.put((datetime.now(), "JSON",
                                  "motor_analyser config saved to "
                                  f"{os.path.basename(CAL_FILE)}"))
        except Exception as e:
            self.error_queue.put((datetime.now(), "JSON",
                                  f"Save failed: {e}"))

    # ── Style ─────────────────────────────────────────────────────────────
    def _build_style(self):
        s = ttk.Style(self)
        s.theme_use("clam")
        s.configure(".", background=C_BG, foreground=C_TEXT,
                    fieldbackground=C_INPUT, troughcolor=C_BORD)
        s.configure("TNotebook", background=C_BG, tabmargins=[2,4,2,0])
        s.configure("TNotebook.Tab", background=C_PANEL, foreground=C_MUTED,
                    padding=[12,5], font=FONT_SMALL)
        s.map("TNotebook.Tab", background=[("selected",C_BG)],
              foreground=[("selected",C_ACC)])
        s.configure("TLabelframe", background=C_BG, foreground=C_ACC,
                    relief="flat", borderwidth=1)
        s.configure("TLabelframe.Label", background=C_BG,
                    foreground=C_ACC, font=FONT_BOLD)
        for nm,fg in [("G.TButton",C_GREEN),("R.TButton",C_RED),
                       ("A.TButton",C_ACC),("Y.TButton",C_YEL)]:
            s.configure(nm, background=C_PANEL, foreground=fg,
                        relief="flat", padding=[8,4])
            s.map(nm, background=[("active",C_BORD)])
        s.configure("TEntry", fieldbackground=C_INPUT, foreground=C_TEXT)
        s.configure("TProgressbar", troughcolor=C_BORD, background=C_ACC)
        s.configure("TCombobox", fieldbackground=C_INPUT, foreground=C_TEXT,
                    background=C_PANEL, arrowcolor=C_ACC)

    # ══════════════════════════════════════════════════════════════════════
    #  UI
    # ══════════════════════════════════════════════════════════════════════
    def _build_ui(self):
        # Top bar
        top = tk.Frame(self, bg=C_PANEL, pady=6, padx=12)
        top.pack(fill="x")

        tk.Label(top, text="Motor Analyser", font=FONT_TITLE,
                 bg=C_PANEL, fg=C_ACC).pack(side="left")
        tk.Label(top, text=f" [{'SIM' if SIMULATION_MODE else 'HW'}]",
                 font=FONT_SMALL, bg=C_PANEL,
                 fg=C_YEL if SIMULATION_MODE else C_GREEN).pack(side="left", padx=4)

        self._chassis_var = tk.StringVar(value=self._json_chassis)
        self._ip_var      = tk.StringVar(value=self._json_ip)
        for lbl, var, w in [("  Chassis:", self._chassis_var, 18),
                              ("  IP:",     self._ip_var,      14)]:
            tk.Label(top, text=lbl, font=FONT_SMALL,
                     bg=C_PANEL, fg=C_MUTED).pack(side="left")
            tk.Entry(top, textvariable=var, width=w,
                     bg=C_INPUT, fg=C_TEXT, relief="flat",
                     font=FONT_MONO, insertbackground=C_TEXT
                     ).pack(side="left", padx=3)

        self._conn_btn  = ttk.Button(top, text="Connect",
                                     style="G.TButton", command=self._connect)
        self._conn_btn.pack(side="left", padx=6)
        self._start_btn = ttk.Button(top, text="▶ Start",
                                     style="G.TButton", command=self._start_acq)
        self._start_btn.pack(side="left", padx=2)
        self._stop_btn  = ttk.Button(top, text="■ Stop",
                                     style="R.TButton", command=self._stop_acq)
        self._stop_btn.pack(side="left", padx=2)
        self._cap_btn   = ttk.Button(top, text="⏺ Capture",
                                     style="Y.TButton", command=self._toggle_capture)
        self._cap_btn.pack(side="left", padx=6)

        self._status_lbl = tk.Label(top, text="● Disconnected",
                                     font=FONT_SMALL, bg=C_PANEL, fg=C_RED)
        self._status_lbl.pack(side="left", padx=8)

        # Notebook
        self._nb = ttk.Notebook(self)
        self._nb.pack(fill="both", expand=True, padx=6, pady=4)

        self._nb.add(self._build_ch_config_tab(), text="  Channel Config  ")
        self._nb.add(self._build_live_tab(),      text="  Live Preview  ")
        self._nb.add(self._build_capture_tab(),   text="  Capture  ")
        self._nb.add(self._build_viewer_tab(),    text="  Waveform Viewer  ")
        self._nb.add(self._build_postproc_tab(),  text="  Post-Process  ")
        self._nb.add(self._build_cal_tab(),       text="  Calibration  ")

        # Error bar
        self._error_log: list = []
        bw = tk.Frame(self, bg="#1c1106")
        bw.pack(fill="x", side="bottom")
        self._err_log_frame  = tk.Frame(bw, bg="#0d0701")
        self._err_log_text   = tk.Text(self._err_log_frame, height=6,
                                        bg="#0d0701", fg=C_RED,
                                        font=FONT_MONOS, wrap="none",
                                        relief="flat", state="disabled")
        esb = ttk.Scrollbar(self._err_log_frame, orient="vertical",
                             command=self._err_log_text.yview)
        self._err_log_text.configure(yscrollcommand=esb.set)
        self._err_log_text.pack(side="left", fill="both",
                                expand=True, padx=(10,0), pady=3)
        esb.pack(side="right", fill="y", pady=3)
        self._err_log_visible = False

        bot = tk.Frame(bw, bg="#1c1106", height=26)
        bot.pack(fill="x", side="bottom")
        bot.pack_propagate(False)
        tk.Label(bot, text="Status:", font=FONT_TINY,
                 bg="#1c1106", fg=C_MUTED).pack(side="left", padx=(10,4))
        self._err_lbl = tk.Label(bot, text="No errors", font=FONT_TINY,
                                  bg="#1c1106", fg=C_MUTED, anchor="w")
        self._err_lbl.pack(side="left", fill="x", expand=True)
        self._err_tog = ttk.Button(bot, text="Show Log (0)",
                                    style="A.TButton",
                                    command=self._toggle_err_log)
        self._err_tog.pack(side="right", padx=4, pady=2)
        ttk.Button(bot, text="Clear", style="R.TButton",
                   command=self._clear_errors).pack(side="right", padx=4, pady=2)

    # ══════════════════════════════════════════════════════════════════════
    #  Tab: Channel Config
    # ══════════════════════════════════════════════════════════════════════
    def _build_ch_config_tab(self):
        tab = tk.Frame(self._nb, bg=C_BG)

        hdr = tk.Frame(tab, bg=C_BG)
        hdr.pack(fill="x", padx=10, pady=6)
        tk.Label(hdr, text="Channel Configuration", font=FONT_HEAD,
                 bg=C_BG, fg=C_ACC).pack(side="left")
        tk.Label(hdr, text="  Set role and name for each channel on Mod2 (Voltage) and Mod3 (Current)",
                 font=FONT_SMALL, bg=C_BG, fg=C_MUTED).pack(side="left", padx=10)
        ttk.Button(hdr, text="Apply Config", style="G.TButton",
                   command=self._apply_ch_config).pack(side="right")

        body = tk.Frame(tab, bg=C_BG)
        body.pack(fill="both", expand=True, padx=10, pady=4)

        self._ch_role_vars: dict[str, list[tk.StringVar]] = {"V": [], "I": []}
        self._ch_name_vars: dict[str, list[tk.StringVar]] = {"V": [], "I": []}

        role_colors = {
            ROLE_V_IN:   C_BLUE, ROLE_V_OUT: "#79c0ff",
            ROLE_I_IN:   C_RED,  ROLE_I_OUT: "#ffa657",
            ROLE_UNUSED: C_MUTED,
        }

        for col, (side, label, roles_list, names_list) in enumerate([
            ("V", f"Module {MOD_V}  —  Voltage Inputs",
             self.cfg.v_roles, self.cfg.v_names),
            ("I", f"Module {MOD_I}  —  Current Inputs",
             self.cfg.i_roles, self.cfg.i_names),
        ]):
            sec = ttk.LabelFrame(body, text=f" {label} ", padding=8)
            sec.grid(row=0, column=col, padx=6, sticky="nsew")
            body.columnconfigure(col, weight=1)

            # Column headers
            for c, (h,w) in enumerate(zip(
                    ["Ch", "Name", "Role"],
                    [4,    24,     10])):
                tk.Label(sec, text=h, font=FONT_BOLD, bg=C_BG,
                         fg=C_MUTED, width=w, anchor="w"
                         ).grid(row=0, column=c, padx=4, sticky="w")

            for ch in range(MAX_CH):
                name_v = tk.StringVar(value=names_list[ch])
                role_v = tk.StringVar(value=roles_list[ch])
                self._ch_name_vars[side].append(name_v)
                self._ch_role_vars[side].append(role_v)

                tk.Label(sec, text=f"{ch:02d}", font=FONT_MONOS,
                         bg=C_BG, fg=C_TEXT, width=4, anchor="w"
                         ).grid(row=ch+1, column=0, padx=4, pady=1, sticky="w")
                tk.Entry(sec, textvariable=name_v, width=22,
                         bg=C_INPUT, fg=C_TEXT, font=FONT_MONOS,
                         insertbackground=C_TEXT, relief="flat"
                         ).grid(row=ch+1, column=1, padx=4, pady=1, sticky="w")

                role_cb = ttk.Combobox(sec, textvariable=role_v,
                                        values=ROLES, width=10,
                                        state="readonly")
                role_cb.grid(row=ch+1, column=2, padx=4, pady=1, sticky="w")

        # Legend
        leg = tk.Frame(tab, bg=C_BG)
        leg.pack(padx=10, pady=6, anchor="w")
        tk.Label(leg, text="Role legend:", font=FONT_SMALL,
                 bg=C_BG, fg=C_MUTED).pack(side="left")
        for role, color in role_colors.items():
            f = tk.Frame(leg, bg=C_BG)
            f.pack(side="left", padx=8)
            tk.Canvas(f, width=12, height=12, bg=color,
                      highlightthickness=0).pack(side="left", padx=(0,3))
            tk.Label(f, text=role, font=FONT_TINY,
                     bg=C_BG, fg=color).pack(side="left")

        return tab

    def _apply_ch_config(self):
        for ch in range(MAX_CH):
            self.cfg.v_roles[ch] = self._ch_role_vars["V"][ch].get()
            self.cfg.i_roles[ch] = self._ch_role_vars["I"][ch].get()
            self.cfg.v_names[ch] = self._ch_name_vars["V"][ch].get()
            self.cfg.i_names[ch] = self._ch_name_vars["I"][ch].get()

        n_v = self.cfg.n_v; n_i = self.cfg.n_i
        self._save_motor_analyser_config()   # persist to JSON immediately
        self._viewer_ch_visible = [True] * self.cfg.n_cols

        if self.daq and self.daq.running:
            messagebox.showwarning("Restart required",
                                   "Stop and restart acquisition to apply new channel config.")
        else:
            messagebox.showinfo("Channel Config Applied",
                                f"Active: {n_v}V + {n_i}I channels  "
                                f"({self.cfg.n_cols} total columns)\n"
                                f"Saved to {os.path.basename(CAL_FILE)}")

    # ══════════════════════════════════════════════════════════════════════
    #  Tab: Live Preview
    # ══════════════════════════════════════════════════════════════════════
    def _build_live_tab(self):
        tab = tk.Frame(self._nb, bg=C_BG)

        # Big efficiency
        eff_f = tk.Frame(tab, bg=C_PANEL, padx=14, pady=8,
                          highlightbackground=C_BORD, highlightthickness=1)
        eff_f.pack(fill="x", padx=10, pady=6)
        tk.Label(eff_f, text="LIVE EFFICIENCY  (current block)",
                 font=FONT_BOLD, bg=C_PANEL, fg=C_MUTED).pack()
        self._live_eff_var = tk.StringVar(value="---")
        tk.Label(eff_f, textvariable=self._live_eff_var,
                 font=FONT_MONOXL, bg=C_PANEL, fg=C_GREEN).pack()

        row_f = tk.Frame(eff_f, bg=C_PANEL)
        row_f.pack()
        self._live_vars: dict[str,tk.StringVar] = {}
        for key, label, fg in [
            ("pin",    "Pin (W)",    C_BLUE),
            ("pout",   "Pout (W)",   C_ACC),
            ("losses", "Losses (W)", C_ORG),
            ("pf_in",  "PF in",      C_TEXT),
            ("pf_out", "PF out",     C_TEXT),
        ]:
            f = tk.Frame(row_f, bg=C_PANEL)
            f.pack(side="left", padx=14)
            tk.Label(f, text=label, font=FONT_TINY, bg=C_PANEL, fg=C_MUTED).pack()
            v = tk.StringVar(value="---")
            self._live_vars[key] = v
            tk.Label(f, textvariable=v, font=FONT_MONOL,
                     bg=C_PANEL, fg=fg).pack()

        # Per-pair table (built dynamically in _update_live_table)
        self._live_table_frame = tk.Frame(tab, bg=C_BG)
        self._live_table_frame.pack(fill="both", expand=True, padx=10, pady=4)
        self._live_pair_rows: list[dict] = []
        self._rebuild_live_table()

        return tab

    def _rebuild_live_table(self):
        for w in self._live_table_frame.winfo_children():
            w.destroy()
        self._live_pair_rows = []
        pairs = self.cfg.paired_phases()
        if not pairs:
            tk.Label(self._live_table_frame,
                     text="No paired V/I channels configured.",
                     font=FONT_SMALL, bg=C_BG, fg=C_MUTED).pack(pady=20)
            return
        hdrs = ["Side","V-Ch","I-Ch","Vrms","Irms","P (W)","S (VA)","PF"]
        for c, h in enumerate(hdrs):
            tk.Label(self._live_table_frame, text=h, font=FONT_BOLD,
                     bg=C_BG, fg=C_MUTED, width=10, anchor="w"
                     ).grid(row=0, column=c, padx=4, pady=2)
        for row_i, (vc_idx, ic_idx, side, vch, ich) in enumerate(pairs):
            bg = C_PANEL if row_i%2==0 else C_BG
            d = {}
            side_fg = C_BLUE if side=="Input" else C_ACC
            tk.Label(self._live_table_frame, text=side, font=FONT_TINY,
                     bg=bg, fg=side_fg, width=8, anchor="w"
                     ).grid(row=row_i+1, column=0, padx=4, pady=1, sticky="w")
            tk.Label(self._live_table_frame,
                     text=f"{self.cfg.v_names[vch][:10]}", font=FONT_TINY,
                     bg=bg, fg=C_BLUE, width=12, anchor="w"
                     ).grid(row=row_i+1, column=1, padx=2, sticky="w")
            tk.Label(self._live_table_frame,
                     text=f"{self.cfg.i_names[ich][:10]}", font=FONT_TINY,
                     bg=bg, fg=C_RED, width=12, anchor="w"
                     ).grid(row=row_i+1, column=2, padx=2, sticky="w")
            for c_i, (key,fg) in enumerate(
                    zip(["vrms","irms","P","S","pf"],
                        [C_ACC,C_RED,C_GREEN,C_TEXT,C_YEL]), start=3):
                var = tk.StringVar(value="---")
                d[key] = var
                tk.Label(self._live_table_frame, textvariable=var,
                         font=FONT_MONOS, bg=bg, fg=fg,
                         width=12, anchor="e"
                         ).grid(row=row_i+1, column=c_i, padx=4, sticky="e")
            self._live_pair_rows.append(d)

    # ══════════════════════════════════════════════════════════════════════
    #  Tab: Capture
    # ══════════════════════════════════════════════════════════════════════
    def _build_capture_tab(self):
        tab = tk.Frame(self._nb, bg=C_BG)

        info = ttk.LabelFrame(tab, text=" Capture Settings ", padding=12)
        info.pack(fill="x", padx=10, pady=8)

        self._rate_lbl = tk.Label(info, text="---", font=FONT_SMALL,
                                   bg=C_BG, fg=C_ACC)
        self._rate_lbl.pack(anchor="w", pady=3)

        # Duration
        df = tk.Frame(info, bg=C_BG)
        df.pack(anchor="w", pady=3)
        tk.Label(df, text="Duration (0=unlimited):", font=FONT_SMALL,
                 bg=C_BG, fg=C_MUTED).pack(side="left")
        self._dur_var = tk.StringVar(value=str(int(self._rec_duration_sec)))
        tk.Entry(df, textvariable=self._dur_var, width=8,
                 bg=C_INPUT, fg=C_TEXT, font=FONT_MONO,
                 insertbackground=C_TEXT, relief="flat").pack(side="left", padx=6)
        tk.Label(df, text="seconds", font=FONT_SMALL,
                 bg=C_BG, fg=C_MUTED).pack(side="left")

        # Prefix
        pf = tk.Frame(info, bg=C_BG)
        pf.pack(anchor="w", pady=3)
        tk.Label(pf, text="File prefix:", font=FONT_SMALL,
                 bg=C_BG, fg=C_MUTED).pack(side="left")
        self._pfx_var = tk.StringVar(value=self._rec_prefix)
        tk.Entry(pf, textvariable=self._pfx_var, width=24,
                 bg=C_INPUT, fg=C_TEXT, font=FONT_MONO,
                 insertbackground=C_TEXT, relief="flat").pack(side="left", padx=6)

        self._cap_status_var = tk.StringVar(value="Not capturing")
        tk.Label(info, textvariable=self._cap_status_var,
                 font=FONT_MONO, bg=C_BG, fg=C_YEL).pack(anchor="w", pady=4)

        self._cap_prog = ttk.Progressbar(info, orient="horizontal",
                                          length=500, mode="determinate")
        self._cap_prog.pack(anchor="w", pady=3)

        # Size estimates
        sz_f = ttk.LabelFrame(tab, text=" File Size Estimates ", padding=8)
        sz_f.pack(fill="x", padx=10, pady=4)
        self._size_lbl = tk.Label(sz_f, text="Configure channels first",
                                   font=FONT_SMALL, bg=C_BG, fg=C_MUTED)
        self._size_lbl.pack(anchor="w")

        return tab

    def _update_rate_labels(self):
        n = self.cfg.n_cols
        rate_mb = self._hw_rate * n * 4 / 1e6 if n > 0 else 0
        self._rate_lbl.config(
            text=f"Rate: {self._hw_rate:,} S/s  ·  "
                 f"{n} columns ({self.cfg.n_v}V + {self.cfg.n_i}I)  ·  "
                 f"{rate_mb:.2f} MB/s")
        parts = []
        for dur, lbl in [(10,"10s"),(30,"30s"),(60,"1min"),(300,"5min")]:
            sz = rate_mb * dur
            u  = "MB" if sz < 1000 else "GB"
            v  = sz if sz < 1000 else sz/1000
            parts.append(f"{lbl}: {v:.1f} {u}")
        self._size_lbl.config(text="  ·  ".join(parts))

    # ══════════════════════════════════════════════════════════════════════
    #  Tab: Waveform Viewer
    # ══════════════════════════════════════════════════════════════════════
    def _build_viewer_tab(self):
        tab = tk.Frame(self._nb, bg=C_BG)

        # Toolbar
        vtb = tk.Frame(tab, bg=C_BG)
        vtb.pack(fill="x", padx=8, pady=4)

        ttk.Button(vtb, text="📂 Open .bin", style="G.TButton",
                   command=self._viewer_open_file).pack(side="left", padx=4)

        self._v_file_lbl = tk.Label(vtb, text="No file loaded",
                                     font=FONT_SMALL, bg=C_BG, fg=C_MUTED)
        self._v_file_lbl.pack(side="left", padx=6)

        self._v_cal_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(vtb, text="Apply cal",
                        variable=self._v_cal_var,
                        command=self._viewer_redraw).pack(side="left", padx=12)

        ttk.Button(vtb, text="↺ Reset", style="A.TButton",
                   command=self._viewer_reset_view).pack(side="right", padx=4)
        ttk.Button(vtb, text="💾 Export CSV", style="Y.TButton",
                   command=self._viewer_export_csv).pack(side="right", padx=4)

        if not HAS_MPL:
            tk.Label(tab, text="matplotlib not installed.\npip install matplotlib",
                     font=FONT_HEAD, bg=C_BG, fg=C_RED).pack(expand=True)
            return tab

        # Channel selector sidebar + plot
        pane = tk.Frame(tab, bg=C_BG)
        pane.pack(fill="both", expand=True)

        # Sidebar
        sb = tk.Frame(pane, bg=C_PANEL, width=200)
        sb.pack(side="left", fill="y", padx=(4,0), pady=4)
        sb.pack_propagate(False)

        # File info
        self._v_info_frame = ttk.LabelFrame(sb, text=" File Info ", padding=5)
        self._v_info_frame.pack(fill="x", padx=6, pady=4)
        self._v_info_vars: dict[str,tk.StringVar] = {}
        for key, lbl in [("samples","Samples"),("dur","Duration"),
                          ("fs","Fs"),("cols","Columns")]:
            row = tk.Frame(self._v_info_frame, bg=C_PANEL)
            row.pack(fill="x", pady=1)
            tk.Label(row, text=f"{lbl}:", font=FONT_TINY,
                     bg=C_PANEL, fg=C_MUTED, width=10, anchor="w").pack(side="left")
            v = tk.StringVar(value="---")
            self._v_info_vars[key] = v
            tk.Label(row, textvariable=v, font=FONT_MONOS,
                     bg=C_PANEL, fg=C_TEXT).pack(side="left")

        # Y-scale
        ys = ttk.LabelFrame(sb, text=" Y Scale ", padding=5)
        ys.pack(fill="x", padx=6, pady=4)
        self._v_yscale = tk.StringVar(value="auto")
        for val, lbl in [("auto","Auto"),("shared","Shared"),("manual","Manual")]:
            tk.Radiobutton(ys, text=lbl, variable=self._v_yscale,
                           value=val, bg=C_PANEL, fg=C_TEXT,
                           selectcolor=C_ACC, activebackground=C_PANEL,
                           font=FONT_TINY, command=self._viewer_redraw
                           ).pack(anchor="w")
        self._v_ymin = tk.StringVar(value="-400")
        self._v_ymax = tk.StringVar(value="+400")
        for lbl, var in [("Min:", self._v_ymin), ("Max:", self._v_ymax)]:
            row = tk.Frame(ys, bg=C_PANEL)
            row.pack(fill="x")
            tk.Label(row, text=lbl, font=FONT_TINY, bg=C_PANEL,
                     fg=C_MUTED, width=5).pack(side="left")
            tk.Entry(row, textvariable=var, width=8, bg=C_INPUT, fg=C_TEXT,
                     font=FONT_MONOS, insertbackground=C_TEXT,
                     relief="flat").pack(side="left")
        ttk.Button(ys, text="Apply", style="A.TButton",
                   command=self._viewer_redraw).pack(pady=2)

        # Channel checkboxes (populated dynamically)
        ch_outer = ttk.LabelFrame(sb, text=" Channels ", padding=5)
        ch_outer.pack(fill="x", padx=6, pady=4)
        qb = tk.Frame(ch_outer, bg=C_PANEL)
        qb.pack(fill="x", pady=(0,4))
        ttk.Button(qb, text="All",  style="G.TButton",
                   command=lambda: self._viewer_set_all(True)).pack(side="left",padx=1)
        ttk.Button(qb, text="None", style="R.TButton",
                   command=lambda: self._viewer_set_all(False)).pack(side="left",padx=1)
        self._v_ch_frame = tk.Frame(ch_outer, bg=C_PANEL)
        self._v_ch_frame.pack(fill="x")
        self._v_ch_vars: list[tk.BooleanVar] = []
        self._viewer_rebuild_ch_list()

        # Stats panel
        self._v_stats_frame = ttk.LabelFrame(sb, text=" Statistics ", padding=5)
        self._v_stats_frame.pack(fill="both", expand=True, padx=6, pady=4)
        self._v_stats_text = tk.Text(self._v_stats_frame, bg=C_PANEL, fg=C_TEXT,
                                      font=FONT_MONOS, relief="flat",
                                      state="disabled", wrap="none", width=28)
        vsb2 = ttk.Scrollbar(self._v_stats_frame, orient="vertical",
                              command=self._v_stats_text.yview)
        self._v_stats_text.configure(yscrollcommand=vsb2.set)
        self._v_stats_text.pack(side="left", fill="both", expand=True)
        vsb2.pack(side="right", fill="y")

        # Plot area
        plot_area = tk.Frame(pane, bg=C_BG)
        plot_area.pack(side="left", fill="both", expand=True, padx=4, pady=4)

        # Nav controls
        nav = tk.Frame(plot_area, bg=C_BG)
        nav.pack(fill="x", pady=(0,3))

        tk.Label(nav, text="Window:", font=FONT_SMALL,
                 bg=C_BG, fg=C_MUTED).pack(side="left")
        self._v_win_ms = tk.StringVar(value="10.0")
        tk.Entry(nav, textvariable=self._v_win_ms, width=7,
                 bg=C_INPUT, fg=C_TEXT, font=FONT_MONO,
                 insertbackground=C_TEXT, relief="flat").pack(side="left", padx=4)
        tk.Label(nav, text="ms", font=FONT_SMALL,
                 bg=C_BG, fg=C_MUTED).pack(side="left")

        for lbl, fn in [("◀◀", lambda: self._viewer_shift(-10)),
                         ("◀",  lambda: self._viewer_shift(-1)),
                         ("▶",  lambda: self._viewer_shift(1)),
                         ("▶▶", lambda: self._viewer_shift(10))]:
            ttk.Button(nav, text=lbl, style="A.TButton", command=fn
                       ).pack(side="left", padx=1)
        ttk.Button(nav, text="🔍+", style="G.TButton",
                   command=lambda: self._viewer_zoom(0.5)).pack(side="left", padx=4)
        ttk.Button(nav, text="🔍−", style="R.TButton",
                   command=lambda: self._viewer_zoom(2.0)).pack(side="left", padx=1)

        self._v_pos_lbl = tk.Label(nav, text="", font=FONT_MONO,
                                    bg=C_BG, fg=C_MUTED)
        self._v_pos_lbl.pack(side="right", padx=8)

        self._v_scroll = ttk.Scrollbar(plot_area, orient="horizontal",
                                        command=self._viewer_on_scroll)
        self._v_scroll.pack(fill="x", side="bottom")

        self._v_fig = Figure(figsize=(10,5.5), facecolor=C_BG)
        self._v_fig.subplots_adjust(left=0.07, right=0.97,
                                     top=0.95, bottom=0.09, hspace=0.08)
        self._v_canvas = FigureCanvasTkAgg(self._v_fig, master=plot_area)
        self._v_canvas.get_tk_widget().pack(fill="both", expand=True)

        tb_frame = tk.Frame(plot_area, bg=C_BG)
        tb_frame.pack(fill="x")
        NavigationToolbar2Tk(self._v_canvas, tb_frame).update()

        self._v_canvas.mpl_connect("motion_notify_event",
                                    self._viewer_mouse_move)
        self.bind("<Left>",  lambda e: self._viewer_shift(-1))
        self.bind("<Right>", lambda e: self._viewer_shift(1))

        return tab

    def _viewer_rebuild_ch_list(self):
        for w in self._v_ch_frame.winfo_children():
            w.destroy()
        self._v_ch_vars = []
        names = self.cfg.all_names() if self._viewer_cfg is None else \
                self._viewer_cfg.all_names()
        roles = self.cfg.all_roles() if self._viewer_cfg is None else \
                self._viewer_cfg.all_roles()
        n = len(names)
        if n == 0:
            tk.Label(self._v_ch_frame, text="No active channels",
                     font=FONT_TINY, bg=C_PANEL, fg=C_MUTED).pack()
            return
        for i in range(n):
            v = tk.BooleanVar(value=True)
            self._v_ch_vars.append(v)
            row = tk.Frame(self._v_ch_frame, bg=C_PANEL)
            row.pack(fill="x", pady=1)
            color = ROLE_COLORS.get(roles[i] if i < len(roles) else ROLE_UNUSED,
                                     C_MUTED)
            tk.Canvas(row, width=9, height=9, bg=color,
                      highlightthickness=0).pack(side="left", padx=(0,3))
            ttk.Checkbutton(row, variable=v,
                            text=f"{names[i][:18]}",
                            style="Dark.TCheckbutton",
                            command=self._viewer_redraw).pack(side="left")
        self._viewer_ch_visible = [True] * n

    def _viewer_set_all(self, state: bool):
        for v in self._v_ch_vars: v.set(state)
        self._viewer_ch_visible = [state] * len(self._v_ch_vars)
        self._viewer_redraw()

    def _viewer_open_file(self, path: str = None):
        if path is None:
            try: init_d = os.path.dirname(os.path.abspath(sys.argv[0]))
            except Exception: init_d = os.getcwd()
            path = filedialog.askopenfilename(
                filetypes=[("Raw binary","*.bin"),("All","*.*")],
                title="Open Motor Capture File",
                initialdir=init_d)
        if not path: return

        def worker():
            try:
                # Load sidecar to get column count
                meta = {}
                sc = path.replace(".bin", "_meta.json")
                if os.path.exists(sc):
                    with open(sc) as f: meta = json.load(f)
                n_col = meta.get("n_columns", self.cfg.n_cols)
                raw   = np.fromfile(path, dtype=np.float32)
                if raw.size % n_col != 0:
                    self.after(0, lambda: messagebox.showerror(
                        "Format error",
                        f"File size {raw.size} not divisible by {n_col}"))
                    return
                raw = raw.reshape(-1, n_col)
                self.after(0, lambda: self._viewer_on_loaded(path, raw, meta))
            except Exception as e:
                self.after(0, lambda: messagebox.showerror("Load error", str(e)))

        self._v_file_lbl.config(text="Loading...", fg=C_YEL)
        threading.Thread(target=worker, daemon=True).start()

    def _viewer_on_loaded(self, path: str, raw: np.ndarray, meta: dict):
        self._viewer_raw       = raw
        self._viewer_file_path = path
        self._viewer_fs        = float(meta.get("hw_rate_hz", self._hw_rate))
        self._viewer_ac_freq   = float(meta.get("ac_freq_hz", self._ac_freq))
        self._viewer_n_samp    = raw.shape[0]

        # Reconstruct cfg from sidecar if possible
        viewer_cfg = ChannelConfig.__new__(ChannelConfig)
        viewer_cfg.v_roles = [ROLE_UNUSED]*MAX_CH
        viewer_cfg.i_roles = [ROLE_UNUSED]*MAX_CH
        viewer_cfg.v_names = [f"V_CH{c:02d}" for c in range(MAX_CH)]
        viewer_cfg.i_names = [f"I_CH{c:02d}" for c in range(MAX_CH)]
        viewer_cfg.v_cal   = self.cfg.v_cal[:]
        viewer_cfg.i_cal   = self.cfg.i_cal[:]

        v_act = meta.get("v_active_chs", self.cfg.v_active)
        i_act = meta.get("i_active_chs", self.cfg.i_active)
        v_names = meta.get("v_names", [])
        i_names = meta.get("i_names", [])
        v_roles = meta.get("v_roles", [ROLE_V_IN]*3 + [ROLE_V_OUT]*3)
        i_roles = meta.get("i_roles", [ROLE_I_IN]*3 + [ROLE_I_OUT]*3)
        for j, ch in enumerate(v_act):
            if ch < MAX_CH:
                viewer_cfg.v_roles[ch] = v_roles[j] if j<len(v_roles) else ROLE_V_IN
                if j<len(v_names): viewer_cfg.v_names[ch] = v_names[j]
        for j, ch in enumerate(i_act):
            if ch < MAX_CH:
                viewer_cfg.i_roles[ch] = i_roles[j] if j<len(i_roles) else ROLE_I_IN
                if j<len(i_names): viewer_cfg.i_names[ch] = i_names[j]
        self._viewer_cfg = viewer_cfg

        dur  = self._viewer_n_samp / self._viewer_fs
        size = os.path.getsize(path)
        self._v_info_vars["samples"].set(f"{self._viewer_n_samp:,}")
        self._v_info_vars["dur"].set(f"{dur:.2f}s")
        self._v_info_vars["fs"].set(f"{self._viewer_fs:,.0f}")
        self._v_info_vars["cols"].set(f"{raw.shape[1]}")

        self._v_file_lbl.config(
            text=f"{os.path.basename(path)}  ({dur:.2f}s  ·  {size/1e6:.1f}MB)",
            fg=C_GREEN)

        # Default to 5 cycles
        win = max(100, int(self._viewer_fs / self._viewer_ac_freq * 5))
        self._viewer_win_start = 0
        self._viewer_win_len   = min(win, self._viewer_n_samp)
        self._v_win_ms.set(f"{self._viewer_win_len/self._viewer_fs*1000:.2f}")

        self._viewer_rebuild_ch_list()
        self._viewer_update_scroll()
        self._viewer_redraw()

        # Switch to Waveform Viewer tab
        self._nb.select(3)

    def _viewer_calibrated(self, raw_win: np.ndarray) -> np.ndarray:
        cfg = self._viewer_cfg or self.cfg
        out = raw_win.astype(np.float64)
        if not self._v_cal_var.get():
            return out
        cals = cfg.all_cals()
        for col, (s, o) in enumerate(cals):
            if col < out.shape[1]:
                out[:, col] = raw_win[:, col] * s + o
        return out

    def _viewer_redraw(self):
        if self._viewer_raw is None or not HAS_MPL: return
        new_len = max(10, int(float(self._v_win_ms.get() or 10)
                              * self._viewer_fs / 1000))
        if new_len != self._viewer_win_len:
            self._viewer_win_len   = new_len
            self._viewer_win_start = int(np.clip(
                self._viewer_win_start, 0,
                max(0, self._viewer_n_samp - self._viewer_win_len)))

        s = self._viewer_win_start
        e = min(s + self._viewer_win_len, self._viewer_n_samp)
        raw_win = self._viewer_raw[s:e]
        cal_win = self._viewer_calibrated(raw_win)

        active = [i for i, v in enumerate(self._v_ch_vars)
                  if v.get() and i < cal_win.shape[1]]
        if not active:
            self._v_fig.clf()
            ax = self._v_fig.add_subplot(111)
            ax.set_facecolor(C_BG)
            ax.text(0.5, 0.5, "No channels selected",
                    ha="center", va="center",
                    color=C_MUTED, fontsize=11, transform=ax.transAxes)
            self._v_canvas.draw_idle()
            return

        cfg   = self._viewer_cfg or self.cfg
        names = cfg.all_names()
        roles = cfg.all_roles()

        # Group into V (first n_v) and I (rest)
        n_v    = cfg.n_v
        v_act  = [i for i in active if i < n_v]
        i_act  = [i for i in active if i >= n_v]
        n_plots= (1 if v_act else 0) + (1 if i_act else 0)
        if n_plots == 0: return

        self._v_fig.clf()
        t_ms   = np.arange(len(raw_win)) / self._viewer_fs * 1000
        axes   = []
        yscale = self._v_yscale.get()
        pidx   = 1

        def make_ax(title, ch_list, ylabel):
            nonlocal pidx
            ax = self._v_fig.add_subplot(n_plots, 1, pidx,
                                          sharex=axes[0] if axes else None)
            ax.set_facecolor(C_BG)
            ax.tick_params(colors=C_MUTED, labelsize=8)
            for sp in ax.spines.values(): sp.set_edgecolor(C_BORD)
            ax.grid(True, color=C_BORD, linewidth=0.4, alpha=0.5)
            ax.set_ylabel(ylabel, color=C_MUTED, fontsize=8)
            ax.set_title(title, color=C_TEXT, fontsize=8, pad=3)
            for ch in ch_list:
                role  = roles[ch] if ch < len(roles) else ROLE_UNUSED
                color = ROLE_COLORS.get(role, C_MUTED)
                lw    = 0.7 if len(t_ms) > 5000 else 1.0
                ax.plot(t_ms, cal_win[:, ch], color=color,
                        linewidth=lw, label=names[ch][:16], alpha=0.9)
            if yscale == "shared":
                mx = max(1e-6, np.nanmax(np.abs(cal_win[:, ch_list])))
                ax.set_ylim(-mx*1.1, mx*1.1)
            elif yscale == "manual":
                try: ax.set_ylim(float(self._v_ymin.get()),
                                  float(self._v_ymax.get()))
                except ValueError: pass
            ax.legend(loc="upper right", fontsize=7,
                      facecolor=C_PANEL, edgecolor=C_BORD,
                      labelcolor=C_TEXT, framealpha=0.85,
                      ncol=min(3, len(ch_list)))
            axes.append(ax); pidx += 1

        if v_act: make_ax("Voltage", v_act,
                           "V (cal)" if self._v_cal_var.get() else "V (raw)")
        if i_act: make_ax("Current", i_act,
                           "A (cal)" if self._v_cal_var.get() else "A (raw)")
        if axes:
            axes[-1].set_xlabel("Time (ms)", color=C_MUTED, fontsize=8)

        self._v_fig.patch.set_facecolor(C_BG)
        self._v_pos_lbl.config(
            text=f"t = {s/self._viewer_fs*1000:.3f} – "
                 f"{e/self._viewer_fs*1000:.3f} ms  ({self._viewer_win_len:,} samp)")
        self._v_canvas.draw_idle()
        self._viewer_update_stats(cal_win, active, cfg)

    def _viewer_update_stats(self, cal_win, active, cfg):
        lines = ["── Channel Stats ───────────────────────────",
                 f"{'Ch':<4} {'Name':<16} {'RMS':>8} {'Peak':>8} {'Hz':>6}",
                 "─" * 46]
        names = cfg.all_names()
        for ch in active:
            st = channel_stats(cal_win[:, ch], self._viewer_fs)
            if not st: continue
            nm = names[ch][:14] if ch < len(names) else f"ch{ch}"
            side = "V" if ch < cfg.n_v else "I"
            rms  = f"{st['rms']:.3f}"
            peak = f"{st['peak']:.3f}"
            hz   = f"{st['freq']:.0f}" if np.isfinite(st['freq']) else "---"
            lines.append(f"{side}{ch%cfg.n_v if cfg.n_v>0 else ch:<3} "
                         f"{nm:<16} {rms:>8} {peak:>8} {hz:>6}")

        # Power analysis
        lines += ["", "── Power Analysis ──────────────────────────"]
        pairs = cfg.paired_phases()
        if not pairs:
            lines.append("  (No paired V/I channels)")
        else:
            spc   = self._viewer_fs / max(1.0, self._viewer_ac_freq)
            n_cyc = max(1, int(cal_win.shape[0] // spc))
            n_use = int(round(n_cyc * spc))
            w     = cal_win[:n_use]
            lines.append(f"  {n_cyc} cycles  ({n_use} samples)")
            lines += ["",
                      f"{'Side':<6} {'P(W)':>9} {'S(VA)':>9} {'PF':>7}",
                      "─" * 35]
            p_in = p_out = s_in = s_out = 0.0
            for vc_idx, ic_idx, side, vch, ich in pairs:
                if (vc_idx is None or ic_idx is None or
                        vc_idx >= w.shape[1] or ic_idx >= w.shape[1]):
                    continue
                v = w[:, vc_idx]; i = w[:, ic_idx]
                P    = float(np.nanmean(v*i))
                vrms = float(np.sqrt(np.nanmean(v**2)))
                irms = float(np.sqrt(np.nanmean(i**2)))
                S    = vrms*irms
                pf   = P/S if S>1e-6 else 0.0
                lbl  = f"{'IN' if side=='Input' else 'OUT'}"
                lines.append(f"{lbl:<6} {P:>9.2f} {S:>9.2f} {pf:>7.4f}")
                if side == "Input": p_in+=P; s_in+=S
                else:               p_out+=P; s_out+=S
            lines += ["─"*35,
                      f"{'3Φ IN':<6} {p_in:>9.2f} {s_in:>9.2f} "
                      f"{p_in/s_in:>7.4f}" if s_in>1e-6 else "3Φ IN  ---",
                      f"{'3Φ OUT':<6} {p_out:>9.2f} {s_out:>9.2f} "
                      f"{p_out/s_out:>7.4f}" if s_out>1e-6 else "3Φ OUT ---",
                      ""]
            if p_in > 1e-3:
                eff = p_out/p_in*100
                bar = "█"*int(eff/5) + "░"*(20-int(eff/5))
                lines += [f"  Efficiency : {eff:.3f}%",
                           f"  [{bar}]",
                           f"  Losses     : {p_in-p_out:.3f} W"]

        self._v_stats_text.config(state="normal")
        self._v_stats_text.delete("1.0","end")
        self._v_stats_text.insert("end", "\n".join(lines))
        self._v_stats_text.config(state="disabled")

    def _viewer_shift(self, steps):
        if self._viewer_raw is None: return
        step = max(1, self._viewer_win_len//2)
        self._viewer_win_start = int(np.clip(
            self._viewer_win_start + steps*step, 0,
            max(0, self._viewer_n_samp - self._viewer_win_len)))
        self._viewer_update_scroll(); self._viewer_redraw()

    def _viewer_zoom(self, factor):
        if self._viewer_raw is None: return
        cen     = self._viewer_win_start + self._viewer_win_len//2
        new_len = int(np.clip(self._viewer_win_len*factor, 10, self._viewer_n_samp))
        self._viewer_win_start = int(np.clip(cen-new_len//2, 0,
                                              self._viewer_n_samp-new_len))
        self._viewer_win_len   = new_len
        self._v_win_ms.set(f"{new_len/self._viewer_fs*1000:.2f}")
        self._viewer_update_scroll(); self._viewer_redraw()

    def _viewer_reset_view(self):
        if self._viewer_raw is None: return
        win = max(100, int(self._viewer_fs/self._viewer_ac_freq*5))
        self._viewer_win_start = 0
        self._viewer_win_len   = min(win, self._viewer_n_samp)
        self._v_win_ms.set(f"{self._viewer_win_len/self._viewer_fs*1000:.2f}")
        self._viewer_update_scroll(); self._viewer_redraw()

    def _viewer_on_scroll(self, *args):
        if self._viewer_raw is None: return
        action = args[0]
        if action == "moveto":
            frac = float(args[1])
            self._viewer_win_start = int(np.clip(
                frac*self._viewer_n_samp, 0,
                max(0, self._viewer_n_samp-self._viewer_win_len)))
        elif action == "scroll":
            self._viewer_shift(int(args[1]))
        self._viewer_update_scroll(); self._viewer_redraw()

    def _viewer_update_scroll(self):
        if self._viewer_n_samp == 0: self._v_scroll.set(0,1); return
        lo = self._viewer_win_start / self._viewer_n_samp
        hi = min(1.0, (self._viewer_win_start+self._viewer_win_len)/self._viewer_n_samp)
        self._v_scroll.set(lo, hi)

    def _viewer_mouse_move(self, event):
        if event.inaxes and event.xdata is not None:
            abs_ms = self._viewer_win_start/self._viewer_fs*1000 + event.xdata
            self._v_pos_lbl.config(
                text=f"cursor {event.xdata:.3f} ms  |  abs {abs_ms:.3f} ms")

    def _viewer_export_csv(self):
        if self._viewer_raw is None:
            messagebox.showwarning("No file","Open a .bin file first."); return
        try: d = os.path.dirname(os.path.abspath(sys.argv[0]))
        except Exception: d = os.getcwd()
        path = filedialog.asksaveasfilename(
            defaultextension=".csv",
            filetypes=[("CSV","*.csv"),("All","*.*")],
            initialdir=d,
            initialfile=f"view_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv")
        if not path: return
        s    = self._viewer_win_start
        e    = min(s+self._viewer_win_len, self._viewer_n_samp)
        raw  = self._viewer_raw[s:e]
        cal  = self._viewer_calibrated(raw)
        cfg  = self._viewer_cfg or self.cfg
        hdr  = ["Sample","Time_ms"] + cfg.all_names()
        with open(path,"w",newline="") as f:
            w = csv.writer(f)
            w.writerow(hdr)
            for n, row in enumerate(cal):
                t_ms = (s+n)/self._viewer_fs*1000
                w.writerow([s+n, f"{t_ms:.6f}"] + [f"{v:.6f}" for v in row])
        messagebox.showinfo("Exported",
                            f"{len(cal):,} samples → {os.path.basename(path)}")

    # ══════════════════════════════════════════════════════════════════════
    #  Tab: Post-Process
    # ══════════════════════════════════════════════════════════════════════
    def _build_postproc_tab(self):
        tab = tk.Frame(self._nb, bg=C_BG)

        ctrl = tk.Frame(tab, bg=C_BG)
        ctrl.pack(fill="x", padx=10, pady=6)
        ttk.Button(ctrl, text="📂 Load .bin", style="A.TButton",
                   command=self._pp_load).pack(side="left", padx=4)
        tk.Label(ctrl, text="  Signal freq (Hz):", font=FONT_SMALL,
                 bg=C_BG, fg=C_MUTED).pack(side="left")
        self._pp_freq_var = tk.StringVar(value=str(int(self._ac_freq)))
        tk.Entry(ctrl, textvariable=self._pp_freq_var, width=6,
                 bg=C_INPUT, fg=C_TEXT, font=FONT_MONO,
                 insertbackground=C_TEXT, relief="flat").pack(side="left", padx=4)
        ttk.Button(ctrl, text="⚙ Process", style="G.TButton",
                   command=self._pp_run).pack(side="left", padx=8)
        ttk.Button(ctrl, text="💾 Export CSV", style="Y.TButton",
                   command=self._pp_export_csv).pack(side="left", padx=4)
        self._pp_file_lbl = tk.Label(ctrl, text="No file",
                                      font=FONT_SMALL, bg=C_BG, fg=C_MUTED)
        self._pp_file_lbl.pack(side="left", padx=14)

        self._pp_prog = ttk.Progressbar(tab, orient="horizontal",
                                         length=600, mode="determinate")
        self._pp_prog.pack(padx=10, pady=4, anchor="w")

        sum_f = ttk.LabelFrame(tab, text=" Average over file ", padding=10)
        sum_f.pack(fill="x", padx=10, pady=6)
        self._pp_vars: dict[str,tk.StringVar] = {}
        cells = [
            [("avg_eff","Avg Efficiency %", C_GREEN),
             ("min_eff","Min Efficiency %", C_RED),
             ("max_eff","Max Efficiency %", C_GREEN)],
            [("avg_p_in","Avg Pin (W)",     C_BLUE),
             ("avg_p_out","Avg Pout (W)",   C_ACC),
             ("avg_losses","Avg Losses (W)",C_ORG)],
            [("avg_pf_in","Avg PF in",      C_TEXT),
             ("avg_pf_out","Avg PF out",    C_TEXT)],
        ]
        for r, row in enumerate(cells):
            for c, (key,lbl,fg) in enumerate(row):
                f = tk.Frame(sum_f, bg=C_BG)
                f.grid(row=r, column=c, padx=18, pady=5, sticky="w")
                tk.Label(f, text=lbl, font=FONT_TINY, bg=C_BG, fg=C_MUTED).pack(anchor="w")
                v = tk.StringVar(value="---")
                self._pp_vars[key] = v
                tk.Label(f, textvariable=v, font=FONT_MONOL,
                         bg=C_BG, fg=fg).pack(anchor="w")

        return tab

    def _pp_load(self):
        try: d = os.path.dirname(os.path.abspath(sys.argv[0]))
        except Exception: d = os.getcwd()
        path = filedialog.askopenfilename(
            filetypes=[("Raw binary","*.bin"),("All","*.*")],
            initialdir=d, title="Load Capture File")
        if not path: return
        self._pp_bin_path = path
        n = os.path.getsize(path) // (self.cfg.n_cols * 4)
        dur = n / self._hw_rate
        self._pp_file_lbl.config(
            text=f"{os.path.basename(path)}  ({n:,} samp · {dur:.1f}s)",
            fg=C_ACC)
        self._pp_result = None
        self._pp_prog["value"] = 0

    def _pp_run(self):
        if not self._pp_bin_path or not os.path.exists(self._pp_bin_path):
            messagebox.showwarning("No file","Load a .bin file first."); return
        try: ac_freq = float(self._pp_freq_var.get())
        except ValueError: messagebox.showerror("Error","Invalid frequency."); return
        self._pp_prog["value"] = 0
        self._pp_file_lbl.config(text="Processing...", fg=C_YEL)
        def worker():
            try:
                result = process_file_chunked(
                    self._pp_bin_path, self.cfg, ac_freq, self._hw_rate,
                    progress_cb=lambda p: self.after(
                        0, lambda: self._pp_prog.__setitem__("value", p)))
                self.after(0, lambda: self._pp_show_result(result))
            except Exception as e:
                self.after(0, lambda: messagebox.showerror("Error", str(e)))
        threading.Thread(target=worker, daemon=True).start()

    def _pp_show_result(self, res: dict):
        self._pp_result = res
        if not res: self._pp_file_lbl.config(text="No result",fg=C_RED); return
        def f(v, sfx=""):
            try: return f"{v:.3f}{sfx}" if np.isfinite(v) else "---"
            except: return "---"
        self._pp_vars["avg_eff"].set(f(res.get("avg_eff",float("nan")),"%"))
        self._pp_vars["min_eff"].set(f(res.get("min_eff",float("nan")),"%"))
        self._pp_vars["max_eff"].set(f(res.get("max_eff",float("nan")),"%"))
        self._pp_vars["avg_p_in"].set(f(res.get("avg_p_in",0)))
        self._pp_vars["avg_p_out"].set(f(res.get("avg_p_out",0)))
        self._pp_vars["avg_losses"].set(f(res.get("avg_losses",0)))
        self._pp_vars["avg_pf_in"].set(f(res.get("avg_pf_in",0)))
        self._pp_vars["avg_pf_out"].set(f(res.get("avg_pf_out",0)))
        n = res.get("n_chunks",0)
        self._pp_file_lbl.config(text=f"Done — {n}s processed", fg=C_GREEN)
        self._pp_prog["value"] = 100

    def _pp_export_csv(self):
        if not self._pp_result:
            messagebox.showwarning("No results","Run post-processing first."); return
        try: d = os.path.dirname(os.path.abspath(sys.argv[0]))
        except Exception: d = os.getcwd()
        path = filedialog.asksaveasfilename(
            defaultextension=".csv",
            filetypes=[("CSV","*.csv")],
            initialdir=d,
            initialfile=f"results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv")
        if not path: return
        res = self._pp_result
        rows = [["Second","Pin_W","Pout_W","Losses_W","PF_in","PF_out","Efficiency_pct"]]
        for s in range(res["n_chunks"]):
            eff = res["ts_eff"][s]
            rows.append([s+1,
                         f"{res['ts_p_in'][s]:.4f}",
                         f"{res['ts_p_out'][s]:.4f}",
                         f"{res['ts_p_in'][s]-res['ts_p_out'][s]:.4f}",
                         f"{res['ts_pf_in'][s]:.4f}",
                         f"{res['ts_pf_out'][s]:.4f}",
                         f"{float(eff):.4f}" if np.isfinite(eff) else ""])
        with open(path,"w",newline="") as f:
            csv.writer(f).writerows(rows)
        messagebox.showinfo("Exported", os.path.basename(path))

    # ══════════════════════════════════════════════════════════════════════
    #  Tab: Calibration
    # ══════════════════════════════════════════════════════════════════════
    def _build_cal_tab(self):
        tab = tk.Frame(self._nb, bg=C_BG)
        hdr = tk.Frame(tab, bg=C_BG)
        hdr.pack(fill="x", padx=10, pady=4)
        tk.Label(hdr, text="Scale & Offset  —  applied during post-processing",
                 font=FONT_HEAD, bg=C_BG, fg=C_ACC).pack(side="left")
        ttk.Button(hdr, text="Apply", style="G.TButton",
                   command=self._apply_cal).pack(side="right")

        body = tk.Frame(tab, bg=C_BG)
        body.pack(fill="both", expand=True, padx=10, pady=4)
        self._vcal_s: list[tk.StringVar] = []
        self._vcal_o: list[tk.StringVar] = []
        self._ical_s: list[tk.StringVar] = []
        self._ical_o: list[tk.StringVar] = []

        for col,(side,sl,ol,cal_src) in enumerate([
            ("Voltage  Mod2", self._vcal_s, self._vcal_o, self.cfg.v_cal),
            ("Current  Mod3", self._ical_s, self._ical_o, self.cfg.i_cal),
        ]):
            sec = ttk.LabelFrame(body, text=f" {side} ", padding=8)
            sec.grid(row=0, column=col, padx=6, sticky="nsew")
            body.columnconfigure(col, weight=1)
            for c,(h,w) in enumerate(zip(
                    ["Ch","Name","Scale","Offset"],[4,22,12,12])):
                tk.Label(sec, text=h, font=FONT_BOLD, bg=C_BG,
                         fg=C_MUTED, width=w, anchor="w"
                         ).grid(row=0,column=c,padx=3,sticky="w")
            for ch in range(MAX_CH):
                s_v = tk.StringVar(value=str(cal_src[ch][0]))
                o_v = tk.StringVar(value=str(cal_src[ch][1]))
                sl.append(s_v); ol.append(o_v)
                tk.Label(sec, text=f"{ch:02d}", font=FONT_MONOS, bg=C_BG,
                         fg=C_TEXT, width=4, anchor="w"
                         ).grid(row=ch+1,column=0,padx=3,pady=1,sticky="w")
                nm  = (self.cfg.v_names[ch] if col==0 else self.cfg.i_names[ch])[:20]
                tk.Label(sec, text=nm, font=FONT_TINY, bg=C_BG,
                         fg=C_MUTED, width=22, anchor="w"
                         ).grid(row=ch+1,column=1,padx=3,pady=1,sticky="w")
                tk.Entry(sec, textvariable=s_v, width=12, bg=C_INPUT, fg=C_TEXT,
                         font=FONT_MONOS, insertbackground=C_TEXT, relief="flat"
                         ).grid(row=ch+1,column=2,padx=3,pady=1)
                tk.Entry(sec, textvariable=o_v, width=12, bg=C_INPUT, fg=C_TEXT,
                         font=FONT_MONOS, insertbackground=C_TEXT, relief="flat"
                         ).grid(row=ch+1,column=3,padx=3,pady=1)
        return tab

    def _apply_cal(self):
        errs = []
        for ch in range(MAX_CH):
            try:
                self.cfg.v_cal[ch] = (float(self._vcal_s[ch].get()),
                                       float(self._vcal_o[ch].get()))
            except ValueError: errs.append(f"V CH{ch}")
            try:
                self.cfg.i_cal[ch] = (float(self._ical_s[ch].get()),
                                       float(self._ical_o[ch].get()))
            except ValueError: errs.append(f"I CH{ch}")
        if errs: messagebox.showwarning("Invalid","\n".join(errs))
        else:    messagebox.showinfo("Cal Applied","Scale/offset updated.")

    # ══════════════════════════════════════════════════════════════════════
    #  Connection & acquisition
    # ══════════════════════════════════════════════════════════════════════
    def _connect(self):
        chassis = self._chassis_var.get().strip()
        ip      = self._ip_var.get().strip()
        self.daq = MotorDAQManager(chassis, ip, self.cfg, self.error_queue)
        if SIMULATION_MODE:
            self._do_connect(); return
        self._conn_btn.config(state="disabled", text="Testing...")
        self._status_lbl.config(text="Testing...", fg=C_YEL)
        def worker():
            ok, msg = self.daq.test_connection()
            self.after(0, lambda: self._on_connect_result(ok, msg))
        threading.Thread(target=worker, daemon=True).start()

    def _do_connect(self):
        if not self.daq:
            chassis = self._chassis_var.get().strip()
            ip      = self._ip_var.get().strip()
            self.daq = MotorDAQManager(chassis, ip, self.cfg, self.error_queue)
        self._connected_ok = True
        self._conn_btn.config(text="Reconnect")
        mode = "SIM" if SIMULATION_MODE else "HW"
        self._status_lbl.config(text=f"● Connected ({mode})", fg=C_GREEN)
        if self._json_auto_start: self.after(500, self._start_acq)

    def _on_connect_result(self, ok, msg):
        self._conn_btn.config(state="normal", text="Reconnect")
        self._connected_ok = ok
        if ok:
            self._status_lbl.config(text=f"● {msg}", fg=C_GREEN)
            if self._json_auto_start: self.after(500, self._start_acq)
        else:
            self._status_lbl.config(text="● Connection failed", fg=C_RED)
            self.error_queue.put((datetime.now(), "Connection", msg))
            messagebox.showerror("Connection Failed", msg)

    def _ensure_connected(self) -> bool:
        if not self.daq or (not SIMULATION_MODE and not self._connected_ok):
            messagebox.showwarning("Not Connected","Connect first."); return False
        return True

    def _start_acq(self):
        if not self._ensure_connected(): return
        self._apply_ch_config()  # sync UI to cfg silently
        if self.cfg.n_cols == 0:
            messagebox.showwarning("No channels",
                                   "Assign at least one channel on the Channel Config tab.")
            return
        self.daq.cfg      = self.cfg
        self.daq.hw_rate  = self._hw_rate
        self.daq.ac_freq  = self._ac_freq
        self.daq.n_cycles = self._n_cycles
        self.daq.start()
        self._update_rate_labels()
        self._rebuild_live_table()
        self._status_lbl.config(
            text=f"● Running  {self._hw_rate:,} S/s  "
                 f"{self.cfg.n_v}V+{self.cfg.n_i}I channels",
            fg=C_GREEN)
        if self._rec_auto_start:
            delay = max(100, int(self._rec_start_delay*1000))
            self.after(delay, self._start_capture_auto)

    def _stop_acq(self):
        if self.daq:
            if self.daq.capturing: self._do_stop_capture()
            self.daq.stop()
        self._status_lbl.config(text="● Stopped", fg=C_MUTED)

    # Apply ch config silently (no popup)
    def _apply_ch_config(self):
        for ch in range(MAX_CH):
            self.cfg.v_roles[ch] = self._ch_role_vars["V"][ch].get()
            self.cfg.i_roles[ch] = self._ch_role_vars["I"][ch].get()
            self.cfg.v_names[ch] = self._ch_name_vars["V"][ch].get()
            self.cfg.i_names[ch] = self._ch_name_vars["I"][ch].get()

    # ══════════════════════════════════════════════════════════════════════
    #  Capture
    # ══════════════════════════════════════════════════════════════════════
    def _make_bin_path(self) -> str:
        try: d = os.path.dirname(os.path.abspath(sys.argv[0]))
        except Exception: d = os.getcwd()
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        return os.path.join(d, f"{self._pfx_var.get().strip() or 'motor_raw'}_{ts}.bin")

    def _toggle_capture(self):
        if not self._ensure_connected(): return
        if not self.daq or not self.daq.running:
            messagebox.showwarning("Not Running","Start acquisition first."); return
        if self.daq.capturing:
            self._do_stop_capture()
        else:
            try: d = os.path.dirname(os.path.abspath(sys.argv[0]))
            except Exception: d = os.getcwd()
            path = filedialog.asksaveasfilename(
                defaultextension=".bin",
                filetypes=[("Raw binary","*.bin"),("All","*.*")],
                title="Save Raw Capture",
                initialdir=d,
                initialfile=os.path.basename(self._make_bin_path()))
            if not path: return
            self._do_start_capture(path)

    def _do_start_capture(self, path: str):
        self._cap_path  = path
        self._cap_start = time.perf_counter()
        self.daq.start_capture(path)
        self._cap_btn.config(text="⏹ Stop Capture")
        self._cap_status_var.set(f"Capturing → {os.path.basename(path)}")
        self.error_queue.put((datetime.now(),"Capture",
                              f"Started: {os.path.basename(path)}"))

    def _do_stop_capture(self):
        n    = self.daq.stop_capture()
        secs = n / max(1, self._hw_rate)
        path = self._cap_path or ""
        self._cap_btn.config(text="⏺ Capture")
        self._cap_status_var.set(
            f"Saved {n:,} samples ({secs:.1f}s)  →  {os.path.basename(path)}")
        self._cap_prog["value"] = 100
        self.error_queue.put((datetime.now(),"Capture",
                              f"Saved {n:,} samples ({secs:.1f}s): {os.path.basename(path)}"))
        # ── Auto-open in Waveform Viewer ──────────────────────────────────
        if path and os.path.exists(path):
            self.after(200, lambda: self._viewer_open_file(path))

    def _start_capture_auto(self):
        if self.daq and self.daq.running:
            self._do_start_capture(self._make_bin_path())

    # ══════════════════════════════════════════════════════════════════════
    #  Poll
    # ══════════════════════════════════════════════════════════════════════
    def _poll(self):
        if self.daq and self.daq.latest_block is not None:
            blk = self.daq.latest_block
            try:
                # Apply calibration to latest block
                cals = self.cfg.all_cals()
                cal  = blk.astype(np.float64).copy()
                for col,(s,o) in enumerate(cals):
                    if col < cal.shape[1]:
                        cal[:, col] = blk[:, col]*s+o
                res = process_window(cal, self.cfg, self._ac_freq, self._hw_rate)
                if res:
                    eff = res.get("efficiency")
                    self._live_eff_var.set(f"{eff:.2f}%" if eff else "---")
                    def f2(v): return f"{v:.2f}" if np.isfinite(v) else "---"
                    self._live_vars["pin"].set(f2(res["p_in"]))
                    self._live_vars["pout"].set(f2(res["p_out"]))
                    self._live_vars["losses"].set(f2(res["losses"]))
                    self._live_vars["pf_in"].set(f"{res['pf_in']:.4f}")
                    self._live_vars["pf_out"].set(f"{res['pf_out']:.4f}")
                    for row_i, pair_res in enumerate(res.get("pairs",[])):
                        if row_i >= len(self._live_pair_rows): break
                        d = self._live_pair_rows[row_i]
                        d["vrms"].set(f"{pair_res['vrms']:8.3f} V")
                        d["irms"].set(f"{pair_res['irms']:8.4f} A")
                        d["P"].set(f"{pair_res['P']:8.2f} W")
                        d["S"].set(f"{pair_res['S']:8.2f} VA")
                        d["pf"].set(f"{pair_res['pf']:.4f}")
            except Exception: pass

        # Capture countdown
        if self.daq and self.daq.capturing and self._cap_start is not None:
            el = time.perf_counter()-self._cap_start
            n  = self.daq._cap_count
            mb = n*self.cfg.n_cols*4/1e6
            m,s = divmod(int(el),60)
            try: dur = float(self._dur_var.get())
            except Exception: dur = 0
            if dur > 0:
                self._cap_prog["value"] = min(100, el/dur*100)
                rem = max(0,dur-el)
                self._cap_status_var.set(
                    f"⏺ {m:02d}:{s:02d}  ·  {mb:.1f} MB  ·  {n:,} samp  ·  {rem:.0f}s left")
                if el >= dur: self._do_stop_capture()
            else:
                self._cap_status_var.set(
                    f"⏺ {m:02d}:{s:02d}  ·  {mb:.1f} MB  ·  {n:,} samp")

        self.after(200, self._poll)

    # ══════════════════════════════════════════════════════════════════════
    #  Error log
    # ══════════════════════════════════════════════════════════════════════
    _MAX_LOG = 500

    def _toggle_err_log(self):
        self._err_log_visible = not self._err_log_visible
        if self._err_log_visible:
            self._err_log_frame.pack(fill="both", side="bottom")
        else:
            self._err_log_frame.pack_forget()
        self._err_tog.config(
            text=("Hide" if self._err_log_visible else "Show Log")
            + f" ({len(self._error_log)})")

    def _clear_errors(self):
        self._err_lbl.config(text="No errors", fg=C_MUTED)
        self._error_log.clear()
        self._err_log_text.config(state="normal")
        self._err_log_text.delete("1.0","end")
        self._err_log_text.config(state="disabled")
        self._err_tog.config(
            text=("Hide" if self._err_log_visible else "Show Log")+" (0)")

    def _poll_errors(self):
        new = []
        try:
            while True: new.append(self.error_queue.get_nowait())
        except queue.Empty: pass
        if new:
            self._error_log.extend(new)
            if len(self._error_log) > self._MAX_LOG:
                self._error_log = self._error_log[-self._MAX_LOG:]
            ts,src,msg = new[-1]
            self._err_lbl.config(
                text=f"[{ts.strftime('%H:%M:%S')}] {src}: {msg}"
                + (f"  (+{len(new)-1})" if len(new)>1 else ""),
                fg=C_RED)
            lines = [f"[{t.strftime('%H:%M:%S')}] {s}: {m}"
                     for t,s,m in new[-50:]]
            self._err_log_text.config(state="normal")
            self._err_log_text.insert("end","\n".join(lines)+"\n")
            self._err_log_text.see("end")
            self._err_log_text.config(state="disabled")
            self._err_tog.config(
                text=("Hide" if self._err_log_visible else "Show Log")
                + f" ({len(self._error_log)})")
        self.after(300, self._poll_errors)

    # ── Close ─────────────────────────────────────────────────────────────
    def destroy(self):
        if self.daq:
            if self.daq.capturing: self.daq.stop_capture()
            self.daq.stop()
        super().destroy()


# ══════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    app = MotorAnalyserApp()
    app.mainloop()
 