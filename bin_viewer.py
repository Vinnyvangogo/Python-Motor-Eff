"""
Motor Capture  —  Binary Waveform Viewer
=========================================
Companion tool to motor_efficiency.py.

Loads the raw .bin capture files written by the Motor Efficiency
script and lets you inspect the waveforms before running post-processing.

Features
--------
  • Load .bin file and its _meta.json sidecar automatically
  • Channel selector — any combination of V and I channels
  • Time window scrubber — navigate through long files
  • Zoom in / out on the time axis
  • Per-channel amplitude scaling (auto or manual)
  • Statistics panel — Vrms, Irms, peak, crest factor, frequency estimate
  • Calibration applied on-the-fly using the shared cdaq_calibration.json
  • Export the current view window as CSV

File format expected
--------------------
  Raw float32 binary, shape (N_samples, 12)
    Columns 0–5  : Voltage CH0–5  (raw sensor volts, pre-calibration)
    Columns 6–11 : Current CH0–5  (raw sensor amps,  pre-calibration)
  Optional sidecar: <filename>_meta.json (auto-written by motor_efficiency.py)

Dependencies: pip install numpy matplotlib
"""

import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import threading
import json
import os
import sys
from datetime import datetime
from typing import Optional

import numpy as np

try:
    import matplotlib
    matplotlib.use("TkAgg")
    from matplotlib.figure import Figure
    from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
    from matplotlib.ticker import AutoLocator, FuncFormatter
    HAS_MPL = True
except ImportError:
    HAS_MPL = False

# ── Channel config (must match motor_efficiency.py) ───────────────────────
N_CH     = 6
N_RAW    = N_CH * 2
MOD_V    = 2
MOD_I    = 3

V_NAMES  = [f"V_PH{s}" for s in ["A_IN","B_IN","C_IN","A_OUT","B_OUT","C_OUT"]]
I_NAMES  = [f"I_PH{s}" for s in ["A_IN","B_IN","C_IN","A_OUT","B_OUT","C_OUT"]]

ALL_NAMES   = V_NAMES + I_NAMES
ALL_UNITS   = ["V"] * N_CH + ["A"] * N_CH

# Phase groupings — CH0-2 are input side, CH3-5 are output side
INPUT_PHASES  = [0, 1, 2]
OUTPUT_PHASES = [3, 4, 5]

# Colours — two contrast palettes for V and I channels
V_COLORS = ["#58a6ff","#3fb950","#f0883e","#79c0ff","#56d364","#ffa657"]
I_COLORS = ["#ff7b72","#d2a8ff","#ffa657","#ff9a9a","#c9a0ff","#ffbf7b"]
ALL_COLORS = V_COLORS + I_COLORS

# ── Shared constants ─────────────────────────────────────────────────────
C_BG    = "#0d1117"
C_PANEL = "#161b22"
C_BORD  = "#30363d"
C_ACC   = "#00b4d8"
C_GREEN = "#39d353"
C_RED   = "#f85149"
C_YEL   = "#e3b341"
C_TEXT  = "#e6edf3"
C_MUTED = "#8b949e"
C_INPUT = "#21262d"

FONT_MONO  = ("Courier New", 9)
FONT_MONOS = ("Courier New", 8)
FONT_SMALL = ("Segoe UI", 9)
FONT_TINY  = ("Segoe UI", 8)
FONT_BOLD  = ("Segoe UI", 10, "bold")
FONT_HEAD  = ("Segoe UI", 12, "bold")
FONT_TITLE = ("Segoe UI", 14, "bold")


# ── Find cdaq_calibration.json ────────────────────────────────────────────
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
    for p in candidates:
        if os.path.exists(p): return p
    return candidates[0] if candidates else "cdaq_calibration.json"

CAL_FILE = _find_cal_file()


# ── Statistics helpers ────────────────────────────────────────────────────
def channel_stats(samples: np.ndarray, fs: float) -> dict:
    """Compute statistics for a single-channel 1D float array."""
    if len(samples) < 2:
        return {}
    rms      = float(np.sqrt(np.nanmean(samples**2)))
    peak     = float(np.nanmax(np.abs(samples)))
    mean_dc  = float(np.nanmean(samples))
    crest    = peak / rms if rms > 1e-9 else float("nan")
    # Frequency estimate via zero-crossing count
    zc = np.where(np.diff(np.sign(samples - mean_dc)))[0]
    freq_est = (len(zc) / 2) / (len(samples) / fs) if len(zc) > 2 else float("nan")
    return dict(rms=rms, peak=peak, mean=mean_dc,
                crest=crest, freq=freq_est)


# ══════════════════════════════════════════════════════════════════════════
#  Main viewer application
# ══════════════════════════════════════════════════════════════════════════
class BinViewer(tk.Tk):

    def __init__(self):
        super().__init__()
        self.title("Motor Capture — Waveform Viewer")
        self.configure(bg=C_BG)
        self.geometry("1440x920")
        self.minsize(1100, 720)

        # File state
        self._raw: Optional[np.ndarray] = None   # full file, float32 (N,12)
        self._fs:   float = 200_000.0
        self._ac_freq: float = 400.0
        self._n_samples: int = 0
        self._file_path: Optional[str] = None
        self._meta: dict = {}

        # View state
        self._win_start = 0       # start sample of current view window
        self._win_len   = 0       # samples in view window
        self._zoom_level = 1.0

        # Calibration
        self._v_cal = [(1.0, 0.0)] * N_CH
        self._i_cal = [(1.0, 0.0)] * N_CH
        self._apply_cal = True

        # Channel visibility
        self._ch_visible = [True] * N_RAW   # 12 channels

        self._load_calibration()
        self._build_style()
        self._build_ui()

    # ── Calibration ───────────────────────────────────────────────────────
    def _load_calibration(self):
        if not os.path.exists(CAL_FILE):
            return
        try:
            with open(CAL_FILE) as f:
                data = json.load(f)
            for rec in data.get("NI_9320_modules_2_to_6", []):
                mod = rec.get("module"); ch = rec.get("channel")
                if ch is None or ch >= N_CH: continue
                s = float(rec.get("scale", 1.0))
                o = float(rec.get("offset", 0.0))
                nm = rec.get("name", "")
                if mod == MOD_V:
                    self._v_cal[ch] = (s, o)
                    if nm: V_NAMES[ch] = nm; ALL_NAMES[ch] = nm
                elif mod == MOD_I:
                    self._i_cal[ch] = (s, o)
                    if nm: I_NAMES[ch] = nm; ALL_NAMES[N_CH+ch] = nm
        except Exception as e:
            print(f"[Viewer] Cal load failed: {e}")

    # ── Style ─────────────────────────────────────────────────────────────
    def _build_style(self):
        s = ttk.Style(self)
        s.theme_use("clam")
        s.configure(".", background=C_BG, foreground=C_TEXT,
                    fieldbackground=C_INPUT, troughcolor=C_BORD)
        s.configure("TLabelframe", background=C_BG, foreground=C_ACC,
                    relief="flat", borderwidth=1)
        s.configure("TLabelframe.Label", background=C_BG,
                    foreground=C_ACC, font=FONT_BOLD)
        for nm, fg in [("G.TButton", C_GREEN), ("R.TButton", C_RED),
                        ("A.TButton", C_ACC),   ("Y.TButton", C_YEL)]:
            s.configure(nm, background=C_PANEL, foreground=fg,
                        relief="flat", padding=[8,4])
            s.map(nm, background=[("active", C_BORD)])
        s.configure("TCheckbutton", background=C_BG, foreground=C_TEXT,
                    font=FONT_TINY)
        s.configure("Dark.TCheckbutton", background=C_PANEL,
                    foreground=C_TEXT, font=FONT_TINY)
        s.configure("TScale", background=C_BG, troughcolor=C_BORD)
        s.configure("TScrollbar", background=C_BORD, troughcolor=C_PANEL)
        s.configure("TEntry", fieldbackground=C_INPUT, foreground=C_TEXT)

    # ── UI ────────────────────────────────────────────────────────────────
    def _build_ui(self):
        # ── Top toolbar ──────────────────────────────────────────────────
        top = tk.Frame(self, bg=C_PANEL, pady=6, padx=12)
        top.pack(fill="x")

        tk.Label(top, text="Waveform Viewer", font=FONT_TITLE,
                 bg=C_PANEL, fg=C_ACC).pack(side="left")

        ttk.Button(top, text="📂 Open .bin",
                   style="G.TButton",
                   command=self._open_file).pack(side="left", padx=12)

        self._file_lbl = tk.Label(top, text="No file loaded",
                                   font=FONT_SMALL, bg=C_PANEL, fg=C_MUTED)
        self._file_lbl.pack(side="left")

        # Cal toggle
        self._cal_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(top, text="Apply calibration",
                        variable=self._cal_var,
                        command=self._on_cal_toggle
                        ).pack(side="left", padx=20)

        # Export current view
        ttk.Button(top, text="💾 Export View CSV",
                   style="Y.TButton",
                   command=self._export_view).pack(side="right", padx=6)

        ttk.Button(top, text="🔁 Reset View",
                   style="A.TButton",
                   command=self._reset_view).pack(side="right", padx=4)

        # ── Main area: sidebar + plot ─────────────────────────────────────
        main = tk.Frame(self, bg=C_BG)
        main.pack(fill="both", expand=True)

        # ── Left sidebar ──────────────────────────────────────────────────
        sidebar = tk.Frame(main, bg=C_PANEL, width=220)
        sidebar.pack(side="left", fill="y", padx=(4,0), pady=4)
        sidebar.pack_propagate(False)

        # File info
        info_f = ttk.LabelFrame(sidebar, text=" File Info ", padding=6)
        info_f.pack(fill="x", padx=6, pady=4)
        self._info_vars: dict[str, tk.StringVar] = {}
        for key, label in [("samples","Samples"),("duration","Duration"),
                            ("fs","Sample rate"),("ac_freq","Signal freq"),
                            ("size","File size")]:
            row = tk.Frame(info_f, bg=C_PANEL)
            row.pack(fill="x", pady=1)
            tk.Label(row, text=f"{label}:", font=FONT_TINY,
                     bg=C_PANEL, fg=C_MUTED, width=12,
                     anchor="w").pack(side="left")
            v = tk.StringVar(value="---")
            self._info_vars[key] = v
            tk.Label(row, textvariable=v, font=FONT_MONOS,
                     bg=C_PANEL, fg=C_TEXT, anchor="w").pack(side="left")

        # Channel selector
        ch_f = ttk.LabelFrame(sidebar, text=" Channels ", padding=6)
        ch_f.pack(fill="x", padx=6, pady=4)

        # Enable all / voltage / current quick buttons
        qb = tk.Frame(ch_f, bg=C_PANEL)
        qb.pack(fill="x", pady=(0,4))
        for label, start, end, col in [
            ("V",  0,  6, C_ACC),
            ("I",  6, 12, C_RED),
            ("All",0, 12, C_MUTED),
        ]:
            ttk.Button(qb, text=label,
                       command=lambda s=start,e=end: self._set_channels(s,e),
                       style="A.TButton"
                       ).pack(side="left", padx=1)

        ttk.Button(qb, text="None",
                   command=lambda: self._set_channels(-1,-1),
                   style="R.TButton").pack(side="left", padx=1)

        self._ch_vars: list[tk.BooleanVar] = []
        for i in range(N_RAW):
            v = tk.BooleanVar(value=True)
            self._ch_vars.append(v)
            row = tk.Frame(ch_f, bg=C_PANEL)
            row.pack(fill="x", pady=1)
            # Colour swatch
            tk.Canvas(row, width=10, height=10, bg=ALL_COLORS[i],
                      highlightthickness=0).pack(side="left", padx=(0,4))
            ttk.Checkbutton(row, variable=v,
                            text=f"{'V' if i<6 else 'I'}{i%6}  {ALL_NAMES[i]}",
                            style="Dark.TCheckbutton",
                            command=self._on_ch_change).pack(side="left")

        # Y-scale mode
        ys_f = ttk.LabelFrame(sidebar, text=" Y Scale ", padding=6)
        ys_f.pack(fill="x", padx=6, pady=4)
        self._yscale_var = tk.StringVar(value="auto")
        for val, label in [("auto","Auto"), ("shared","Shared"),
                            ("manual","Manual")]:
            tk.Radiobutton(ys_f, text=label, variable=self._yscale_var,
                           value=val, bg=C_PANEL, fg=C_TEXT,
                           selectcolor=C_ACC, activebackground=C_PANEL,
                           font=FONT_TINY, command=self._redraw
                           ).pack(anchor="w")

        self._ymin_var = tk.StringVar(value="-400")
        self._ymax_var = tk.StringVar(value="+400")
        for label, var in [("Min:", self._ymin_var),
                            ("Max:", self._ymax_var)]:
            row = tk.Frame(ys_f, bg=C_PANEL)
            row.pack(fill="x", pady=1)
            tk.Label(row, text=label, font=FONT_TINY,
                     bg=C_PANEL, fg=C_MUTED, width=4).pack(side="left")
            tk.Entry(row, textvariable=var, width=8,
                     bg=C_INPUT, fg=C_TEXT, font=FONT_MONO,
                     insertbackground=C_TEXT, relief="flat"
                     ).pack(side="left")

        ttk.Button(ys_f, text="Apply", style="A.TButton",
                   command=self._redraw).pack(pady=2)

        # Stats panel — populated on draw
        stats_f = ttk.LabelFrame(sidebar, text=" Statistics (view window) ",
                                  padding=6)
        stats_f.pack(fill="both", expand=True, padx=6, pady=4)

        self._stats_text = tk.Text(stats_f, bg=C_PANEL, fg=C_TEXT,
                                    font=FONT_MONOS, relief="flat",
                                    state="disabled", wrap="none",
                                    width=32)
        sb = ttk.Scrollbar(stats_f, orient="vertical",
                            command=self._stats_text.yview)
        self._stats_text.configure(yscrollcommand=sb.set)
        self._stats_text.pack(side="left", fill="both", expand=True)
        sb.pack(side="right", fill="y")

        # ── Right: plot + nav ─────────────────────────────────────────────
        plot_area = tk.Frame(main, bg=C_BG)
        plot_area.pack(side="left", fill="both", expand=True, padx=4, pady=4)

        if not HAS_MPL:
            tk.Label(plot_area,
                     text="matplotlib not installed.\n"
                          "pip install matplotlib",
                     font=FONT_HEAD, bg=C_BG, fg=C_RED).pack(expand=True)
            return

        # Navigation controls
        nav = tk.Frame(plot_area, bg=C_BG)
        nav.pack(fill="x", pady=(0,4))

        tk.Label(nav, text="Window:", font=FONT_SMALL,
                 bg=C_BG, fg=C_MUTED).pack(side="left")

        self._win_ms_var = tk.StringVar(value="10.0")
        tk.Entry(nav, textvariable=self._win_ms_var, width=7,
                 bg=C_INPUT, fg=C_TEXT, font=FONT_MONO,
                 insertbackground=C_TEXT, relief="flat").pack(side="left", padx=4)
        tk.Label(nav, text="ms", font=FONT_SMALL,
                 bg=C_BG, fg=C_MUTED).pack(side="left")

        ttk.Button(nav, text="◀◀", style="A.TButton",
                   command=lambda: self._shift(-10)).pack(side="left", padx=2)
        ttk.Button(nav, text="◀",  style="A.TButton",
                   command=lambda: self._shift(-1)).pack(side="left", padx=1)
        ttk.Button(nav, text="▶",  style="A.TButton",
                   command=lambda: self._shift(1)).pack(side="left", padx=1)
        ttk.Button(nav, text="▶▶", style="A.TButton",
                   command=lambda: self._shift(10)).pack(side="left", padx=2)

        ttk.Button(nav, text="🔍+", style="G.TButton",
                   command=lambda: self._zoom(0.5)).pack(side="left", padx=4)
        ttk.Button(nav, text="🔍−", style="R.TButton",
                   command=lambda: self._zoom(2.0)).pack(side="left", padx=1)

        self._pos_lbl = tk.Label(nav, text="", font=FONT_MONO,
                                  bg=C_BG, fg=C_MUTED)
        self._pos_lbl.pack(side="right", padx=8)

        # Scrollbar for file navigation
        self._nav_scroll = ttk.Scrollbar(plot_area, orient="horizontal",
                                          command=self._on_scroll)
        self._nav_scroll.pack(fill="x", side="bottom")

        # Matplotlib figure
        self._fig = Figure(figsize=(10, 6), facecolor=C_BG)
        self._fig.subplots_adjust(left=0.07, right=0.97,
                                   top=0.96, bottom=0.08,
                                   hspace=0.08)
        self._canvas = FigureCanvasTkAgg(self._fig, master=plot_area)
        self._canvas.get_tk_widget().pack(fill="both", expand=True)

        # Matplotlib toolbar (zoom/pan/save)
        toolbar_frame = tk.Frame(plot_area, bg=C_BG)
        toolbar_frame.pack(fill="x")
        toolbar = NavigationToolbar2Tk(self._canvas, toolbar_frame)
        toolbar.configure(background=C_BG)
        toolbar.update()

        # Mouse cursor position display
        self._canvas.mpl_connect("motion_notify_event", self._on_mouse_move)

        # Keyboard shortcuts
        self.bind("<Left>",  lambda e: self._shift(-1))
        self.bind("<Right>", lambda e: self._shift(1))
        self.bind("<Up>",    lambda e: self._zoom(0.5))
        self.bind("<Down>",  lambda e: self._zoom(2.0))

    # ── File loading ──────────────────────────────────────────────────────
    def _open_file(self):
        path = filedialog.askopenfilename(
            filetypes=[("Raw binary", "*.bin"), ("All", "*.*")],
            title="Open Motor Capture File")
        if not path:
            return
        self._load_file(path)

    def _load_file(self, path: str):
        def worker():
            try:
                raw = np.fromfile(path, dtype=np.float32)
                if raw.size % N_RAW != 0:
                    self.after(0, lambda: messagebox.showerror(
                        "Format error",
                        f"File size {raw.size} is not divisible by {N_RAW}. "
                        f"Expected float32 with {N_RAW} columns."))
                    return
                raw = raw.reshape(-1, N_RAW)
                self.after(0, lambda: self._on_file_loaded(path, raw))
            except Exception as e:
                self.after(0, lambda: messagebox.showerror(
                    "Load Error", str(e)))
        self._file_lbl.config(text="Loading...", fg=C_YEL)
        threading.Thread(target=worker, daemon=True).start()

    def _on_file_loaded(self, path: str, raw: np.ndarray):
        self._raw        = raw
        self._file_path  = path
        self._n_samples  = raw.shape[0]

        # Load sidecar metadata
        meta_path = path.replace(".bin", "_meta.json")
        self._meta = {}
        if os.path.exists(meta_path):
            try:
                with open(meta_path) as f:
                    self._meta = json.load(f)
                self._fs      = float(self._meta.get("hw_rate_hz",  200_000))
                self._ac_freq = float(self._meta.get("ac_freq_hz",  400.0))
                # Update channel names from sidecar if present
                col_names = self._meta.get("column_names", [])
                for i, nm in enumerate(col_names[:N_RAW]):
                    ALL_NAMES[i] = nm
                    if i < N_CH:   V_NAMES[i] = nm
                    else:          I_NAMES[i-N_CH] = nm
            except Exception as e:
                print(f"[Viewer] Meta load failed: {e}")

        # Update info panel
        dur  = self._n_samples / self._fs
        size = os.path.getsize(path)
        self._info_vars["samples"].set(f"{self._n_samples:,}")
        self._info_vars["duration"].set(f"{dur:.3f} s")
        self._info_vars["fs"].set(f"{self._fs:,.0f} S/s")
        self._info_vars["ac_freq"].set(f"{self._ac_freq:.0f} Hz")
        self._info_vars["size"].set(f"{size/1e6:.2f} MB")

        fname = os.path.basename(path)
        self._file_lbl.config(
            text=f"{fname}  ({self._n_samples:,} samples  ·  {dur:.2f}s)",
            fg=C_GREEN)

        # Default view: first 5 cycles of the AC signal
        win_samples = max(100, int(self._fs / self._ac_freq * 5))
        self._win_start = 0
        self._win_len   = min(win_samples, self._n_samples)
        self._win_ms_var.set(f"{self._win_len / self._fs * 1000:.2f}")

        self._update_scrollbar()
        self._redraw()

    # ── Calibration application ───────────────────────────────────────────
    def _calibrated_view(self, raw_win: np.ndarray) -> np.ndarray:
        """Return a (N,12) array with calibration applied if enabled."""
        if not self._cal_var.get():
            return raw_win.astype(np.float64)
        out = raw_win.astype(np.float64).copy()
        for ch in range(N_CH):
            s, o = self._v_cal[ch]
            out[:, ch]       = raw_win[:, ch]       * s + o
            s, o = self._i_cal[ch]
            out[:, N_CH+ch]  = raw_win[:, N_CH+ch]  * s + o
        return out

    # ── Navigation ────────────────────────────────────────────────────────
    def _current_win_samples(self) -> int:
        try:
            ms = float(self._win_ms_var.get())
            return max(10, int(ms * self._fs / 1000))
        except ValueError:
            return self._win_len

    def _shift(self, steps: int):
        """Shift view by ±steps × half-window."""
        if self._raw is None:
            return
        step = max(1, self._win_len // 2)
        self._win_start = int(np.clip(
            self._win_start + steps * step, 0,
            max(0, self._n_samples - self._win_len)))
        self._update_scrollbar()
        self._redraw()

    def _zoom(self, factor: float):
        """Multiply window length by factor, keeping centre fixed."""
        if self._raw is None:
            return
        centre      = self._win_start + self._win_len // 2
        new_len     = int(np.clip(self._win_len * factor, 10, self._n_samples))
        new_start   = int(np.clip(centre - new_len // 2, 0,
                                   self._n_samples - new_len))
        self._win_len   = new_len
        self._win_start = new_start
        self._win_ms_var.set(f"{new_len / self._fs * 1000:.2f}")
        self._update_scrollbar()
        self._redraw()

    def _reset_view(self):
        if self._raw is None:
            return
        win_samples = max(100, int(self._fs / self._ac_freq * 5))
        self._win_start = 0
        self._win_len   = min(win_samples, self._n_samples)
        self._win_ms_var.set(f"{self._win_len / self._fs * 1000:.2f}")
        self._update_scrollbar()
        self._redraw()

    def _on_scroll(self, *args):
        if self._raw is None:
            return
        action = args[0]
        if action == "moveto":
            frac = float(args[1])
            self._win_start = int(np.clip(
                frac * self._n_samples, 0,
                max(0, self._n_samples - self._win_len)))
        elif action == "scroll":
            self._shift(int(args[1]))
        self._update_scrollbar()
        self._redraw()

    def _update_scrollbar(self):
        if self._n_samples == 0:
            self._nav_scroll.set(0, 1)
            return
        lo = self._win_start / self._n_samples
        hi = min(1.0, (self._win_start + self._win_len) / self._n_samples)
        self._nav_scroll.set(lo, hi)

    def _on_ch_change(self):
        for i, v in enumerate(self._ch_vars):
            self._ch_visible[i] = v.get()
        self._redraw()

    def _set_channels(self, start: int, end: int):
        for i, v in enumerate(self._ch_vars):
            v.set(start <= i < end)
            self._ch_visible[i] = v.get()
        self._redraw()

    def _on_cal_toggle(self):
        self._apply_cal = self._cal_var.get()
        self._redraw()

    # ── Drawing ───────────────────────────────────────────────────────────
    def _redraw(self):
        if self._raw is None or not HAS_MPL:
            return

        # Update window length from the entry field
        new_len = self._current_win_samples()
        if new_len != self._win_len:
            self._win_len   = new_len
            self._win_start = int(np.clip(self._win_start, 0,
                                           max(0, self._n_samples - self._win_len)))

        # Slice the window
        s = self._win_start
        e = min(s + self._win_len, self._n_samples)
        raw_win = self._raw[s:e]

        cal_win = self._calibrated_view(raw_win)

        # Determine active channels and subplot count
        active = [i for i in range(N_RAW) if self._ch_visible[i]]
        if not active:
            self._fig.clf()
            ax = self._fig.add_subplot(111)
            ax.set_facecolor(C_BG)
            ax.text(0.5, 0.5, "No channels selected",
                    ha="center", va="center",
                    color=C_MUTED, fontsize=12, transform=ax.transAxes)
            self._canvas.draw_idle()
            return

        # Group V and I channels to decide subplot layout
        v_active = [i for i in active if i < N_CH]
        i_active = [i for i in active if i >= N_CH]
        n_plots  = (1 if v_active else 0) + (1 if i_active else 0)

        self._fig.clf()
        t_ms = np.arange(len(raw_win)) / self._fs * 1000   # time in ms

        axes = []
        plot_idx = 1
        y_scale = self._yscale_var.get()

        def _make_ax(title, channels, units):
            nonlocal plot_idx
            ax = self._fig.add_subplot(n_plots, 1, plot_idx,
                                        sharex=axes[0] if axes else None)
            ax.set_facecolor(C_BG)
            ax.tick_params(colors=C_MUTED, labelsize=8)
            for spine in ax.spines.values():
                spine.set_edgecolor(C_BORD)
            ax.grid(True, color=C_BORD, linewidth=0.5, alpha=0.6)
            ax.set_ylabel(units, color=C_MUTED, fontsize=9)
            ax.set_title(title, color=C_TEXT, fontsize=9, pad=4)

            for ch in channels:
                data = cal_win[:, ch]
                lw = 0.8 if len(data) > 5000 else 1.0
                ax.plot(t_ms, data, color=ALL_COLORS[ch],
                        linewidth=lw, label=ALL_NAMES[ch], alpha=0.9)

            # Y-scale
            if y_scale == "shared":
                all_data = np.concatenate([cal_win[:, ch] for ch in channels])
                ymax = np.nanmax(np.abs(all_data)) * 1.1 or 1
                ax.set_ylim(-ymax, ymax)
            elif y_scale == "manual":
                try:
                    ax.set_ylim(float(self._ymin_var.get()),
                                float(self._ymax_var.get()))
                except ValueError:
                    pass

            ax.legend(loc="upper right", fontsize=7,
                      facecolor=C_PANEL, edgecolor=C_BORD,
                      labelcolor=C_TEXT, framealpha=0.85,
                      ncol=min(3, len(channels)))
            axes.append(ax)
            plot_idx += 1
            return ax

        if v_active:
            _make_ax("Voltage Channels", v_active,
                     "V (cal)" if self._cal_var.get() else "V (raw)")
        if i_active:
            _make_ax("Current Channels", i_active,
                     "A (cal)" if self._cal_var.get() else "A (raw)")

        # X-axis label only on bottom subplot
        if axes:
            axes[-1].set_xlabel("Time (ms)", color=C_MUTED, fontsize=9)
            t_start_s = s / self._fs
            t_end_s   = e / self._fs
            axes[-1].xaxis.set_major_formatter(
                FuncFormatter(lambda x, _: f"{x:.3f}"))

        self._fig.patch.set_facecolor(C_BG)

        # Position label
        self._pos_lbl.config(
            text=f"t = {s/self._fs*1000:.3f} – {e/self._fs*1000:.3f} ms  "
                 f"({self._win_len:,} samples)")

        self._canvas.draw_idle()
        self._update_stats(cal_win, active)

    def _update_stats(self, cal_win: np.ndarray, active: list[int]):
        """Compute and display per-channel stats plus power/PF/efficiency
        for the current view window.

        Power calculations require at least one V and one I channel to be
        visible.  V and I channels are matched by phase index (V0↔I0 … V5↔I5).
        """
        lines = []

        # ── Per-channel table ─────────────────────────────────────────────
        lines.append("── Channel Stats ─────────────────────────────")
        lines.append(f"{'Ch':<4} {'Name':<18} {'RMS':>8} {'Peak':>8} "
                     f"{'Crest':>6} {'Hz':>7}")
        lines.append("─" * 55)
        for ch in active:
            st = channel_stats(cal_win[:, ch], self._fs)
            if not st:
                continue
            units = ALL_UNITS[ch]
            rms   = f"{st['rms']:.3f}{units}"
            peak  = f"{st['peak']:.3f}{units}"
            crest = f"{st['crest']:.2f}" if np.isfinite(st['crest']) else "---"
            freq  = f"{st['freq']:.0f}"  if np.isfinite(st['freq'])  else "---"
            name  = ALL_NAMES[ch][:16]
            label = f"{'V' if ch<N_CH else 'I'}{ch%N_CH}"
            lines.append(f"{label:<4} {name:<18} {rms:>8} {peak:>8} "
                         f"{crest:>6} {freq:>7}")

        # ── Power, PF, Efficiency ─────────────────────────────────────────
        lines.append("")
        lines.append("── Power Analysis ────────────────────────────")

        # Determine which phase indices have BOTH V and I visible
        v_active_phases = {ch     for ch in active if ch < N_CH}
        i_active_phases = {ch-N_CH for ch in active if ch >= N_CH}
        paired_phases   = sorted(v_active_phases & i_active_phases)

        if not paired_phases:
            lines.append("  (Need matching V and I channels visible)")
            lines.append("  e.g. V0 + I0 for phase A-IN power")
        else:
            # Align to complete AC cycles to make mean(v*i) exact
            spc   = self._fs / max(1.0, self._ac_freq)
            n_cyc = max(1, int(cal_win.shape[0] // spc))
            n_use = int(round(n_cyc * spc))
            w     = cal_win[:n_use]

            if n_use < 2:
                lines.append("  (Window too short for power calc)")
            else:
                lines.append(f"  Cycles in view: {n_cyc}  "
                             f"({n_use} samples used)")
                lines.append("")
                lines.append(f"{'Ph':<5} {'Signal':<16} {'P(W)':>9} "
                             f"{'S(VA)':>9} {'Q(VAR)':>9} {'PF':>6}")
                lines.append("─" * 58)

                p_in = p_out = s_in = s_out = 0.0

                for ph in paired_phases:
                    v_col = ph            # column in cal_win
                    i_col = N_CH + ph

                    v = w[:, v_col]
                    i = w[:, i_col]

                    vrms = float(np.sqrt(np.nanmean(v**2)))
                    irms = float(np.sqrt(np.nanmean(i**2)))
                    P    = float(np.nanmean(v * i))     # real power W
                    S    = vrms * irms                   # apparent power VA
                    Q    = float(np.sqrt(max(0.0, S**2 - P**2)))
                    pf   = P / S if S > 1e-6 else 0.0

                    side = "IN " if ph in INPUT_PHASES else "OUT"
                    name = V_NAMES[ph][:14]
                    ph_label = f"{side}{ph%3+1}"

                    lines.append(f"{ph_label:<5} {name:<16} "
                                 f"{P:>9.2f} {S:>9.2f} "
                                 f"{Q:>9.2f} {pf:>6.4f}")

                    if ph in INPUT_PHASES:
                        p_in  += P;  s_in  += S
                    else:
                        p_out += P;  s_out += S

                # Three-phase totals
                lines.append("─" * 58)

                def _pf(p, s):
                    return f"{p/s:.4f}" if s > 1e-6 else "---"

                lines.append(f"{'3Φ IN':<5} {'Total Input':<16} "
                             f"{p_in:>9.2f} {s_in:>9.2f} "
                             f"{'---':>9} {_pf(p_in,s_in):>6}")
                lines.append(f"{'3Φ OUT':<5} {'Total Output':<16} "
                             f"{p_out:>9.2f} {s_out:>9.2f} "
                             f"{'---':>9} {_pf(p_out,s_out):>6}")

                lines.append("")
                losses = p_in - p_out
                if p_in > 1e-3:
                    eff = p_out / p_in * 100
                    eff_bar = "█" * int(eff/5) + "░" * (20 - int(eff/5))
                    lines.append(f"  Efficiency  : {eff:.3f} %")
                    lines.append(f"  [{eff_bar}]")
                    lines.append(f"  Losses      : {losses:.3f} W")
                    lines.append(f"  PF (input)  : {_pf(p_in, s_in)}")
                    lines.append(f"  PF (output) : {_pf(p_out, s_out)}")
                else:
                    lines.append("  (Input power too low for efficiency calc)")

        self._stats_text.config(state="normal")
        self._stats_text.delete("1.0", "end")
        self._stats_text.insert("end", "\n".join(lines))
        self._stats_text.config(state="disabled")

    # ── Mouse position ────────────────────────────────────────────────────
    def _on_mouse_move(self, event):
        if event.inaxes and event.xdata is not None:
            t_ms  = event.xdata
            t_abs = self._win_start / self._fs * 1000 + t_ms
            self._pos_lbl.config(
                text=f"cursor: {t_ms:.3f} ms   abs: {t_abs:.3f} ms")

    # ── Export ────────────────────────────────────────────────────────────
    def _export_view(self):
        if self._raw is None:
            messagebox.showwarning("No file", "Open a .bin file first.")
            return
        try:
            d = os.path.dirname(os.path.abspath(sys.argv[0]))
        except Exception:
            d = os.getcwd()
        ts   = datetime.now().strftime("%Y%m%d_%H%M%S")
        path = filedialog.asksaveasfilename(
            defaultextension=".csv",
            filetypes=[("CSV", "*.csv"), ("All", "*.*")],
            title="Export View Window as CSV",
            initialdir=d,
            initialfile=f"view_export_{ts}.csv")
        if not path:
            return

        s = self._win_start
        e = min(s + self._win_len, self._n_samples)
        raw_win = self._raw[s:e]
        cal_win = self._calibrated_view(raw_win)

        import csv
        hdr = (["Sample", "Time_ms"]
               + [f"V{ch}_{V_NAMES[ch]}_{'cal' if self._cal_var.get() else 'raw'}"
                  for ch in range(N_CH)]
               + [f"I{ch}_{I_NAMES[ch]}_{'cal' if self._cal_var.get() else 'raw'}"
                  for ch in range(N_CH)])

        with open(path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(hdr)
            for n, row in enumerate(cal_win):
                t_ms = (s + n) / self._fs * 1000
                w.writerow([s+n, f"{t_ms:.6f}"]
                           + [f"{v:.6f}" for v in row])

        messagebox.showinfo("Exported",
                            f"View window exported to:\n{os.path.basename(path)}\n"
                            f"({len(cal_win):,} samples)")


# ══════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    if not HAS_MPL:
        print("ERROR: matplotlib is required.")
        print("Install it with:  pip install matplotlib")
        sys.exit(1)
    app = BinViewer()
    app.mainloop()
