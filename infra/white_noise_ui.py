#!/usr/bin/python3
"""Standalone White Noise & Ambient Audio UI for Whisper Typer."""
import json
import os
from pathlib import Path
import re
import sys
import tkinter as tk
from tkinter import filedialog, ttk

sys.path.insert(0, str(Path(__file__).resolve().parent))
from white_noise import CONFIG_DIR, PRESETS, WhiteNoiseClient

UI_SETTINGS_PATH = CONFIG_DIR / "white_noise_ui.json"


class WhiteNoiseUI:
    FONT = "JetBrains Mono"
    THEMES = {
        "dark": {
            "BG": "#171b22",
            "PANEL": "#222832",
            "CARD_BG": "#1e242d",
            "CARD_ACTIVE": "#2a3746",
            "FG": "#e8edf4",
            "MUTED": "#9ba8ba",
            "METADATA": "#778394",
            "BORDER": "#303741",
            "ACCENT": "#58a6ff",
            "SUCCESS": "#3fb950",
            "WARNING": "#d29922",
            "PLAY_BG": "#238636",
            "PLAY_FG": "#ffffff",
            "PAUSE_BG": "#8b949e",
            "PAUSE_FG": "#ffffff",
        },
        "light": {
            "BG": "#fdfbf5",
            "PANEL": "#f6f1e6",
            "CARD_BG": "#f0ebd8",
            "CARD_ACTIVE": "#e5dcc5",
            "FG": "#3a3326",
            "MUTED": "#756b55",
            "METADATA": "#8a7f66",
            "BORDER": "#efe9db",
            "ACCENT": "#1f6feb",
            "SUCCESS": "#2f6a3a",
            "WARNING": "#8a6300",
            "PLAY_BG": "#2f6a3a",
            "PLAY_FG": "#ffffff",
            "PAUSE_BG": "#756b55",
            "PAUSE_FG": "#ffffff",
        },
    }

    def __init__(self, root: tk.Tk):
        self.root = root
        self.client = WhiteNoiseClient(autostart=True)
        self.settings = self._load_settings()
        self.theme = self.settings.get("theme", "dark")
        if self.theme not in self.THEMES:
            self.theme = "dark"
        self._apply_theme_colors()

        self.topmost = tk.BooleanVar(value=self.settings.get("topmost", False))
        self.status = {}
        self.timer_str = tk.StringVar(value="")
        self.volume_var = tk.DoubleVar(value=20.0)
        self.preset_var = tk.StringVar(value="river_rain")
        self.state_var = tk.StringVar(value="PAUSED")
        self.closing = False

        self._init_window()
        self._build_ui()
        self._sync_status()
        self._poll_status()

    def _load_settings(self) -> dict:
        try:
            if UI_SETTINGS_PATH.is_file():
                return json.loads(UI_SETTINGS_PATH.read_text())
        except Exception:
            pass
        return {}

    def _save_settings(self):
        try:
            CONFIG_DIR.mkdir(parents=True, exist_ok=True)
            data = {
                "theme": self.theme,
                "topmost": self.topmost.get(),
                "geometry": self.root.geometry(),
            }
            tmp = UI_SETTINGS_PATH.with_suffix(".tmp")
            tmp.write_text(json.dumps(data, indent=2))
            tmp.replace(UI_SETTINGS_PATH)
        except Exception:
            pass

    def _apply_theme_colors(self):
        for k, v in self.THEMES[self.theme].items():
            setattr(self, k, v)

    def _init_window(self):
        self.root.title("White Noise · Whisper Typer")
        self.root.configure(bg=self.BG)
        self.root.minsize(440, 520)
        geometry = self.settings.get("geometry", "460x560")
        if re.fullmatch(r"\d+x\d+(?:[+-]\d+[+-]\d+)?", str(geometry)):
            self.root.geometry(geometry)
        self.root.attributes("-topmost", self.topmost.get())
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

    def _build_ui(self):
        # 1. Header Toolbar
        header = tk.Frame(self.root, bg=self.BG)
        header.pack(fill="x", padx=16, pady=(14, 6))

        title_frame = tk.Frame(header, bg=self.BG)
        title_frame.pack(side="left")

        self.title_lbl = tk.Label(
            title_frame,
            text="White Noise",
            font=(self.FONT, 13, "bold"),
            bg=self.BG,
            fg=self.FG,
        )
        self.title_lbl.pack(side="left")

        self.badge = tk.Label(
            title_frame,
            textvariable=self.state_var,
            font=(self.FONT, 8, "bold"),
            bg=self.PANEL,
            fg=self.MUTED,
            padx=7,
            pady=2,
            relief="flat",
        )
        self.badge.pack(side="left", padx=10)

        tools = tk.Frame(header, bg=self.BG)
        tools.pack(side="right")

        self.topmost_btn = tk.Button(
            tools,
            text="📌 Pin",
            font=(self.FONT, 9),
            relief="flat",
            padx=6,
            pady=2,
            takefocus=False,
            command=self._toggle_topmost,
            bg=self.PANEL,
            fg=self.FG,
        )
        self.topmost_btn.pack(side="right", padx=3)

        self.theme_btn = tk.Button(
            tools,
            text="◐",
            font=(self.FONT, 10),
            relief="flat",
            padx=6,
            pady=2,
            takefocus=False,
            command=self._toggle_theme,
            bg=self.PANEL,
            fg=self.FG,
        )
        self.theme_btn.pack(side="right", padx=3)

        # 2. Main Content Container
        self.body = tk.Frame(self.root, bg=self.BG)
        self.body.pack(fill="both", expand=True, padx=16, pady=6)

        # Section: Sound Profiles
        sound_lbl = tk.Label(
            self.body,
            text="SOUND PROFILE",
            font=(self.FONT, 8, "bold"),
            bg=self.BG,
            fg=self.METADATA,
            anchor="w",
        )
        sound_lbl.pack(fill="x", pady=(4, 6))

        self.presets_frame = tk.Frame(self.body, bg=self.BG)
        self.presets_frame.pack(fill="x", pady=(0, 10))

        preset_items = [
            ("river_rain", "🌊 River & Rain", "Original natural river & rain ASMR recording"),
            ("mellow_rain", "🌧️ Mellow Rain", "Same river & rain with sharp sounds cut out (tinnitus aid)"),
            ("brown", "🟤 Brown Noise", "Deep, warm, low-frequency rumble (focus/sleep)"),
            ("pink", "🌸 Pink Noise", "Balanced 1/f spectrum, natural rainfall"),
            ("white", "⚪ White Noise", "Full-spectrum crisp masking noise"),
            ("rain", "🌦️ Gentle Rain", "Synthetic rain shower & soothing water stream"),
            ("custom", "📁 Custom File…", "Load any local audio file (.opus/.mp3/.wav)"),
        ]

        self.preset_cards = {}
        for key, title, desc in preset_items:
            card = tk.Frame(
                self.presets_frame,
                bg=self.CARD_BG,
                padx=10,
                pady=6,
                highlightthickness=1,
                highlightbackground=self.BORDER,
                cursor="hand2",
            )
            card.pack(fill="x", pady=2)

            t_lbl = tk.Label(
                card,
                text=title,
                font=(self.FONT, 10, "bold"),
                bg=self.CARD_BG,
                fg=self.FG,
                anchor="w",
            )
            t_lbl.pack(fill="x")

            d_lbl = tk.Label(
                card,
                text=desc,
                font=(self.FONT, 8),
                bg=self.CARD_BG,
                fg=self.MUTED,
                anchor="w",
            )
            d_lbl.pack(fill="x")

            # Bind clicks
            for widget in (card, t_lbl, d_lbl):
                widget.bind("<Button-1>", lambda _, k=key: self._select_preset(k))

            self.preset_cards[key] = {"frame": card, "title": t_lbl, "desc": d_lbl}

        # Section: Volume
        vol_header = tk.Frame(self.body, bg=self.BG)
        vol_header.pack(fill="x", pady=(8, 4))

        tk.Label(
            vol_header,
            text="VOLUME",
            font=(self.FONT, 8, "bold"),
            bg=self.BG,
            fg=self.METADATA,
        ).pack(side="left")

        self.vol_pct_lbl = tk.Label(
            vol_header,
            text="50%",
            font=(self.FONT, 9, "bold"),
            bg=self.BG,
            fg=self.FG,
        )
        self.vol_pct_lbl.pack(side="right")

        vol_slider_frame = tk.Frame(self.body, bg=self.BG)
        vol_slider_frame.pack(fill="x", pady=(0, 10))

        self.mute_btn = tk.Button(
            vol_slider_frame,
            text="🔈",
            font=(self.FONT, 10),
            bg=self.PANEL,
            fg=self.FG,
            relief="flat",
            padx=6,
            pady=0,
            command=self._toggle_mute,
        )
        self.mute_btn.pack(side="left", padx=(0, 8))

        self.slider = ttk.Scale(
            vol_slider_frame,
            from_=0,
            to=100,
            orient="horizontal",
            variable=self.volume_var,
            command=self._on_volume_change,
        )
        self.slider.pack(side="left", fill="x", expand=True)

        # Section: Sleep Timer
        timer_header = tk.Frame(self.body, bg=self.BG)
        timer_header.pack(fill="x", pady=(8, 4))

        tk.Label(
            timer_header,
            text="SLEEP TIMER",
            font=(self.FONT, 8, "bold"),
            bg=self.BG,
            fg=self.METADATA,
        ).pack(side="left")

        self.timer_countdown_lbl = tk.Label(
            timer_header,
            textvariable=self.timer_str,
            font=(self.FONT, 8),
            bg=self.BG,
            fg=self.SUCCESS,
        )
        self.timer_countdown_lbl.pack(side="right")

        timer_btns_frame = tk.Frame(self.body, bg=self.BG)
        timer_btns_frame.pack(fill="x", pady=(0, 14))

        self.timer_buttons = {}
        for mins, label in [(0, "Off"), (15, "15m"), (30, "30m"), (45, "45m"), (60, "1h"), (120, "2h")]:
            btn = tk.Button(
                timer_btns_frame,
                text=label,
                font=(self.FONT, 8),
                bg=self.PANEL,
                fg=self.MUTED,
                relief="flat",
                padx=6,
                pady=2,
                takefocus=False,
                command=lambda m=mins: self._set_timer(m),
            )
            btn.pack(side="left", expand=True, fill="x", padx=2)
            self.timer_buttons[mins] = btn

        # Big Play / Pause Action Button
        self.play_button = tk.Button(
            self.root,
            text="▶ Play",
            font=(self.FONT, 12, "bold"),
            bg=self.PLAY_BG,
            fg=self.PLAY_FG,
            activebackground=self.SUCCESS,
            activeforeground="#ffffff",
            relief="flat",
            pady=10,
            takefocus=False,
            command=self._toggle_playback,
            cursor="hand2",
        )
        self.play_button.pack(fill="x", padx=16, pady=(0, 16))

    def _sync_status(self):
        try:
            self.status = self.client.status()
        except Exception:
            return

        playing = self.status.get("playing", False)
        self.state_var.set("PLAYING" if playing else "PAUSED")
        self.badge.configure(
            fg=self.SUCCESS if playing else self.MUTED,
            bg=self.PANEL,
        )

        if playing:
            self.play_button.configure(
                text="⏸ Pause",
                bg=self.PAUSE_BG,
                fg=self.PAUSE_FG,
            )
        else:
            self.play_button.configure(
                text="▶ Play",
                bg=self.PLAY_BG,
                fg=self.PLAY_FG,
            )

        # Volume
        vol = round(self.status.get("volume", 0.5) * 100, 1)
        self.volume_var.set(vol)
        pct_text = f"{int(vol)}%" if vol.is_integer() else f"{vol:.1f}%"
        self.vol_pct_lbl.configure(text=pct_text)
        self.mute_btn.configure(text="🔇" if vol == 0 else "🔈")

        # Preset
        active_preset = self.status.get("preset", "river_rain")
        self.preset_var.set(active_preset)
        for k, widgets in self.preset_cards.items():
            is_active = k == active_preset
            bg_col = self.CARD_ACTIVE if is_active else self.CARD_BG
            widgets["frame"].configure(
                bg=bg_col,
                highlightbackground=self.ACCENT if is_active else self.BORDER,
            )
            widgets["title"].configure(
                bg=bg_col,
                fg=self.ACCENT if is_active else self.FG,
            )
            widgets["desc"].configure(bg=bg_col)

        # Timer
        rem = self.status.get("timer_remaining", 0)
        mins_setting = self.status.get("timer_minutes", 0)
        if rem > 0:
            m = rem // 60
            s = rem % 60
            self.timer_str.set(f"Auto-off in {m}m {s:02d}s")
        else:
            self.timer_str.set("")

        for m, btn in self.timer_buttons.items():
            if m == mins_setting:
                btn.configure(bg=self.ACCENT, fg="#ffffff")
            else:
                btn.configure(bg=self.PANEL, fg=self.MUTED)

    def _poll_status(self):
        if self.closing:
            return
        self._sync_status()
        self.root.after(800, self._poll_status)

    def _toggle_playback(self):
        try:
            self.client.toggle()
            self._sync_status()
        except Exception:
            pass

    def _select_preset(self, key: str):
        if key == "custom":
            filetypes = [
                ("Audio Files", "*.opus *.ogg *.mp3 *.wav *.flac *.m4a *.aac"),
                ("All Files", "*.*"),
            ]
            path = filedialog.askopenfilename(
                title="Select Ambient Audio File",
                filetypes=filetypes,
            )
            if path:
                self.client.set_custom_file(path)
                self.preset_cards["custom"]["desc"].configure(text=Path(path).name)
        else:
            self.client.set_preset(key)
        self._sync_status()

    def _on_volume_change(self, val):
        vol_pct = round(float(val), 1)
        pct_text = f"{int(vol_pct)}%" if vol_pct.is_integer() else f"{vol_pct:.1f}%"
        self.vol_pct_lbl.configure(text=pct_text)
        self.mute_btn.configure(text="🔇" if vol_pct == 0 else "🔈")
        try:
            self.client.set_volume(round(vol_pct / 100.0, 3))
        except Exception:
            pass

    def _toggle_mute(self):
        current_vol = self.volume_var.get()
        if current_vol > 0:
            self._pre_mute_vol = current_vol
            self.volume_var.set(0)
            self._on_volume_change(0)
        else:
            restore = getattr(self, "_pre_mute_vol", 50.0)
            self.volume_var.set(restore)
            self._on_volume_change(restore)

    def _set_timer(self, minutes: int):
        try:
            self.client.set_timer(minutes)
            self._sync_status()
        except Exception:
            pass

    def _toggle_topmost(self):
        new_val = not self.topmost.get()
        self.topmost.set(new_val)
        self.root.attributes("-topmost", new_val)
        self.topmost_btn.configure(text="📌 Pinned" if new_val else "📌 Pin")
        self._save_settings()

    def _toggle_theme(self):
        self.theme = "light" if self.theme == "dark" else "dark"
        self._apply_theme_colors()
        self.root.configure(bg=self.BG)
        self.body.configure(bg=self.BG)
        self.title_lbl.configure(bg=self.BG, fg=self.FG)
        self.topmost_btn.configure(bg=self.PANEL, fg=self.FG)
        self.theme_btn.configure(bg=self.PANEL, fg=self.FG)
        self._sync_status()
        self._save_settings()

    def _on_close(self):
        self.closing = True
        self._save_settings()
        self.root.destroy()


def launch_ui():
    root = tk.Tk(className="WhiteNoise")
    app = WhiteNoiseUI(root)
    root.mainloop()


if __name__ == "__main__":
    launch_ui()
