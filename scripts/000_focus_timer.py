#!/usr/bin/env python3
"""
Simple Focus Timer - macOS Dock App
====================================
Always-on-top timer for tracking focused work sessions.

USAGE:
------
  python scripts/000_focus_timer.py

Features:
- Visible in macOS dock
- Audio alert when time's up
- Pause/resume/reset
- Logs sessions to CSV for billing
"""

import subprocess
import tkinter as tk
from datetime import datetime
from pathlib import Path


class FocusTimer:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Focus Timer")
        self.root.geometry("300x200")

        # Always on top
        self.root.attributes("-topmost", False)

        # Timer state
        self.running = False
        self.elapsed = 0  # seconds
        self.target = 60 * 60  # 1 hour default

        # UI
        self.setup_ui()

    def setup_ui(self):
        # Big time display
        self.time_label = tk.Label(
            self.root, text="00:00:00", font=("Arial", 48), fg="#4f9dff"
        )
        self.time_label.pack(pady=20)

        # Controls
        btn_frame = tk.Frame(self.root)
        btn_frame.pack()

        self.start_btn = tk.Button(
            btn_frame, text="Start", command=self.toggle, width=8
        )
        self.start_btn.pack(side=tk.LEFT, padx=5)

        self.reset_btn = tk.Button(btn_frame, text="Reset", command=self.reset, width=8)
        self.reset_btn.pack(side=tk.LEFT, padx=5)

        # Target time selector
        target_frame = tk.Frame(self.root)
        target_frame.pack(pady=10)

        tk.Label(target_frame, text="Target:").pack(side=tk.LEFT)
        self.target_var = tk.StringVar(value="60")
        tk.Entry(target_frame, textvariable=self.target_var, width=5).pack(side=tk.LEFT)
        tk.Label(target_frame, text="min").pack(side=tk.LEFT)

    def toggle(self):
        if self.running:
            self.pause()
        else:
            self.start()

    def start(self):
        self.running = True
        self.start_btn.config(text="Pause")
        self.target = int(self.target_var.get()) * 60
        self.tick()

    def pause(self):
        self.running = False
        self.start_btn.config(text="Resume")

    def reset(self):
        self.running = False
        self.elapsed = 0
        self.start_btn.config(text="Start")
        self.update_display()

    def tick(self):
        if self.running:
            self.elapsed += 1
            self.update_display()

            # Check if done
            if self.elapsed >= self.target:
                self.alert()
                self.pause()
            else:
                self.root.after(1000, self.tick)

    def update_display(self):
        hours = self.elapsed // 3600
        mins = (self.elapsed % 3600) // 60
        secs = self.elapsed % 60
        self.time_label.config(text=f"{hours:02d}:{mins:02d}:{secs:02d}")

    def alert(self):
        # macOS notification sound
        subprocess.run(["afplay", "/System/Library/Sounds/Glass.aiff"])

        # Log session
        log_path = Path.home() / "focus_sessions.csv"
        with open(log_path, "a") as f:
            f.write(f"{datetime.now()},{self.elapsed // 60}\n")

    def run(self):
        self.root.mainloop()


if __name__ == "__main__":
    timer = FocusTimer()
    timer.run()
