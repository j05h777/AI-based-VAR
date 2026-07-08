from pathlib import Path
import tkinter as tk
from tkinter import ttk, messagebox
import subprocess
import sys

APP_TITLE = "AI-Based VAR System"

BASE_DIR = Path(r"D:\joshb\VAR_code")
LOGO_FILE = BASE_DIR / "vars_logo.png"

COMMANDS = {
    "Player Tracking": {
        "cmd": [sys.executable, str(BASE_DIR / "player_tracking" / "short_track.py")],
        "cwd": BASE_DIR / "player_tracking",
    },
    "Foul Detection": {
        "cmd": ["conda", "run", "-n", "vars", "python", "main.py"],
        "cwd": BASE_DIR / "sn-mvfoul" / "VARS interface",
    },
    "Offside Classification": {
        "cmd": [sys.executable, "GUI.py"],
        "cwd": BASE_DIR / "offside_detection",
    },
}


class VARLauncherApp:
    def __init__(self, root):
        self.root = root
        self.root.title(APP_TITLE)
        self.root.state("zoomed")
        self.root.configure(bg="#0d0d3f")

        self.status_var = tk.StringVar(value="Ready.")
        self.logo_image = None

        self.configure_style()
        self.build_ui()

    def configure_style(self):
        style = ttk.Style()
        try:
            style.theme_use("clam")
        except tk.TclError:
            pass

        style.configure("Main.TFrame", background="#0d0d3f")
        style.configure("Bottom.TFrame", background="#0d0d3f")
        style.configure("Launch.TButton", font=("Arial", 12, "bold"), padding=(18, 12))
        style.configure("Status.TLabel", background="#0d0d3f", foreground="white", font=("Arial", 10))

    def build_ui(self):
        main = ttk.Frame(self.root, style="Main.TFrame", padding=20)
        main.pack(fill="both", expand=True)

        image_frame = ttk.Frame(main, style="Main.TFrame")
        image_frame.pack(fill="both", expand=True)

        self.image_label = tk.Label(
            image_frame,
            bg="#0d0d3f",
            bd=0,
            highlightthickness=0,
        )
        self.image_label.pack(expand=True)

        self.load_logo()

        button_frame = ttk.Frame(main, style="Bottom.TFrame")
        button_frame.pack(side="bottom", pady=(10, 0))

        ttk.Button(
            button_frame,
            text="Player Tracking",
            style="Launch.TButton",
            command=lambda: self.run_module("Player Tracking"),
        ).pack(side="left", padx=10)

        ttk.Button(
            button_frame,
            text="Foul Detection",
            style="Launch.TButton",
            command=lambda: self.run_module("Foul Detection"),
        ).pack(side="left", padx=10)

        ttk.Button(
            button_frame,
            text="Offside Classification",
            style="Launch.TButton",
            command=lambda: self.run_module("Offside Classification"),
        ).pack(side="left", padx=10)

        ttk.Label(
            main,
            textvariable=self.status_var,
            style="Status.TLabel"
        ).pack(side="bottom", pady=(12, 0))

    def load_logo(self):
        try:
            if not LOGO_FILE.exists():
                self.image_label.config(
                    text=f"Logo image not found:\n{LOGO_FILE}",
                    fg="white",
                    font=("Arial", 14),
                    justify="center",
                )
                self.status_var.set(f"Missing logo: {LOGO_FILE}")
                return

            self.logo_image = tk.PhotoImage(file=str(LOGO_FILE))
            self.image_label.config(image=self.logo_image, text="")
            self.status_var.set("Logo loaded successfully.")
        except Exception as e:
            self.image_label.config(
                text=f"Could not load logo:\n{e}",
                fg="white",
                font=("Arial", 14),
                justify="center",
            )
            self.status_var.set("Logo load failed.")

    def run_module(self, module_name):
        config = COMMANDS.get(module_name)

        if not config:
            self.status_var.set(f"No command linked for {module_name}.")
            messagebox.showerror("Error", f"No command linked for {module_name}.")
            return

        cmd = config["cmd"]
        cwd = config["cwd"]

        if not cwd.exists():
            self.status_var.set(f"Folder not found: {cwd}")
            messagebox.showerror("Folder Not Found", f"The folder does not exist:\n{cwd}")
            return

        try:
            subprocess.Popen(cmd, cwd=str(cwd))
            self.status_var.set(f"Launched: {module_name}")
        except FileNotFoundError:
            self.status_var.set(f"Launch failed: {module_name}")
            messagebox.showerror(
                "Launch Error",
                f"Executable or script not found for:\n{module_name}\n\nCommand: {cmd}\nWorking folder: {cwd}",
            )
        except Exception as e:
            self.status_var.set(f"Launch failed: {module_name}")
            messagebox.showerror(
                "Launch Error",
                f"Could not launch {module_name}.\n\n{e}",
            )


def main():
    root = tk.Tk()
    app = VARLauncherApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()