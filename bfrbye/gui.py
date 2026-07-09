import base64
import os
import tkinter as tk
from tkinter import messagebox
from threading import Thread
from bfrbye.icon import icon
from bfrbye.config import load_config, save_config
from bfrbye.tracker import HandTracker


def create_main_window():
    """
    Creates the main application window with Start, Preview, and Configuration buttons.
    """
    root = tk.Tk()
    root.title("BFRBye")
    root.geometry("280x160")

    # Set icon
    icon_image = tk.PhotoImage(data=base64.b64decode(icon))
    root.iconphoto(True, icon_image)

    # Load config
    config = load_config()

    # Tracker instance
    tracker = HandTracker(config)
    tracker_active = [False]  # mutable container for nested access

    def start_tracking():
        tracker_active[0] = True
        root.iconify()

        thread = Thread(target=tracker.run, daemon=True)
        thread.start()
        start_button.config(text="Running", state="disabled")
        preview_button.config(state="disabled")

    def start_preview():
        tracker_active[0] = True
        root.iconify()

        thread = Thread(target=tracker.run_preview, daemon=True)
        thread.start()
        preview_button.config(text="Preview…", state="disabled")
        start_button.config(state="disabled")

        # Poll thread to re-enable buttons when preview closes
        def check_thread():
            if thread.is_alive():
                root.after(500, check_thread)
            else:
                tracker_active[0] = False
                preview_button.config(text="Preview", state="normal")
                start_button.config(state="normal")
                root.deiconify()
        root.after(500, check_thread)

    # Buttons
    start_button = tk.Button(root, text="Start", command=start_tracking)
    start_button.pack(pady=8)

    preview_button = tk.Button(root, text="Preview", command=start_preview)
    preview_button.pack(pady=8)

    config_button = tk.Button(root, text="Configuration",
                              command=lambda: open_config_window(root, config))
    config_button.pack(pady=8)

    return root


def open_config_window(parent, config):
    win = tk.Toplevel(parent)
    win.title("Configuration")
    win.resizable(False, False)

    # ── Notion ─────────────────────────────────────────────────
    tk.Label(win, text="Notion Token:").grid(row=0, column=0, sticky="w", padx=5, pady=3)
    token_entry = tk.Entry(win, width=40)
    token_entry.insert(0, config["notion"].get("token", ""))
    token_entry.grid(row=0, column=1, padx=5, pady=3)

    tk.Label(win, text="Database ID:").grid(row=1, column=0, sticky="w", padx=5, pady=3)
    db_entry = tk.Entry(win, width=40)
    db_entry.insert(0, config["notion"].get("database_id", ""))
    db_entry.grid(row=1, column=1, padx=5, pady=3)

    # ── Storage ────────────────────────────────────────────────
    tk.Label(win, text="Storage methods:").grid(row=2, column=0, sticky="w", padx=5, pady=3)
    methods = {"csv": tk.BooleanVar(), "txt": tk.BooleanVar(), "notion": tk.BooleanVar()}
    for i, method in enumerate(methods):
        methods[method].set(method in config["storage"].get("methods", []))
        tk.Checkbutton(win, text=method.upper(), variable=methods[method]).grid(
            row=2, column=1 + i, padx=3
        )

    # ── Processing ─────────────────────────────────────────────
    proc = config.setdefault("processing", {})

    tk.Label(win, text="Process interval:").grid(row=3, column=0, sticky="w", padx=5, pady=3)
    interval_var = tk.IntVar(value=proc.get("interval", 1))
    interval_spin = tk.Spinbox(win, from_=1, to=20, width=5,
                               textvariable=interval_var)
    interval_spin.grid(row=3, column=1, sticky="w", padx=5, pady=3)
    tk.Label(win, text="frames (1 = every frame)").grid(row=3, column=1, columnspan=3,
                                                         sticky="w", padx=75, pady=3)

    tk.Label(win, text="Mouth zone:").grid(row=4, column=0, sticky="w", padx=5, pady=3)
    pad_var = tk.DoubleVar(value=proc.get("mouth_padding", 0.5))
    pad_scale = tk.Scale(win, from_=0.0, to=3.0, resolution=0.1,
                         orient="horizontal", length=180,
                         variable=pad_var, showvalue=True)
    pad_scale.grid(row=4, column=1, columnspan=2, sticky="w", padx=5, pady=3)

    tk.Label(win, text="Trigger frames:").grid(row=5, column=0, sticky="w", padx=5, pady=3)
    trigger_var = tk.IntVar(value=proc.get("trigger_frames", 5))
    trigger_spin = tk.Spinbox(win, from_=1, to=30, width=5,
                              textvariable=trigger_var)
    trigger_spin.grid(row=5, column=1, sticky="w", padx=5, pady=3)
    tk.Label(win, text="frames to enter ACTIVE").grid(
        row=5, column=1, columnspan=3, sticky="w", padx=75, pady=3
    )

    tk.Label(win, text="Min duration:").grid(row=6, column=0, sticky="w", padx=5, pady=3)
    mindur_var = tk.DoubleVar(value=proc.get("min_duration", 0.5))
    mindur_scale = tk.Scale(win, from_=0.2, to=3.0, resolution=0.1,
                            orient="horizontal", length=180,
                            variable=mindur_var, showvalue=True)
    mindur_scale.grid(row=6, column=1, columnspan=2, sticky="w", padx=5, pady=3)

    tk.Label(win, text="Cooldown:").grid(row=7, column=0, sticky="w", padx=5, pady=3)
    cooldown_var = tk.DoubleVar(value=proc.get("cooldown", 0.5))
    cooldown_scale = tk.Scale(win, from_=0.0, to=5.0, resolution=0.1,
                              orient="horizontal", length=180,
                              variable=cooldown_var, showvalue=True)
    cooldown_scale.grid(row=7, column=1, columnspan=2, sticky="w", padx=5, pady=3)

    # ── Camera resolution ──────────────────────────────────────
    resolutions = [(160, 120), (320, 240), (424, 240), (640, 480), (848, 480), (960, 540), (1280, 720)]
    res_labels = [f"{w}x{h}" for w, h in resolutions]

    cur_w = proc.get("camera_width", 640)
    cur_h = proc.get("camera_height", 480)
    cur_res = f"{cur_w}x{cur_h}"
    res_var = tk.StringVar(value=cur_res if cur_res in res_labels else res_labels[3])

    tk.Label(win, text="Camera res:").grid(row=8, column=0, sticky="w", padx=5, pady=3)
    res_menu = tk.OptionMenu(win, res_var, *res_labels)
    res_menu.config(width=10)
    res_menu.grid(row=8, column=1, sticky="w", padx=5, pady=3)

    # ── Save ───────────────────────────────────────────────────
    def save_and_close():
        config["notion"]["token"] = token_entry.get().strip()
        config["notion"]["database_id"] = db_entry.get().strip()
        config["storage"]["methods"] = [m for m, var in methods.items() if var.get()]
        config["processing"]["interval"] = interval_var.get()
        config["processing"]["mouth_padding"] = round(pad_var.get(), 1)
        config["processing"]["trigger_frames"] = trigger_var.get()
        config["processing"]["min_duration"] = mindur_var.get()
        config["processing"]["cooldown"] = cooldown_var.get()
        w_str, h_str = res_var.get().split("x")
        config["processing"]["camera_width"] = int(w_str)
        config["processing"]["camera_height"] = int(h_str)
        save_config(config)
        messagebox.showinfo("Saved", "Configuration saved successfully")
        win.destroy()

    tk.Button(win, text="Save", command=save_and_close).grid(
        row=9, column=0, columnspan=3, pady=12
    )
