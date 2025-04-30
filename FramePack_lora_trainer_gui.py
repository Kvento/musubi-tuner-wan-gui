import tkinter as tk
from tkinter import ttk, filedialog, messagebox, Menu
import subprocess
import threading
import json
import os
import signal

# Dark theme color scheme
BG_COLOR = "#2C3E50"       # Main background (dark gray with blue tint)
FG_COLOR = "#ECF0F1"       # Light text
ACCENT_COLOR = "#2980B9"   # Blue accent for tabs
ENTRY_BG = "#1B2A38"       # Entry field background (darker than main)
BUTTON_ACTIVE = "#1B2A38"  # Active button background
BORDER_COLOR = "#333333"   # Dark border color
ACTIVE_ENTRY_BG = "white"  # Background color for active entry field
ACTIVE_ENTRY_FG = "black"  # Text color for active entry field

class FramePackLoRATrainerGUI:
    def __init__(self, master):
        self.master = master
        master.title("FramePack LoRA Trainer")
        master.geometry("900x1024")
        master.configure(bg=BG_COLOR)

        self.current_process = None
        self.training_thread = None
        self.process_group_id = None
        self.user_scrolled = False  # Flag for manual console scrolling

        # Initialize settings with default values
        self.settings = {
            "DATASET_CONFIG": "dataset/dataset_example.toml",
            "DIT_MODEL": "path/to/dit_model.safetensors",
            "VAE_MODEL": "path/to/vae_model.pth",
            "TEXT_ENCODER_1": "path/to/text_encoder1.pth",
            "TEXT_ENCODER_2": "path/to/text_encoder2.pth",
            "IMAGE_ENCODER": "path/to/image_encoder.pth",
            "LORA_OUTPUT_DIR": "Output_LoRAs/",
            "LORA_NAME": "My_Best_Lora_v1",
            "LEARNING_RATE": 2e-5,
            "NETWORK_DIM": 32,
            "NETWORK_ALPHA": 4,
            "MAX_TRAIN_EPOCHS": 70,
            "SAVE_EVERY_N_EPOCHS": 10,
            "SEED": 1234,
            "OPTIMIZER_TYPE": "adamw8bit",
            "OPTIMIZER_ARGS": "",
            "ATTENTION_MECHANISM": "none",
            "LOGGING_DIR": "",
            "LOG_WITH": "none",
            "LOG_PREFIX": "",
            "IMG_IN_TXT_IN_OFFLOADING": False,
            "LR_SCHEDULER": "constant",
            "LR_WARMUP_STEPS": "",
            "LR_DECAY_STEPS": "",
            "TIMESTEP_SAMPLING": "shift",
            "DISCRETE_FLOW_SHIFT": "3.0",
            "WEIGHTING_SCHEME": "none",
            "METADATA_TITLE": "",
            "METADATA_AUTHOR": "",
            "METADATA_DESCRIPTION": "",
            "METADATA_LICENSE": "",
            "METADATA_TAGS": "",
        }

        self.optimizer_types = ["adamw", "adamw8bit", "adafactor", "torch.optim.AdamW", "bitsandbytes.optim.AdEMAMix8bit", "bitsandbytes.optim.PagedAdEMAMix8bit", "came"]

        self.setup_styles()

        # Create notebook and tabs
        self.notebook = ttk.Notebook(master)
        self.notebook.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        self.training_tab = ttk.Frame(self.notebook)
        self.training_tab.bind("<Button-1>", self.remove_focus)
        self.notebook.add(self.training_tab, text="Training Settings")

        self.advanced_tab = ttk.Frame(self.notebook)
        self.advanced_tab.bind("<Button-1>", self.remove_focus)
        self.notebook.add(self.advanced_tab, text="Advanced Settings")

        # Initialize tab contents
        self.create_training_settings()
        self.create_advanced_settings()

        # Create context menu for copying console text
        self.context_menu = Menu(self.master, tearoff=0)
        self.context_menu.add_command(label="Copy", command=self.copy_selected_text)

    def remove_focus(self, event):
        self.master.focus_set()

    def setup_styles(self):
        style = ttk.Style()
        style.theme_use("clam")
        style.configure(".", background=BG_COLOR, foreground=FG_COLOR)
        style.configure("TFrame", background=BG_COLOR)
        style.configure("TLabel", background=BG_COLOR, foreground=FG_COLOR)
        style.configure("TButton", background=BG_COLOR, foreground=FG_COLOR, bordercolor=BORDER_COLOR, borderwidth=1, focusthickness=3, focuscolor=BG_COLOR, padding=[5, 1])
        style.map("TButton", background=[("active", BUTTON_ACTIVE), ("pressed", BUTTON_ACTIVE)], foreground=[("active", FG_COLOR), ("pressed", FG_COLOR)])
        style.configure("TCheckbutton", background=BG_COLOR, foreground=FG_COLOR)
        style.map("TCheckbutton", background=[("active", BG_COLOR)], foreground=[("active", FG_COLOR)])
        style.configure("TNotebook", background=BG_COLOR, borderwidth=0)
        style.configure("TNotebook.Tab", background=BG_COLOR, foreground=FG_COLOR, padding=[5, 2])
        style.map("TNotebook.Tab", background=[("selected", ACCENT_COLOR)], foreground=[("selected", FG_COLOR)])
        style.configure("TEntry", fieldbackground=ENTRY_BG, foreground=FG_COLOR, bordercolor=BORDER_COLOR)
        style.map("TEntry", fieldbackground=[("focus", ACTIVE_ENTRY_BG)], foreground=[("focus", ACTIVE_ENTRY_FG)])
        style.configure("TCombobox", fieldbackground=ENTRY_BG, background=BG_COLOR, foreground=FG_COLOR, bordercolor=BORDER_COLOR)
        style.map("TCombobox", fieldbackground=[("focus", ACTIVE_ENTRY_BG), ("readonly", ENTRY_BG), ("!disabled", ENTRY_BG)], foreground=[("focus", ACTIVE_ENTRY_FG), ("readonly", FG_COLOR), ("!disabled", FG_COLOR)], selectbackground=[("readonly", ENTRY_BG), ("!disabled", ENTRY_BG)], selectforeground=[("readonly", FG_COLOR), ("!disabled", FG_COLOR)])
        style.configure("Vertical.TScrollbar", background=ENTRY_BG, troughcolor=BG_COLOR, bordercolor=BORDER_COLOR, arrowcolor=FG_COLOR, darkcolor=BG_COLOR, lightcolor=BG_COLOR)
        style.map("Vertical.TScrollbar", background=[("active", BUTTON_ACTIVE), ("pressed", BUTTON_ACTIVE)])

    def create_training_settings(self):
        row = 0
        ttk.Label(self.training_tab, text="Training Settings", font=("Arial", 12, "bold")).grid(row=row, column=0, columnspan=3, pady=(10, 10))
        row += 1

        button_frame_top = ttk.Frame(self.training_tab)
        button_frame_top.grid(row=row, column=0, columnspan=3, pady=5)
        ttk.Button(button_frame_top, text="Load Settings", command=self.load_settings).pack(side=tk.LEFT, padx=10)
        ttk.Button(button_frame_top, text="Save Settings", command=self.save_settings).pack(side=tk.LEFT, padx=10)
        row += 1

        settings_config = [
            ("Dataset Config", "DATASET_CONFIG", "file"),
            ("DiT Model", "DIT_MODEL", "file"),
            ("VAE Model", "VAE_MODEL", "file"),
            ("Text Encoder 1", "TEXT_ENCODER_1", "file"),
            ("Text Encoder 2", "TEXT_ENCODER_2", "file"),
            ("Image Encoder", "IMAGE_ENCODER", "file"),
            ("LoRA Output Dir", "LORA_OUTPUT_DIR", "directory"),
            ("LoRA Name", "LORA_NAME", "text"),
            ("Learning Rate", "LEARNING_RATE", "float"),
            ("Network Dim", "NETWORK_DIM", "int"),
            ("Network Alpha", "NETWORK_ALPHA", "float"),
            ("Max Train Epochs", "MAX_TRAIN_EPOCHS", "int"),
            ("Save Every N Epochs", "SAVE_EVERY_N_EPOCHS", "int"),
            ("Seed", "SEED", "int"),
            ("Optimizer Type", "OPTIMIZER_TYPE", "dropdown"),
            ("Optimizer Args", "OPTIMIZER_ARGS", "text"),
        ]

        self.entries = {}

        for label_text, key, input_type in settings_config:
            ttk.Label(self.training_tab, text=f"{label_text}:").grid(row=row, column=0, sticky=tk.W, padx=5, pady=2)
            if input_type == "dropdown":
                var = tk.StringVar(value=self.settings[key])
                self.entries[key] = ttk.Combobox(self.training_tab, textvariable=var, values=self.optimizer_types, state="readonly")
                self.entries[key].current(self.optimizer_types.index(self.settings[key]))
            else:
                self.entries[key] = ttk.Entry(self.training_tab, width=40)
                self.entries[key].insert(0, self.settings[key])
            self.entries[key].grid(row=row, column=1, sticky=tk.EW, padx=5, pady=2)
            if input_type in ["file", "directory"]:
                ttk.Button(self.training_tab, text="Browse", command=lambda k=key, t=input_type: self.browse_file(k, t)).grid(row=row, column=2, sticky=tk.W, padx=5)
            row += 1

        self.enable_cache_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(self.training_tab, text="Enable Cache Preparation", variable=self.enable_cache_var).grid(row=row, column=0, columnspan=3, pady=5)
        row += 1

        button_frame = ttk.Frame(self.training_tab)
        button_frame.grid(row=row, column=0, columnspan=3, pady=10)
        ttk.Button(button_frame, text="Start Training", command=self.start_training).pack(side=tk.LEFT, padx=10)
        ttk.Button(button_frame, text="Stop Training", command=self.stop_training).pack(side=tk.LEFT, padx=10)
        row += 1

        self.console_frame = ttk.Frame(self.training_tab)
        self.console_frame.grid(row=row, column=0, columnspan=3, padx=5, pady=5, sticky="nsew")
        self.console_output = tk.Text(self.console_frame, height=10, width=80, bg=ENTRY_BG, fg=FG_COLOR, wrap="word", state="disabled", selectbackground="white", selectforeground="black")
        self.console_output.grid(row=0, column=0, sticky="nsew")
        self.console_scrollbar = ttk.Scrollbar(self.console_frame, orient="vertical", command=self.console_output.yview, style="Vertical.TScrollbar")
        self.console_scrollbar.grid(row=0, column=1, sticky="ns")
        self.console_output.configure(yscrollcommand=self.console_scrollbar.set)
        self.console_output.bind("<MouseWheel>", self.on_mousewheel)
        self.console_output.bind("<Button-4>", self.on_mousewheel)
        self.console_output.bind("<Button-5>", self.on_mousewheel)
        self.console_output.bind("<Button-3>", self.show_context_menu)
        self.training_tab.grid_rowconfigure(row, weight=1)
        self.training_tab.grid_columnconfigure(1, weight=1)
        self.console_frame.grid_rowconfigure(0, weight=1)
        self.console_frame.grid_columnconfigure(0, weight=1)

    def create_advanced_settings(self):
        row = 0
        ttk.Label(self.advanced_tab, text="Advanced Settings", font=("Arial", 12, "bold")).grid(row=row, column=0, columnspan=3, pady=(10, 10))
        row += 1

        # Attention Mechanism
        ttk.Label(self.advanced_tab, text="Attention Mechanism:").grid(row=row, column=0, sticky=tk.W, padx=5, pady=2)
        self.attention_var = tk.StringVar(value=self.settings["ATTENTION_MECHANISM"])
        attention_options = ["none", "sdpa", "flash_attn", "sage_attn", "xformers", "flash3", "split_attn"]
        self.entries["ATTENTION_MECHANISM"] = ttk.Combobox(self.advanced_tab, textvariable=self.attention_var, values=attention_options, state="readonly")
        self.entries["ATTENTION_MECHANISM"].grid(row=row, column=1, sticky=tk.EW, padx=5, pady=2)
        row += 1

        # Logging
        ttk.Label(self.advanced_tab, text="Logging Directory:").grid(row=row, column=0, sticky=tk.W, padx=5, pady=2)
        self.entries["LOGGING_DIR"] = ttk.Entry(self.advanced_tab, width=40)
        self.entries["LOGGING_DIR"].grid(row=row, column=1, sticky=tk.EW, padx=5, pady=2)
        ttk.Button(self.advanced_tab, text="Browse", command=lambda: self.browse_directory("LOGGING_DIR")).grid(row=row, column=2, sticky=tk.W, padx=5)
        row += 1

        ttk.Label(self.advanced_tab, text="Log With:").grid(row=row, column=0, sticky=tk.W, padx=5, pady=2)
        self.log_with_var = tk.StringVar(value=self.settings["LOG_WITH"])
        log_with_options = ["none", "tensorboard", "wandb", "all"]
        self.entries["LOG_WITH"] = ttk.Combobox(self.advanced_tab, textvariable=self.log_with_var, values=log_with_options, state="readonly")
        self.entries["LOG_WITH"].grid(row=row, column=1, sticky=tk.EW, padx=5, pady=2)
        row += 1

        ttk.Label(self.advanced_tab, text="Log Prefix:").grid(row=row, column=0, sticky=tk.W, padx=5, pady=2)
        self.entries["LOG_PREFIX"] = ttk.Entry(self.advanced_tab, width=40)
        self.entries["LOG_PREFIX"].grid(row=row, column=1, sticky=tk.EW, padx=5, pady=2)
        row += 1

        # Memory Management
        self.img_in_txt_in_offloading_var = tk.BooleanVar(value=self.settings["IMG_IN_TXT_IN_OFFLOADING"])
        ttk.Checkbutton(self.advanced_tab, text="Offload img_in and txt_in to CPU", variable=self.img_in_txt_in_offloading_var).grid(row=row, column=0, columnspan=3, pady=5)
        self.entries["IMG_IN_TXT_IN_OFFLOADING"] = self.img_in_txt_in_offloading_var
        row += 1

        # Learning Rate Scheduler
        ttk.Label(self.advanced_tab, text="LR Scheduler:").grid(row=row, column=0, sticky=tk.W, padx=5, pady=2)
        self.lr_scheduler_var = tk.StringVar(value=self.settings["LR_SCHEDULER"])
        lr_scheduler_options = ["linear", "cosine", "cosine_with_restarts", "polynomial", "constant", "constant_with_warmup", "adafactor"]
        self.entries["LR_SCHEDULER"] = ttk.Combobox(self.advanced_tab, textvariable=self.lr_scheduler_var, values=lr_scheduler_options, state="readonly")
        self.entries["LR_SCHEDULER"].grid(row=row, column=1, sticky=tk.EW, padx=5, pady=2)
        row += 1

        ttk.Label(self.advanced_tab, text="LR Warmup Steps:").grid(row=row, column=0, sticky=tk.W, padx=5, pady=2)
        self.entries["LR_WARMUP_STEPS"] = ttk.Entry(self.advanced_tab, width=40)
        self.entries["LR_WARMUP_STEPS"].grid(row=row, column=1, sticky=tk.EW, padx=5, pady=2)
        row += 1

        ttk.Label(self.advanced_tab, text="LR Decay Steps:").grid(row=row, column=0, sticky=tk.W, padx=5, pady=2)
        self.entries["LR_DECAY_STEPS"] = ttk.Entry(self.advanced_tab, width=40)
        self.entries["LR_DECAY_STEPS"].grid(row=row, column=1, sticky=tk.EW, padx=5, pady=2)
        row += 1

        # Timestep Sampling
        ttk.Label(self.advanced_tab, text="Timestep Sampling:").grid(row=row, column=0, sticky=tk.W, padx=5, pady=2)
        self.timestep_sampling_var = tk.StringVar(value=self.settings["TIMESTEP_SAMPLING"])
        timestep_sampling_options = ["sigma", "uniform", "sigmoid", "shift"]
        self.entries["TIMESTEP_SAMPLING"] = ttk.Combobox(self.advanced_tab, textvariable=self.timestep_sampling_var, values=timestep_sampling_options, state="readonly")
        self.entries["TIMESTEP_SAMPLING"].grid(row=row, column=1, sticky=tk.EW, padx=5, pady=2)
        row += 1

        ttk.Label(self.advanced_tab, text="Discrete Flow Shift:").grid(row=row, column=0, sticky=tk.W, padx=5, pady=2)
        self.entries["DISCRETE_FLOW_SHIFT"] = ttk.Entry(self.advanced_tab, width=40)
        self.entries["DISCRETE_FLOW_SHIFT"].insert(0, self.settings["DISCRETE_FLOW_SHIFT"])
        self.entries["DISCRETE_FLOW_SHIFT"].grid(row=row, column=1, sticky=tk.EW, padx=5, pady=2)
        row += 1

        # Weighting Scheme
        ttk.Label(self.advanced_tab, text="Weighting Scheme:").grid(row=row, column=0, sticky=tk.W, padx=5, pady=2)
        self.weighting_scheme_var = tk.StringVar(value=self.settings["WEIGHTING_SCHEME"])
        weighting_scheme_options = ["logit_normal", "mode", "cosmap", "sigma_sqrt", "none"]
        self.entries["WEIGHTING_SCHEME"] = ttk.Combobox(self.advanced_tab, textvariable=self.weighting_scheme_var, values=weighting_scheme_options, state="readonly")
        self.entries["WEIGHTING_SCHEME"].grid(row=row, column=1, sticky=tk.EW, padx=5, pady=2)
        row += 1

        # Metadata
        ttk.Label(self.advanced_tab, text="Metadata Title:").grid(row=row, column=0, sticky=tk.W, padx=5, pady=2)
        self.entries["METADATA_TITLE"] = ttk.Entry(self.advanced_tab, width=40)
        self.entries["METADATA_TITLE"].grid(row=row, column=1, sticky=tk.EW, padx=5, pady=2)
        row += 1

        ttk.Label(self.advanced_tab, text="Metadata Author:").grid(row=row, column=0, sticky=tk.W, padx=5, pady=2)
        self.entries["METADATA_AUTHOR"] = ttk.Entry(self.advanced_tab, width=40)
        self.entries["METADATA_AUTHOR"].grid(row=row, column=1, sticky=tk.EW, padx=5, pady=2)
        row += 1

        ttk.Label(self.advanced_tab, text="Metadata Description:").grid(row=row, column=0, sticky=tk.W, padx=5, pady=2)
        self.entries["METADATA_DESCRIPTION"] = ttk.Entry(self.advanced_tab, width=40)
        self.entries["METADATA_DESCRIPTION"].grid(row=row, column=1, sticky=tk.EW, padx=5, pady=2)
        row += 1

        ttk.Label(self.advanced_tab, text="Metadata License:").grid(row=row, column=0, sticky=tk.W, padx=5, pady=2)
        self.entries["METADATA_LICENSE"] = ttk.Entry(self.advanced_tab, width=40)
        self.entries["METADATA_LICENSE"].grid(row=row, column=1, sticky=tk.EW, padx=5, pady=2)
        row += 1

        ttk.Label(self.advanced_tab, text="Metadata Tags:").grid(row=row, column=0, sticky=tk.W, padx=5, pady=2)
        self.entries["METADATA_TAGS"] = ttk.Entry(self.advanced_tab, width=40)
        self.entries["METADATA_TAGS"].grid(row=row, column=1, sticky=tk.EW, padx=5, pady=2)
        row += 1

        self.advanced_tab.grid_column_configure(1, weight=1)

    def show_context_menu(self, event):
        try:
            self.context_menu.tk_popup(event.x_root, event.y_root)
        finally:
            self.context_menu.grab_release()

    def copy_selected_text(self):
        if self.console_output.selection_get():
            self.master.clipboard_clear()
            self.master.clipboard_append(self.console_output.selection_get())

    def browse_directory(self, setting_name):
        path = filedialog.askdirectory()
        if path:
            self.entries[setting_name].delete(0, tk.END)
            self.entries[setting_name].insert(0, path)

    def on_mousewheel(self, event):
        if self.console_output.yview()[1] < 1.0:
            self.user_scrolled = True
        else:
            self.user_scrolled = False

    def update_console(self, line):
        self.console_output.configure(state="normal")
        self.console_output.insert(tk.END, line)
        if not self.user_scrolled:
            self.console_output.yview(tk.END)
        self.console_output.configure(state="disabled")

    def browse_file(self, setting_name, input_type):
        if input_type == "directory":
            path = filedialog.askdirectory()
        else:
            path = filedialog.askopenfilename()
        if path:
            self.settings[setting_name] = path
            self.entries[setting_name].delete(0, tk.END)
            self.entries[setting_name].insert(0, self.settings[setting_name])

    def run_subprocess(self, cmd, name, callback=None):
        env = os.environ.copy()
        env["PYTHONIOENCODING"] = "utf-8"
        if os.name == 'nt':
            creationflags = subprocess.CREATE_NEW_PROCESS_GROUP
            preexec_fn = None
        else:
            creationflags = 0
            preexec_fn = os.setsid
        process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, bufsize=1, universal_newlines=True, encoding='utf-8', env=env, creationflags=creationflags, preexec_fn=preexec_fn)
        self.current_process = process
        if os.name == 'nt':
            self.process_group_id = process.pid

        def read_output(pipe, output_type):
            while True:
                line = pipe.readline()
                if not line:
                    break
                self.master.after(0, self.update_console, f"{name} {output_type}: {line}")
            pipe.close()

        threading.Thread(target=read_output, args=(process.stdout, "STDOUT"), daemon=True).start()
        threading.Thread(target=read_output, args=(process.stderr, "STDERR"), daemon=True).start()

        def check_process():
            process.wait()
            self.master.after(0, self.update_console, f"{name} process completed.\n")
            self.current_process = None
            if callback:
                callback()

        threading.Thread(target=check_process, daemon=True).start()

    def start_training(self):
        optimizer_type = self.entries["OPTIMIZER_TYPE"].get()
        if optimizer_type == "came":
            messagebox.showwarning("Предупреждение", "Оптимизатор 'came' не поддерживается в текущей версии. Пожалуйста, выберите другой оптимизатор, например 'adamw' или 'adamw8bit'.")
            return

        self.settings.update({
            "DATASET_CONFIG": self.entries["DATASET_CONFIG"].get(),
            "DIT_MODEL": self.entries["DIT_MODEL"].get(),
            "VAE_MODEL": self.entries["VAE_MODEL"].get(),
            "TEXT_ENCODER_1": self.entries["TEXT_ENCODER_1"].get(),
            "TEXT_ENCODER_2": self.entries["TEXT_ENCODER_2"].get(),
            "IMAGE_ENCODER": self.entries["IMAGE_ENCODER"].get(),
            "LORA_OUTPUT_DIR": self.entries["LORA_OUTPUT_DIR"].get(),
            "LORA_NAME": self.entries["LORA_NAME"].get(),
            "LEARNING_RATE": float(self.entries["LEARNING_RATE"].get()),
            "NETWORK_DIM": int(self.entries["NETWORK_DIM"].get()),
            "NETWORK_ALPHA": float(self.entries["NETWORK_ALPHA"].get()),
            "MAX_TRAIN_EPOCHS": int(self.entries["MAX_TRAIN_EPOCHS"].get()),
            "SAVE_EVERY_N_EPOCHS": int(self.entries["SAVE_EVERY_N_EPOCHS"].get()),
            "SEED": int(self.entries["SEED"].get()),
            "OPTIMIZER_TYPE": optimizer_type,
            "OPTIMIZER_ARGS": self.entries["OPTIMIZER_ARGS"].get(),
            "ATTENTION_MECHANISM": self.entries["ATTENTION_MECHANISM"].get(),
            "LOGGING_DIR": self.entries["LOGGING_DIR"].get(),
            "LOG_WITH": self.entries["LOG_WITH"].get(),
            "LOG_PREFIX": self.entries["LOG_PREFIX"].get(),
            "IMG_IN_TXT_IN_OFFLOADING": self.entries["IMG_IN_TXT_IN_OFFLOADING"].get(),
            "LR_SCHEDULER": self.entries["LR_SCHEDULER"].get(),
            "LR_WARMUP_STEPS": self.entries["LR_WARMUP_STEPS"].get(),
            "LR_DECAY_STEPS": self.entries["LR_DECAY_STEPS"].get(),
            "TIMESTEP_SAMPLING": self.entries["TIMESTEP_SAMPLING"].get(),
            "DISCRETE_FLOW_SHIFT": self.entries["DISCRETE_FLOW_SHIFT"].get(),
            "WEIGHTING_SCHEME": self.entries["WEIGHTING_SCHEME"].get(),
            "METADATA_TITLE": self.entries["METADATA_TITLE"].get(),
            "METADATA_AUTHOR": self.entries["METADATA_AUTHOR"].get(),
            "METADATA_DESCRIPTION": self.entries["METADATA_DESCRIPTION"].get(),
            "METADATA_LICENSE": self.entries["METADATA_LICENSE"].get(),
            "METADATA_TAGS": self.entries["METADATA_TAGS"].get(),
        })

        command = [
            "accelerate", "launch",
            "--num_cpu_threads_per_process", "2",
            "--mixed_precision", "bf16",
            "train_network.py",  # Updated to generic script name based on FramePack docs
            "--dit", self.settings["DIT_MODEL"],
            "--vae", self.settings["VAE_MODEL"],
            "--text_encoder", self.settings["TEXT_ENCODER_1"],
            "--text_encoder_2", self.settings["TEXT_ENCODER_2"],
            "--image_encoder", self.settings["IMAGE_ENCODER"],
            "--dataset_config", self.settings["DATASET_CONFIG"],
            "--output_dir", self.settings["LORA_OUTPUT_DIR"],
            "--output_name", self.settings["LORA_NAME"],
            "--learning_rate", str(self.settings["LEARNING_RATE"]),
            "--network_dim", str(self.settings["NETWORK_DIM"]),
            "--network_alpha", str(self.settings["NETWORK_ALPHA"]),
            "--max_train_epochs", str(self.settings["MAX_TRAIN_EPOCHS"]),
            "--save_every_n_epochs", str(self.settings["SAVE_EVERY_N_EPOCHS"]),
            "--seed", str(self.settings["SEED"]),
            "--optimizer_type", self.settings["OPTIMIZER_TYPE"],
            "--network_module", "networks.lora"  # Assumed based on standard LoRA training
        ]

        if self.settings["OPTIMIZER_ARGS"]:
            command.extend(["--optimizer_args", self.settings["OPTIMIZER_ARGS"]])

        attention = self.settings["ATTENTION_MECHANISM"]
        if attention != "none":
            command.append(f"--{attention}")

        logging_dir = self.settings["LOGGING_DIR"]
        if logging_dir:
            command.extend(["--logging_dir", logging_dir])

        log_with = self.settings["LOG_WITH"]
        if log_with != "none":
            command.extend(["--log_with", log_with])

        log_prefix = self.settings["LOG_PREFIX"]
        if log_prefix:
            command.extend(["--log_prefix", log_prefix])

        if self.settings["IMG_IN_TXT_IN_OFFLOADING"]:
            command.append("--img_in_txt_in_offloading")

        lr_scheduler = self.settings["LR_SCHEDULER"]
        if lr_scheduler:
            command.extend(["--lr_scheduler", lr_scheduler])

        lr_warmup_steps = self.settings["LR_WARMUP_STEPS"]
        if lr_warmup_steps:
            command.extend(["--lr_warmup_steps", lr_warmup_steps])

        lr_decay_steps = self.settings["LR_DECAY_STEPS"]
        if lr_decay_steps:
            command.extend(["--lr_decay_steps", lr_decay_steps])

        weighting_scheme = self.settings["WEIGHTING_SCHEME"]
        if weighting_scheme != "none":
            command.extend(["--weighting_scheme", weighting_scheme])

        metadata_title = self.settings["METADATA_TITLE"]
        if metadata_title:
            command.extend(["--metadata_title", metadata_title])

        metadata_author = self.settings["METADATA_AUTHOR"]
        if metadata_author:
            command.extend(["--metadata_author", metadata_author])

        metadata_description = self.settings["METADATA_DESCRIPTION"]
        if metadata_description:
            command.extend(["--metadata_description", metadata_description])

        metadata_license = self.settings["METADATA_LICENSE"]
        if metadata_license:
            command.extend(["--metadata_license", metadata_license])

        metadata_tags = self.settings["METADATA_TAGS"]
        if metadata_tags:
            command.extend(["--metadata_tags", metadata_tags])

        cache_preparation_command = [
            "python", "cache_latents.py",  # Assumed generic script name
            "--dataset_config", self.settings["DATASET_CONFIG"],
            "--vae", self.settings["VAE_MODEL"],
        ]

        text_encoder_caching_command = [
            "python", "cache_text_encoder_outputs.py",  # Assumed generic script name
            "--dataset_config", self.settings["DATASET_CONFIG"],
            "--text_encoder", self.settings["TEXT_ENCODER_1"],
            "--text_encoder_2", self.settings["TEXT_ENCODER_2"],
        ]

        self.console_output.configure(state="normal")
        self.console_output.delete(1.0, tk.END)
        self.console_output.configure(state="disabled")

        if self.enable_cache_var.get():
            self.update_console("Starting cache preparation...\n")

            def on_text_encoder_caching_complete():
                self.update_console("Text encoder caching completed.\nStarting training...\n")
                self.run_subprocess(command, "Training")

            def on_cache_preparation_complete():
                self.update_console("Cache preparation completed.\nStarting text encoder caching...\n")
                self.run_subprocess(text_encoder_caching_command, "Text Encoder Caching", on_text_encoder_caching_complete)

            self.run_subprocess(cache_preparation_command, "Cache Preparation", on_cache_preparation_complete)
        else:
            self.update_console("Starting training without caching...\n")
            self.run_subprocess(command, "Training")

    def stop_training(self):
        if self.current_process and self.current_process.poll() is None:
            try:
                if os.name == 'nt':
                    self.current_process.send_signal(signal.CTRL_BREAK_EVENT)
                else:
                    os.killpg(os.getpgid(self.current_process.pid), signal.SIGTERM)
            except Exception as e:
                self.update_console("Error stopping process: " + str(e) + "\n")
            try:
                self.current_process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                try:
                    self.current_process.kill()
                    self.current_process.wait()
                except Exception as e:
                    self.update_console("Error killing process: " + str(e) + "\n")
            self.current_process = None
            if self.training_thread:
                self.training_thread.join(timeout=1)
                self.training_thread = None
            self.update_console("Training stopped\n")
        else:
            self.update_console("No active process to stop\n")

    def save_settings(self):
        current_settings = {}
        for key, entry in self.entries.items():
            if isinstance(entry, ttk.Combobox):
                current_settings[key] = entry.get()
            elif isinstance(entry, tk.BooleanVar):
                current_settings[key] = entry.get()
            else:
                current_settings[key] = entry.get()
        current_settings["ENABLE_CACHE"] = self.enable_cache_var.get()
        file_path = filedialog.asksaveasfilename(defaultextension=".json", filetypes=[("JSON files", "*.json")])
        if file_path:
            with open(file_path, "w") as f:
                json.dump(current_settings, f, indent=4)

    def load_settings(self):
        file_path = filedialog.askopenfilename(filetypes=[("JSON files", "*.json")])
        if file_path:
            with open(file_path, "r") as f:
                loaded_settings = json.load(f)
            for key, value in loaded_settings.items():
                if key in self.entries:
                    if isinstance(self.entries[key], ttk.Combobox):
                        self.entries[key].set(value)
                    elif isinstance(self.entries[key], tk.BooleanVar):
                        self.entries[key].set(value)
                    else:
                        self.entries[key].delete(0, tk.END)
                        self.entries[key].insert(0, value)
            if "ENABLE_CACHE" in loaded_settings:
                self.enable_cache_var.set(loaded_settings["ENABLE_CACHE"])

root = tk.Tk()
gui = FramePackLoRATrainerGUI(root)
root.mainloop()