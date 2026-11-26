# views/main_window.py
"""Tkinter view — no detection logic, only presentation + user events."""
from __future__ import annotations

import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

import cv2
from PIL import Image, ImageTk

from controllers.app_controller import AppController
from config.settings import OUTPUT_DIR, SUPPORTED_EXTENSIONS


class MainWindow:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("Wall Crack Detection System")
        self.root.geometry("1200x800")
        self.root.minsize(900, 650)

        self.controller = AppController(output_dir=OUTPUT_DIR)

        self.image_path: Path | None = None
        self.original_photo = None
        self.annotated_photo = None

        self._build_interface()

    # ------------------------------------------------------------------
    # GUI construction
    # ------------------------------------------------------------------
    def _build_interface(self):
        main_frame = ttk.Frame(self.root, padding=10)
        main_frame.pack(fill=tk.BOTH, expand=True)

        ttk.Label(
            main_frame,
            text="Wall Crack Detection System",
            font=("Arial", 20, "bold"),
        ).pack(pady=10)

        # --- Input controls ------------------------------------------------
        input_frame = ttk.LabelFrame(
            main_frame, text="Inspection Information", padding=10
        )
        input_frame.pack(fill=tk.X, pady=5)

        ttk.Label(input_frame, text="Wall ID:").grid(
            row=0, column=0, padx=5, pady=5
        )
        self.wall_id_entry = ttk.Entry(input_frame, width=25)
        self.wall_id_entry.grid(row=0, column=1, padx=5, pady=5)
        self.wall_id_entry.insert(0, "WALL-001")

        ttk.Label(input_frame, text="Pixels per mm:").grid(
            row=0, column=2, padx=5, pady=5
        )
        self.calibration_entry = ttk.Entry(input_frame, width=15)
        self.calibration_entry.grid(row=0, column=3, padx=5, pady=5)

        ttk.Label(input_frame, text="Leave blank if uncalibrated").grid(
            row=0, column=4, padx=5, pady=5
        )

        # --- Buttons -------------------------------------------------------
        button_frame = ttk.Frame(main_frame)
        button_frame.pack(fill=tk.X, pady=10)

        self.load_button = ttk.Button(
            button_frame, text="Load Image", command=self.load_image
        )
        self.load_button.pack(side=tk.LEFT, padx=5)

        self.detect_button = ttk.Button(
            button_frame, text="Detect & Save",
            command=self.start_detection, state=tk.DISABLED,
        )
        self.detect_button.pack(side=tk.LEFT, padx=5)

        # --- Image preview -------------------------------------------------
        preview_frame = ttk.Frame(main_frame)
        preview_frame.pack(fill=tk.BOTH, expand=True, pady=5)

        original_frame = ttk.LabelFrame(
            preview_frame, text="Original Image", padding=5
        )
        original_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5)

        annotated_frame = ttk.LabelFrame(
            preview_frame, text="Detected Cracks", padding=5
        )
        annotated_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5)

        self.original_label = ttk.Label(original_frame, text="No image loaded")
        self.original_label.pack(fill=tk.BOTH, expand=True)

        self.annotated_label = ttk.Label(
            annotated_frame, text="Detection preview"
        )
        self.annotated_label.pack(fill=tk.BOTH, expand=True)

        # --- Results -------------------------------------------------------
        result_frame = ttk.LabelFrame(
            main_frame, text="Detection Results", padding=5
        )
        result_frame.pack(fill=tk.BOTH, expand=True, pady=5)

        self.result_text = tk.Text(result_frame, height=9, wrap=tk.WORD)
        self.result_text.pack(fill=tk.BOTH, expand=True)

        # --- Status bar ----------------------------------------------------
        self.status_label = ttk.Label(
            main_frame, text="Ready", relief=tk.SUNKEN, anchor=tk.W
        )
        self.status_label.pack(fill=tk.X, pady=5)

        self.progress = ttk.Progressbar(main_frame, mode="indeterminate")
        self.progress.pack(fill=tk.X)

    # ------------------------------------------------------------------
    # Image loading
    # ------------------------------------------------------------------
    def load_image(self):
        filetypes = [
            ("Image files", " ".join(f"*{e}" for e in sorted(SUPPORTED_EXTENSIONS))),
            ("All files", "*.*"),
        ]
        path = filedialog.askopenfilename(
            title="Select Wall Image", filetypes=filetypes
        )
        if not path:
            return

        self.image_path = Path(path)
        image = cv2.imread(str(self.image_path))
        if image is None:
            messagebox.showerror("Error", "Unable to load image.")
            return

        self._display_image(image, self.original_label, "original")
        self.annotated_label.configure(image="", text="Detection preview")
        self.annotated_photo = None
        self.detect_button.configure(state=tk.NORMAL)

        self.result_text.delete("1.0", tk.END)
        self.result_text.insert(
            tk.END, f"Loaded image: {self.image_path.name}\n"
        )
        self.status_label.configure(text="Image loaded successfully.")

    # ------------------------------------------------------------------
    def _display_image(self, image, label, image_type):
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(rgb)
        pil_image.thumbnail((520, 350))
        photo = ImageTk.PhotoImage(pil_image)
        label.configure(image=photo, text="")
        if image_type == "original":
            self.original_photo = photo
        else:
            self.annotated_photo = photo

    # ------------------------------------------------------------------
    # Detection
    # ------------------------------------------------------------------
    def start_detection(self):
        if self.image_path is None:
            return

        wall_id = self.wall_id_entry.get().strip()
        if not wall_id:
            messagebox.showwarning(
                "Missing information", "Please enter a Wall ID."
            )
            return

        calibration_text = self.calibration_entry.get().strip()
        pixel_per_mm = None
        if calibration_text:
            try:
                pixel_per_mm = float(calibration_text)
                if pixel_per_mm <= 0:
                    raise ValueError
            except ValueError:
                messagebox.showerror(
                    "Invalid calibration",
                    "Pixels per mm must be a positive number.",
                )
                return

        self.detect_button.configure(state=tk.DISABLED)
        self.load_button.configure(state=tk.DISABLED)
        self.progress.start(10)

        self.controller.start_detection(
            image_path=self.image_path,
            wall_id=wall_id,
            pixel_per_mm=pixel_per_mm,
            on_success=lambda r: self.root.after(0, self._on_success, r),
            on_error=lambda e: self.root.after(0, self._on_error, e),
            on_status=lambda s: self.root.after(0, self._set_status, s),
        )

    # ------------------------------------------------------------------
    def _set_status(self, text: str):
        self.status_label.configure(text=text)

    def _on_success(self, result: dict):
        self.progress.stop()
        self.detect_button.configure(state=tk.NORMAL)
        self.load_button.configure(state=tk.NORMAL)

        # Re-read the annotated image from disk for display
        annotated = cv2.imread(result["annotated"])
        if annotated is not None:
            self._display_image(annotated, self.annotated_label, "annotated")

        self._populate_results(result)
        self.status_label.configure(
            text="Detection completed. Results saved to MongoDB."
        )

    def _on_error(self, error: str):
        self.progress.stop()
        self.detect_button.configure(state=tk.NORMAL)
        self.load_button.configure(state=tk.NORMAL)
        self.status_label.configure(text="Detection failed.")
        messagebox.showerror("Detection Error", error)

    # ------------------------------------------------------------------
    def _populate_results(self, result: dict):
        self.result_text.delete("1.0", tk.END)
        self.result_text.insert(
            tk.END,
            f"DETECTION COMPLETED\n"
            f"==============================\n\n"
            f"Total cracks detected: {result['crack_count']}\n"
            f"MongoDB document ID: {result['mongodb_id']}\n\n",
        )
        # Crack details are stored in the JSON; read them back for display
        import json
        try:
            with open(result["json"], encoding="utf-8") as fh:
                doc = json.load(fh)
            for crack in doc.get("cracks", []):
                self.result_text.insert(
                    tk.END,
                    f"{crack['crack_id']}\n"
                    f"  Centroid: {crack['coordinates_px']['centroid']}\n"
                    f"  Bounding box: {crack['coordinates_px']['bounding_box']}\n"
                    f"  Length: {crack['length_px']} px\n"
                    f"  Mean width: {crack['mean_width_px']} px\n"
                    f"  Max width: {crack['max_width_px']} px\n"
                    f"  Orientation: {crack['orientation_degrees']}°\n\n",
                )
        except (OSError, KeyError, json.JSONDecodeError):
            pass

        self.result_text.insert(
            tk.END,
            f"\nOUTPUT FILES\n"
            f"Annotated: {result['annotated']}\n"
            f"Mask: {result['mask']}\n"
            f"JSON: {result['json']}\n",
        )