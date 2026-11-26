# main.py
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from pathlib import Path
from datetime import datetime, timezone
import threading
import hashlib
import json

import cv2
import numpy as np
from PIL import Image, ImageTk
from pymongo import MongoClient
from skimage.morphology import skeletonize


# ============================================================
# CONFIGURATION
# ============================================================
import os
from dotenv import load_dotenv

load_dotenv()

# The database name "document" is already included in this URI.
MONGODB_URI = os.getenv("MONGODB_URI")

COLLECTION_NAME = "cracks"

OUTPUT_DIR = Path("detection_results")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Detection parameters
MIN_CRACK_AREA = 15
MAX_CRACK_AREA = 100000
BLACKHAT_KERNEL_SIZE = 15
MORPH_KERNEL_SIZE = 3


# ============================================================
# MONGODB
# ============================================================

def get_database():
    """
    Connect using the single MongoDB connection string.
    The database is selected automatically from the URI.
    """

    client = MongoClient(
        MONGODB_URI,
        serverSelectionTimeoutMS=5000
    )

    # Verify connection
    client.admin.command("ping")

    # Get database from connection string
    db = client.get_default_database()

    collection = db[COLLECTION_NAME]

    return client, collection


# ============================================================
# UTILITIES
# ============================================================

def calculate_sha256(file_path):
    sha256 = hashlib.sha256()

    with open(file_path, "rb") as file:
        for chunk in iter(lambda: file.read(8192), b""):
            sha256.update(chunk)

    return sha256.hexdigest()


def safe_filename(filename):
    return Path(filename).stem.replace(" ", "_")


# ============================================================
# CRACK DETECTION
# ============================================================

def detect_cracks(image_path, pixels_per_mm=None):
    """
    Detect cracks using classical OpenCV image processing.

    Returns:
        original: Original image
        mask: Binary crack mask
        annotated: Image with detected cracks
        cracks: List of crack measurements
    """

    image = cv2.imread(str(image_path))

    if image is None:
        raise ValueError("Unable to read the selected image.")

    original = image.copy()
    annotated = image.copy()

    # --------------------------------------------------------
    # 1. Grayscale conversion
    # --------------------------------------------------------

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # --------------------------------------------------------
    # 2. Contrast enhancement
    # --------------------------------------------------------

    clahe = cv2.createCLAHE(
        clipLimit=2.0,
        tileGridSize=(8, 8)
    )

    enhanced = clahe.apply(gray)

    # --------------------------------------------------------
    # 3. Black-hat morphological transformation
    # --------------------------------------------------------

    kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT,
        (
            BLACKHAT_KERNEL_SIZE,
            BLACKHAT_KERNEL_SIZE
        )
    )

    blackhat = cv2.morphologyEx(
        enhanced,
        cv2.MORPH_BLACKHAT,
        kernel
    )

    # --------------------------------------------------------
    # 4. Thresholding
    # --------------------------------------------------------

    _, mask = cv2.threshold(
        blackhat,
        0,
        255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )

    # --------------------------------------------------------
    # 5. Morphological noise removal
    # --------------------------------------------------------

    morph_kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (MORPH_KERNEL_SIZE, MORPH_KERNEL_SIZE)
    )

    mask = cv2.morphologyEx(
        mask,
        cv2.MORPH_CLOSE,
        morph_kernel
    )

    # --------------------------------------------------------
    # 6. Connected component analysis
    # --------------------------------------------------------

    count, labels, stats, centroids = (
        cv2.connectedComponentsWithStats(
            mask,
            connectivity=8
        )
    )

    cracks = []

    for i in range(1, count):

        x = int(stats[i, cv2.CC_STAT_LEFT])
        y = int(stats[i, cv2.CC_STAT_TOP])

        width = int(stats[i, cv2.CC_STAT_WIDTH])
        height = int(stats[i, cv2.CC_STAT_HEIGHT])

        area = int(stats[i, cv2.CC_STAT_AREA])

        # Filter small noise and excessively large regions
        if area < MIN_CRACK_AREA:
            continue

        if area > MAX_CRACK_AREA:
            continue

        # ----------------------------------------------------
        # 7. Extract individual component
        # ----------------------------------------------------

        component = (labels == i).astype(np.uint8)

        # ----------------------------------------------------
        # 8. Skeletonization
        # ----------------------------------------------------

        skeleton = skeletonize(component > 0)

        skeleton_length = int(np.count_nonzero(skeleton))

        # ----------------------------------------------------
        # 9. Width estimation using distance transform
        # ----------------------------------------------------

        component_uint8 = component.astype(np.uint8)

        distance = cv2.distanceTransform(
            component_uint8,
            cv2.DIST_L2,
            5
        )

        skeleton_distances = distance[skeleton]

        if len(skeleton_distances) > 0:
            mean_width = float(
                np.mean(skeleton_distances) * 2
            )

            max_width = float(
                np.max(skeleton_distances) * 2
            )
        else:
            mean_width = 0.0
            max_width = 0.0

        # ----------------------------------------------------
        # 10. Orientation estimation using PCA
        # ----------------------------------------------------

        points = np.column_stack(
            np.where(component > 0)
        )

        if len(points) >= 2:

            mean, eigenvectors = cv2.PCACompute(
                points.astype(np.float32),
                mean=None
            )

            direction = eigenvectors[0]

            angle = np.degrees(
                np.arctan2(
                    direction[0],
                    direction[1]
                )
            )

        else:
            angle = 0.0

        # ----------------------------------------------------
        # 11. Centroid
        # ----------------------------------------------------

        centroid_x = float(centroids[i][0])
        centroid_y = float(centroids[i][1])

        # ----------------------------------------------------
        # 12. Physical measurement conversion
        # ----------------------------------------------------

        if pixels_per_mm and pixels_per_mm > 0:

            length_mm = skeleton_length / pixels_per_mm
            mean_width_mm = mean_width / pixels_per_mm
            max_width_mm = max_width / pixels_per_mm

        else:

            length_mm = None
            mean_width_mm = None
            max_width_mm = None

        # ----------------------------------------------------
        # 13. Crack record
        # ----------------------------------------------------

        crack = {
            "crack_id": len(cracks) + 1,

            "centroid": {
                "x": round(centroid_x, 2),
                "y": round(centroid_y, 2)
            },

            "bounding_box": {
                "x": x,
                "y": y,
                "width": width,
                "height": height
            },

            "area_px": area,

            "length_px": skeleton_length,

            "mean_width_px": round(mean_width, 2),

            "max_width_px": round(max_width, 2),

            "orientation_degrees": round(
                float(angle),
                2
            ),

            "length_mm": (
                round(length_mm, 3)
                if length_mm is not None
                else None
            ),

            "mean_width_mm": (
                round(mean_width_mm, 3)
                if mean_width_mm is not None
                else None
            ),

            "max_width_mm": (
                round(max_width_mm, 3)
                if max_width_mm is not None
                else None
            ),

            # Cannot be measured from a single ordinary RGB image
            "depth_mm": None
        }

        cracks.append(crack)

        # ----------------------------------------------------
        # 14. Draw detection on original image
        # ----------------------------------------------------

        cv2.rectangle(
            annotated,
            (x, y),
            (x + width, y + height),
            (0, 0, 255),
            2
        )

        cv2.circle(
            annotated,
            (
                int(centroid_x),
                int(centroid_y)
            ),
            4,
            (0, 255, 0),
            -1
        )

        cv2.putText(
            annotated,
            f"Crack {len(cracks)}",
            (x, max(y - 8, 15)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 0, 255),
            1
        )

    return original, mask, annotated, cracks


# ============================================================
# IMAGE PROCESSING AND DATABASE STORAGE
# ============================================================

def process_image(image_path, wall_id, pixels_per_mm=None):

    image_path = Path(image_path)

    original, mask, annotated, cracks = detect_cracks(
        image_path,
        pixels_per_mm
    )

    timestamp = datetime.now(timezone.utc)

    filename = safe_filename(image_path.name)

    timestamp_string = timestamp.strftime(
        "%Y%m%d_%H%M%S"
    )

    # --------------------------------------------------------
    # Save detection outputs
    # --------------------------------------------------------

    annotated_path = OUTPUT_DIR / (
        f"{filename}_{timestamp_string}_annotated.jpg"
    )

    mask_path = OUTPUT_DIR / (
        f"{filename}_{timestamp_string}_mask.png"
    )

    json_path = OUTPUT_DIR / (
        f"{filename}_{timestamp_string}.json"
    )

    cv2.imwrite(
        str(annotated_path),
        annotated
    )

    cv2.imwrite(
        str(mask_path),
        mask
    )

    image_hash = calculate_sha256(image_path)

    # --------------------------------------------------------
    # Build MongoDB document
    # --------------------------------------------------------

    document = {
        "wall_id": wall_id,

        "source_image": {
            "filename": image_path.name,
            "sha256": image_hash
        },

        "inspection": {
            "timestamp": timestamp,
            "method": "opencv_classical_image_processing",
            "image_width": int(original.shape[1]),
            "image_height": int(original.shape[0]),
            "pixels_per_mm": pixels_per_mm
        },

        "detection": {
            "total_cracks": len(cracks),
            "cracks": cracks
        },

        "outputs": {
            "annotated_image": str(annotated_path),
            "binary_mask": str(mask_path)
        }
    }

    # --------------------------------------------------------
    # Save JSON locally
    # --------------------------------------------------------

    with open(json_path, "w", encoding="utf-8") as file:

        json.dump(
            document,
            file,
            indent=4,
            default=str
        )

    # --------------------------------------------------------
    # Insert into MongoDB
    # --------------------------------------------------------

    client = None

    try:

        client, collection = get_database()

        result = collection.insert_one(document)

        mongo_id = str(result.inserted_id)

    finally:

        if client is not None:
            client.close()

    return {
        "cracks": cracks,
        "annotated": annotated,
        "mask": mask,
        "mongo_id": mongo_id,
        "annotated_path": str(annotated_path),
        "mask_path": str(mask_path),
        "json_path": str(json_path)
    }


# ============================================================
# TKINTER APPLICATION
# ============================================================

class CrackDetectionUI:

    def __init__(self, root):

        self.root = root

        self.root.title(
            "Wall Crack Detection System"
        )

        self.root.geometry("1200x800")
        self.root.minsize(900, 650)

        self.image_path = None
        self.original_photo = None
        self.annotated_photo = None

        self.build_interface()

    # --------------------------------------------------------
    # GUI construction
    # --------------------------------------------------------

    def build_interface(self):

        main_frame = ttk.Frame(
            self.root,
            padding=10
        )

        main_frame.pack(
            fill=tk.BOTH,
            expand=True
        )

        # ----------------------------------------------------
        # Title
        # ----------------------------------------------------

        title = ttk.Label(
            main_frame,
            text="Wall Crack Detection System",
            font=("Arial", 20, "bold")
        )

        title.pack(pady=10)

        # ----------------------------------------------------
        # Input controls
        # ----------------------------------------------------

        input_frame = ttk.LabelFrame(
            main_frame,
            text="Inspection Information",
            padding=10
        )

        input_frame.pack(
            fill=tk.X,
            pady=5
        )

        ttk.Label(
            input_frame,
            text="Wall ID:"
        ).grid(
            row=0,
            column=0,
            padx=5,
            pady=5
        )

        self.wall_id_entry = ttk.Entry(
            input_frame,
            width=25
        )

        self.wall_id_entry.grid(
            row=0,
            column=1,
            padx=5,
            pady=5
        )

        self.wall_id_entry.insert(
            0,
            "WALL-001"
        )

        ttk.Label(
            input_frame,
            text="Pixels per mm:"
        ).grid(
            row=0,
            column=2,
            padx=5,
            pady=5
        )

        self.calibration_entry = ttk.Entry(
            input_frame,
            width=15
        )

        self.calibration_entry.grid(
            row=0,
            column=3,
            padx=5,
            pady=5
        )

        self.calibration_entry.insert(
            0,
            ""
        )

        ttk.Label(
            input_frame,
            text="Leave blank if uncalibrated"
        ).grid(
            row=0,
            column=4,
            padx=5,
            pady=5
        )

        # ----------------------------------------------------
        # Buttons
        # ----------------------------------------------------

        button_frame = ttk.Frame(main_frame)

        button_frame.pack(
            fill=tk.X,
            pady=10
        )

        self.load_button = ttk.Button(
            button_frame,
            text="Load Image",
            command=self.load_image
        )

        self.load_button.pack(
            side=tk.LEFT,
            padx=5
        )

        self.detect_button = ttk.Button(
            button_frame,
            text="Detect & Save",
            command=self.start_detection,
            state=tk.DISABLED
        )

        self.detect_button.pack(
            side=tk.LEFT,
            padx=5
        )

        # ----------------------------------------------------
        # Image preview
        # ----------------------------------------------------

        preview_frame = ttk.Frame(main_frame)

        preview_frame.pack(
            fill=tk.BOTH,
            expand=True,
            pady=5
        )

        original_frame = ttk.LabelFrame(
            preview_frame,
            text="Original Image",
            padding=5
        )

        original_frame.pack(
            side=tk.LEFT,
            fill=tk.BOTH,
            expand=True,
            padx=5
        )

        annotated_frame = ttk.LabelFrame(
            preview_frame,
            text="Detected Cracks",
            padding=5
        )

        annotated_frame.pack(
            side=tk.LEFT,
            fill=tk.BOTH,
            expand=True,
            padx=5
        )

        self.original_label = ttk.Label(
            original_frame,
            text="No image loaded"
        )

        self.original_label.pack(
            fill=tk.BOTH,
            expand=True
        )

        self.annotated_label = ttk.Label(
            annotated_frame,
            text="Detection preview"
        )

        self.annotated_label.pack(
            fill=tk.BOTH,
            expand=True
        )

        # ----------------------------------------------------
        # Results
        # ----------------------------------------------------

        result_frame = ttk.LabelFrame(
            main_frame,
            text="Detection Results",
            padding=5
        )

        result_frame.pack(
            fill=tk.BOTH,
            expand=True,
            pady=5
        )

        self.result_text = tk.Text(
            result_frame,
            height=9,
            wrap=tk.WORD
        )

        self.result_text.pack(
            fill=tk.BOTH,
            expand=True
        )

        # ----------------------------------------------------
        # Status bar
        # ----------------------------------------------------

        self.status_label = ttk.Label(
            main_frame,
            text="Ready",
            relief=tk.SUNKEN,
            anchor=tk.W
        )

        self.status_label.pack(
            fill=tk.X,
            pady=5
        )

        self.progress = ttk.Progressbar(
            main_frame,
            mode="indeterminate"
        )

        self.progress.pack(
            fill=tk.X
        )

    # --------------------------------------------------------
    # Load image
    # --------------------------------------------------------

    def load_image(self):

        path = filedialog.askopenfilename(
            title="Select Wall Image",
            filetypes=[
                (
                    "Image files",
                    "*.jpg *.jpeg *.png *.bmp *.tif *.tiff"
                ),
                ("All files", "*.*")
            ]
        )

        if not path:
            return

        self.image_path = Path(path)

        image = cv2.imread(str(self.image_path))

        if image is None:

            messagebox.showerror(
                "Error",
                "Unable to load image."
            )

            return

        self.display_image(
            image,
            self.original_label,
            "original"
        )

        self.annotated_label.configure(
            image="",
            text="Detection preview"
        )

        self.annotated_photo = None

        self.detect_button.configure(
            state=tk.NORMAL
        )

        self.result_text.delete(
            "1.0",
            tk.END
        )

        self.result_text.insert(
            tk.END,
            f"Loaded image: {self.image_path.name}\n"
        )

        self.status_label.configure(
            text="Image loaded successfully."
        )

    # --------------------------------------------------------
    # Display image
    # --------------------------------------------------------

    def display_image(self, image, label, image_type):

        rgb = cv2.cvtColor(
            image,
            cv2.COLOR_BGR2RGB
        )

        pil_image = Image.fromarray(rgb)

        max_width = 520
        max_height = 350

        pil_image.thumbnail(
            (max_width, max_height)
        )

        photo = ImageTk.PhotoImage(pil_image)

        label.configure(
            image=photo,
            text=""
        )

        if image_type == "original":

            self.original_photo = photo

        else:

            self.annotated_photo = photo

    # --------------------------------------------------------
    # Start detection in background thread
    # --------------------------------------------------------

    def start_detection(self):

        if self.image_path is None:
            return

        wall_id = self.wall_id_entry.get().strip()

        if not wall_id:

            messagebox.showwarning(
                "Missing information",
                "Please enter a Wall ID."
            )

            return

        calibration_text = (
            self.calibration_entry.get().strip()
        )

        pixels_per_mm = None

        if calibration_text:

            try:

                pixels_per_mm = float(
                    calibration_text
                )

                if pixels_per_mm <= 0:
                    raise ValueError

            except ValueError:

                messagebox.showerror(
                    "Invalid calibration",
                    "Pixels per mm must be a positive number."
                )

                return

        self.detect_button.configure(
            state=tk.DISABLED
        )

        self.load_button.configure(
            state=tk.DISABLED
        )

        self.progress.start(10)

        self.status_label.configure(
            text="Detecting cracks and saving to MongoDB..."
        )

        thread = threading.Thread(
            target=self.detection_worker,
            args=(
                str(self.image_path),
                wall_id,
                pixels_per_mm
            ),
            daemon=True
        )

        thread.start()

    # --------------------------------------------------------
    # Background processing
    # --------------------------------------------------------

    def detection_worker(
        self,
        image_path,
        wall_id,
        pixels_per_mm
    ):

        try:

            result = process_image(
                image_path,
                wall_id,
                pixels_per_mm
            )

            self.root.after(
                0,
                lambda: self.detection_success(result)
            )

        except Exception as error:

            self.root.after(
                0,
                lambda e=str(error): self.detection_error(e)
            )

    # --------------------------------------------------------
    # Detection success
    # --------------------------------------------------------

    def detection_success(self, result):

        self.progress.stop()

        self.detect_button.configure(
            state=tk.NORMAL
        )

        self.load_button.configure(
            state=tk.NORMAL
        )

        self.display_image(
            result["annotated"],
            self.annotated_label,
            "annotated"
        )

        cracks = result["cracks"]

        self.result_text.delete(
            "1.0",
            tk.END
        )

        self.result_text.insert(
            tk.END,
            "DETECTION COMPLETED\n"
        )

        self.result_text.insert(
            tk.END,
            "==============================\n\n"
        )

        self.result_text.insert(
            tk.END,
            f"Total cracks detected: {len(cracks)}\n"
        )

        self.result_text.insert(
            tk.END,
            f"MongoDB document ID: {result['mongo_id']}\n\n"
        )

        for crack in cracks:

            self.result_text.insert(
                tk.END,
                f"Crack #{crack['crack_id']}\n"
            )

            self.result_text.insert(
                tk.END,
                f"  Centroid: {crack['centroid']}\n"
            )

            self.result_text.insert(
                tk.END,
                f"  Bounding box: {crack['bounding_box']}\n"
            )

            self.result_text.insert(
                tk.END,
                f"  Length: {crack['length_px']} px\n"
            )

            self.result_text.insert(
                tk.END,
                f"  Mean width: {crack['mean_width_px']} px\n"
            )

            self.result_text.insert(
                tk.END,
                f"  Maximum width: {crack['max_width_px']} px\n"
            )

            self.result_text.insert(
                tk.END,
                f"  Orientation: {crack['orientation_degrees']} degrees\n"
            )

            self.result_text.insert(
                tk.END,
                f"  Depth: Not measurable from this image\n\n"
            )

        if not cracks:

            self.result_text.insert(
                tk.END,
                "No cracks detected.\n"
            )

        self.result_text.insert(
            tk.END,
            "\nOUTPUT FILES\n"
        )

        self.result_text.insert(
            tk.END,
            f"Annotated: {result['annotated_path']}\n"
        )

        self.result_text.insert(
            tk.END,
            f"Mask: {result['mask_path']}\n"
        )

        self.result_text.insert(
            tk.END,
            f"JSON: {result['json_path']}\n"
        )

        self.status_label.configure(
            text="Detection completed. Results saved to MongoDB."
        )

    # --------------------------------------------------------
    # Detection error
    # --------------------------------------------------------

    def detection_error(self, error):

        self.progress.stop()

        self.detect_button.configure(
            state=tk.NORMAL
        )

        self.load_button.configure(
            state=tk.NORMAL
        )

        self.status_label.configure(
            text="Detection failed."
        )

        messagebox.showerror(
            "Detection Error",
            str(error)
        )


# ============================================================
# APPLICATION ENTRY POINT
# ============================================================

if __name__ == "__main__":

    root = tk.Tk()

    app = CrackDetectionUI(root)

    root.mainloop()
