"""Wall Crack Detector - entry point.

Implements the MAD-based adaptive threshold method from:
Avendaño, Leander & Karoumi, "Image-Based Concrete Crack Detection Method
Using the Median Absolute Deviation", Sensors 2024, 24, 2736.
"""
import sys

from PyQt6.QtWidgets import QApplication

from controllers.main_controller import MainController
from models.image_model import ImageModel
from views.main_window import MainWindow


def main() -> int:
    app = QApplication(sys.argv)
    model = ImageModel()
    view = MainWindow()
    controller = MainController(model, view)  # noqa: F841 (keeps references alive)
    view.show()
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
