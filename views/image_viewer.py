"""Zoomable / pannable image widget."""
from __future__ import annotations

import numpy as np
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QImage, QPainter, QPixmap
from PyQt6.QtWidgets import QGraphicsPixmapItem, QGraphicsScene, QGraphicsView


class ImageViewer(QGraphicsView):
    def __init__(self) -> None:
        super().__init__()
        self._scene = QGraphicsScene(self)
        self.setScene(self._scene)
        self._item: QGraphicsPixmapItem | None = None
        self.setDragMode(QGraphicsView.DragMode.ScrollHandDrag)
        self.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform)
        self.setTransformationAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        self.setBackgroundBrush(Qt.GlobalColor.darkGray)

    def set_rgb(self, rgb: np.ndarray, fit: bool = False) -> None:
        h, w, _ = rgb.shape
        rgb = np.ascontiguousarray(rgb)
        qimg = QImage(rgb.data, w, h, 3 * w, QImage.Format.Format_RGB888).copy()
        pix = QPixmap.fromImage(qimg)
        first = self._item is None
        if first:
            self._item = self._scene.addPixmap(pix)
        else:
            self._item.setPixmap(pix)
        self._scene.setSceneRect(self._item.boundingRect())
        if first or fit:
            self.fit()

    def fit(self) -> None:
        if self._item:
            self.fitInView(self._item, Qt.AspectRatioMode.KeepAspectRatio)

    def wheelEvent(self, event) -> None:
        factor = 1.25 if event.angleDelta().y() > 0 else 0.8
        self.scale(factor, factor)
