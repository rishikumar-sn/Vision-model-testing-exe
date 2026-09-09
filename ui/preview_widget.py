from __future__ import annotations

from typing import Optional

import cv2
import numpy as np
from PyQt6.QtCore import QPointF, QRectF, Qt, pyqtSignal
from PyQt6.QtGui import QColor, QFont, QImage, QPainter, QPen
from PyQt6.QtWidgets import QSizePolicy, QWidget

class RoiPreviewWidget(QWidget):
    roi_changed = pyqtSignal(object)  # normalized QRectF or None

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.setMouseTracking(True)
        self.setMinimumSize(720, 480)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)

        self._image: Optional[QImage] = None
        self._image_width = 0
        self._image_height = 0

        # ROI is stored in normalized image coordinates [0,1]. This makes it persist
        # even if the preview widget changes size or the camera is restarted.
        self._roi_norm: Optional[QRectF] = None
        self._drag_start_norm: Optional[QPointF] = None
        self._dragging = False

    @property
    def roi_norm(self) -> Optional[QRectF]:
        return QRectF(self._roi_norm) if self._roi_norm is not None else None

    def has_roi(self) -> bool:
        return self._roi_norm is not None and self._roi_norm.width() > 0 and self._roi_norm.height() > 0

    def clear_roi(self) -> None:
        self._roi_norm = None
        self._drag_start_norm = None
        self._dragging = False
        self.roi_changed.emit(None)
        self.update()

    def set_full_frame_roi(self) -> None:
        self._roi_norm = QRectF(0.0, 0.0, 1.0, 1.0)
        self.roi_changed.emit(self.roi_norm)
        self.update()

    def set_frame(self, frame_bgr: np.ndarray) -> None:
        if frame_bgr is None or frame_bgr.size == 0:
            return

        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        h, w, channels = rgb.shape
        bytes_per_line = channels * w
        # copy() detaches QImage storage from the temporary NumPy buffer.
        self._image = QImage(
            rgb.data,
            w,
            h,
            bytes_per_line,
            QImage.Format.Format_RGB888,
        ).copy()
        self._image_width = w
        self._image_height = h
        self.update()

    def _image_target_rect(self) -> QRectF:
        if self._image is None or self._image_width <= 0 or self._image_height <= 0:
            return QRectF()

        widget_w = float(self.width())
        widget_h = float(self.height())
        image_aspect = self._image_width / self._image_height
        widget_aspect = widget_w / widget_h if widget_h > 0 else image_aspect

        if widget_aspect > image_aspect:
            draw_h = widget_h
            draw_w = draw_h * image_aspect
            x = (widget_w - draw_w) / 2.0
            y = 0.0
        else:
            draw_w = widget_w
            draw_h = draw_w / image_aspect
            x = 0.0
            y = (widget_h - draw_h) / 2.0

        return QRectF(x, y, draw_w, draw_h)

    def _widget_to_norm(self, point: QPointF) -> Optional[QPointF]:
        rect = self._image_target_rect()
        if rect.isNull() or not rect.contains(point):
            return None

        nx = (point.x() - rect.left()) / rect.width()
        ny = (point.y() - rect.top()) / rect.height()
        nx = max(0.0, min(1.0, nx))
        ny = max(0.0, min(1.0, ny))
        return QPointF(nx, ny)

    def _norm_rect_to_widget(self, norm_rect: QRectF) -> QRectF:
        target = self._image_target_rect()
        return QRectF(
            target.left() + norm_rect.left() * target.width(),
            target.top() + norm_rect.top() * target.height(),
            norm_rect.width() * target.width(),
            norm_rect.height() * target.height(),
        )

    def paintEvent(self, event) -> None:  # noqa: N802 (Qt API naming)
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform, True)
        painter.fillRect(self.rect(), QColor(18, 20, 24))

        if self._image is None:
            painter.setPen(QColor(180, 184, 191))
            painter.setFont(QFont("Arial", 14))
            painter.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter, "Waiting for camera...")
            return

        target = self._image_target_rect()
        painter.drawImage(target, self._image)

        # Dim the preview outside the ROI so the selected crop is visually clear.
        if self._roi_norm is not None:
            roi_widget = self._norm_rect_to_widget(self._roi_norm)

            painter.save()
            painter.setBrush(QColor(0, 0, 0, 105))
            painter.setPen(Qt.PenStyle.NoPen)

            # Four rectangles around the ROI.
            painter.drawRect(QRectF(target.left(), target.top(), target.width(), max(0.0, roi_widget.top() - target.top())))
            painter.drawRect(QRectF(target.left(), roi_widget.bottom(), target.width(), max(0.0, target.bottom() - roi_widget.bottom())))
            painter.drawRect(QRectF(target.left(), roi_widget.top(), max(0.0, roi_widget.left() - target.left()), roi_widget.height()))
            painter.drawRect(QRectF(roi_widget.right(), roi_widget.top(), max(0.0, target.right() - roi_widget.right()), roi_widget.height()))
            painter.restore()

            pen = QPen(QColor(0, 220, 170), 3)
            painter.setPen(pen)
            painter.setBrush(Qt.BrushStyle.NoBrush)
            painter.drawRect(roi_widget)

            # ROI label.
            if self._image_width and self._image_height:
                rw = int(round(self._roi_norm.width() * self._image_width))
                rh = int(round(self._roi_norm.height() * self._image_height))
                label = f"ROI {rw} × {rh}"
                label_rect = QRectF(roi_widget.left(), max(target.top(), roi_widget.top() - 29), 145, 26)
                painter.fillRect(label_rect, QColor(0, 0, 0, 175))
                painter.setPen(QColor(255, 255, 255))
                painter.drawText(label_rect, Qt.AlignmentFlag.AlignCenter, label)

    def mousePressEvent(self, event) -> None:  # noqa: N802
        if event.button() != Qt.MouseButton.LeftButton or self._image is None:
            return

        norm = self._widget_to_norm(event.position())
        if norm is None:
            return

        self._drag_start_norm = norm
        self._roi_norm = QRectF(norm, norm)
        self._dragging = True
        self.update()

    def mouseMoveEvent(self, event) -> None:  # noqa: N802
        if not self._dragging or self._drag_start_norm is None:
            return

        # Clamp dragging even if cursor leaves the displayed image area.
        target = self._image_target_rect()
        px = max(target.left(), min(target.right(), event.position().x()))
        py = max(target.top(), min(target.bottom(), event.position().y()))
        current = self._widget_to_norm(QPointF(px, py))
        if current is None:
            return

        rect = QRectF(self._drag_start_norm, current).normalized()
        self._roi_norm = QRectF(
            max(0.0, min(1.0, rect.left())),
            max(0.0, min(1.0, rect.top())),
            max(0.0, min(1.0, rect.width())),
            max(0.0, min(1.0, rect.height())),
        )
        self.roi_changed.emit(self.roi_norm)
        self.update()

    def mouseReleaseEvent(self, event) -> None:  # noqa: N802
        if event.button() != Qt.MouseButton.LeftButton or not self._dragging:
            return

        self._dragging = False
        if self._roi_norm is None:
            return

        # Ignore tiny accidental clicks/drags.
        if self._roi_norm.width() < 0.005 or self._roi_norm.height() < 0.005:
            self._roi_norm = None
            self.roi_changed.emit(None)
        else:
            self.roi_changed.emit(self.roi_norm)
        self.update()


