"""Dialog to choose what the bot captures: full screen, a screen region or a game window."""
import logging
import os
import sys
import time

from PIL.ImageQt import ImageQt
from PyQt6 import QtCore, QtGui, QtWidgets
from PyQt6.QtCore import Qt

from poker.tools.capture_target import (MIN_SIZE, MODE_FULL_SCREEN, MODE_REGION, MODE_WINDOW, CaptureTarget,
                                        CaptureTargetError, grab_image, list_windows)

# pylint: disable=logging-fstring-interpolation,too-many-instance-attributes,too-many-statements,too-many-locals
# pylint: disable=c-extension-no-member

log = logging.getLogger(__name__)

PICK_REGION = 'region'
PICK_WINDOW = 'window'
HIDE_DELAY_MS = 300  # time for the screen to repaint after hiding our windows
PREVIEW_SIZE = (420, 260)


def pil_to_pixmap(image):
    return QtGui.QPixmap.fromImage(ImageQt(image.convert('RGBA')).copy())


class ScreenPicker(QtWidgets.QWidget):
    """
    Full screen overlay showing a frozen screenshot.

    In region mode the user drags a rectangle, in window mode the user clicks a window.
    Emits ``picked`` with a region tuple or a WindowInfo, or ``cancelled``.
    """
    picked = QtCore.pyqtSignal(object)
    cancelled = QtCore.pyqtSignal()

    def __init__(self, screenshot, windows, mode, screen=None):
        super().__init__(None, Qt.WindowType.FramelessWindowHint | Qt.WindowType.WindowStaysOnTopHint)
        self.mode = mode
        self.windows = windows
        self.pixmap = pil_to_pixmap(screenshot)
        screen = screen or QtGui.QGuiApplication.primaryScreen()
        geometry = screen.geometry()
        # Coordinates used for capturing: physical pixels (= screenshot pixels) on Windows and Linux,
        # points on macOS (window bounds and screencapture regions are given in points there).
        if sys.platform == 'darwin':
            self.capture_size = (geometry.width(), geometry.height())
        else:
            self.capture_size = screenshot.size
        self.start = None
        self.current = None
        self.hover = None
        self.setGeometry(geometry)
        self.setCursor(Qt.CursorShape.CrossCursor)
        self.setMouseTracking(True)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)

    # coordinate conversion between widget (logical pixels) and capture coordinates
    def _scale(self):
        return self.capture_size[0] / max(1, self.width()), self.capture_size[1] / max(1, self.height())

    def to_capture(self, point):
        sx, sy = self._scale()
        return int(round(point.x() * sx)), int(round(point.y() * sy))

    def to_widget_rect(self, rect):
        sx, sy = self._scale()
        left, top, right, bottom = rect
        return QtCore.QRectF(left / sx, top / sy, (right - left) / sx, (bottom - top) / sy)

    def selected_region(self):
        if self.start is None or self.current is None:
            return None
        x1, y1 = self.to_capture(self.start)
        x2, y2 = self.to_capture(self.current)
        return min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2)

    def window_at(self, point):
        x, y = self.to_capture(point)
        for window in self.windows:  # top most first
            if window.contains(x, y):
                return window
        return None

    # events
    def showEvent(self, event):  # pylint: disable=invalid-name
        super().showEvent(event)
        self.activateWindow()
        self.setFocus()

    def keyPressEvent(self, event):  # pylint: disable=invalid-name
        if event.key() == Qt.Key.Key_Escape:
            self.close()
            self.cancelled.emit()

    def mousePressEvent(self, event):  # pylint: disable=invalid-name
        if event.button() == Qt.MouseButton.RightButton:
            self.close()
            self.cancelled.emit()
            return
        if event.button() != Qt.MouseButton.LeftButton:
            return
        if self.mode == PICK_REGION:
            self.start = self.current = event.position()
        else:
            window = self.window_at(event.position())
            if window is not None:
                self.close()
                self.picked.emit(window)
        self.update()

    def mouseMoveEvent(self, event):  # pylint: disable=invalid-name
        self.current = event.position()
        if self.mode == PICK_WINDOW:
            self.hover = self.window_at(self.current)
        self.update()

    def mouseReleaseEvent(self, event):  # pylint: disable=invalid-name
        if self.mode != PICK_REGION or event.button() != Qt.MouseButton.LeftButton or self.start is None:
            return
        self.current = event.position()
        region = self.selected_region()
        if region[2] - region[0] < MIN_SIZE or region[3] - region[1] < MIN_SIZE:
            self.start = self.current = None  # a click, not a drag
            self.update()
            return
        self.close()
        self.picked.emit(region)

    def paintEvent(self, _event):  # pylint: disable=invalid-name
        painter = QtGui.QPainter(self)
        painter.setRenderHint(QtGui.QPainter.RenderHint.SmoothPixmapTransform)
        full = QtCore.QRectF(self.rect())
        painter.drawPixmap(full, self.pixmap, QtCore.QRectF(self.pixmap.rect()))
        painter.fillRect(full, QtGui.QColor(0, 0, 0, 110))

        highlight = None
        label = ''
        if self.mode == PICK_REGION:
            region = self.selected_region()
            if region is not None:
                highlight = self.to_widget_rect(region)
                x1, y1, x2, y2 = region
                label = f"{x1},{y1} - {x2},{y2}  ({x2 - x1} x {y2 - y1})"
            hint = "Drag to select the capture region.  Esc or right click: cancel"
        else:
            if self.hover is not None:
                highlight = self.to_widget_rect(self.hover.rect)
                label = f"{self.hover.title}  ({self.hover.width} x {self.hover.height})"
            hint = "Click the game window.  Esc or right click: cancel"

        if highlight is not None:
            sx, sy = self.pixmap.width() / max(1, self.width()), self.pixmap.height() / max(1, self.height())
            source = QtCore.QRectF(highlight.x() * sx, highlight.y() * sy, highlight.width() * sx,
                                   highlight.height() * sy)
            painter.drawPixmap(highlight, self.pixmap, source)
            painter.setPen(QtGui.QPen(QtGui.QColor(0, 200, 255), 2))
            painter.drawRect(highlight)
            self._draw_label(painter, label, highlight.topLeft() + QtCore.QPointF(0, -6))

        self._draw_label(painter, hint, QtCore.QPointF(16, 30))
        if self.mode == PICK_REGION and self.current is not None and self.start is None:
            x, y = self.to_capture(self.current)
            self._draw_label(painter, f"{x}, {y}", self.current + QtCore.QPointF(14, 24))
        painter.end()

    def _draw_label(self, painter, text, position):
        if not text:
            return
        metrics = painter.fontMetrics()
        width = metrics.horizontalAdvance(text) + 12
        height = metrics.height() + 6
        x = min(max(2.0, position.x()), self.width() - width - 2)
        y = min(max(height + 2.0, position.y()), self.height() - 2.0)
        box = QtCore.QRectF(x, y - height, width, height)
        painter.fillRect(box, QtGui.QColor(20, 20, 20, 220))
        painter.setPen(QtGui.QColor(255, 255, 255))
        painter.drawText(box, Qt.AlignmentFlag.AlignCenter, text)


class CaptureTargetDialog(QtWidgets.QDialog):
    """
    Choose and save the capture target. Emits ``saved`` with the saved CaptureTarget.

    The dialog is not modal: it hides itself (and the bot's other windows) while the user picks
    on screen, which would end a modal exec() loop.
    """
    saved = QtCore.pyqtSignal(object)

    def __init__(self, parent=None, target=None):
        super().__init__(parent)
        self.setWindowTitle("Capture area")
        self.target = target or CaptureTarget.from_config()
        self.windows = []
        self.picker = None
        self.hidden_widgets = []
        self._build_ui()
        self._load(self.target)
        self.refresh_windows()

    # ui
    def _build_ui(self):
        layout = QtWidgets.QVBoxLayout(self)
        intro = QtWidgets.QLabel(
            "Choose what the bot captures. The table is only searched inside this area, "
            "so a smaller area is faster. Mouse clicks are adjusted to the area's position.")
        intro.setWordWrap(True)
        layout.addWidget(intro)

        self.radio_full = QtWidgets.QRadioButton("Full screen")
        self.radio_region = QtWidgets.QRadioButton("Screen region (coordinates)")
        self.radio_window = QtWidgets.QRadioButton("Game window")
        self.mode_group = QtWidgets.QButtonGroup(self)
        for radio in (self.radio_full, self.radio_region, self.radio_window):
            self.mode_group.addButton(radio)
        layout.addWidget(self.radio_full)

        layout.addWidget(self.radio_region)
        region_row = QtWidgets.QHBoxLayout()
        self.region_boxes = []
        for name in ('x1', 'y1', 'x2', 'y2'):
            box = QtWidgets.QSpinBox()
            box.setRange(-20000, 20000)
            box.setPrefix(f"{name}: ")
            box.valueChanged.connect(lambda _value: self.radio_region.setChecked(True))
            self.region_boxes.append(box)
            region_row.addWidget(box)
        self.select_region_button = QtWidgets.QPushButton("Select on screen...")
        self.select_region_button.clicked.connect(lambda: self.start_picker(PICK_REGION))
        region_row.addWidget(self.select_region_button)
        layout.addLayout(region_row)

        layout.addWidget(self.radio_window)
        window_row = QtWidgets.QHBoxLayout()
        self.window_combo = QtWidgets.QComboBox()
        self.window_combo.setMinimumWidth(280)
        self.window_combo.activated.connect(self._window_chosen)
        window_row.addWidget(self.window_combo, 1)
        refresh = QtWidgets.QPushButton("Refresh")
        refresh.clicked.connect(self.refresh_windows)
        window_row.addWidget(refresh)
        self.pick_window_button = QtWidgets.QPushButton("Pick on screen...")
        self.pick_window_button.clicked.connect(lambda: self.start_picker(PICK_WINDOW))
        window_row.addWidget(self.pick_window_button)
        layout.addLayout(window_row)
        title_row = QtWidgets.QHBoxLayout()
        title_row.addWidget(QtWidgets.QLabel("Title contains:"))
        self.title_edit = QtWidgets.QLineEdit()
        self.title_edit.setToolTip("The window whose title contains this text is captured. Shorten it to the part "
                                   "that does not change, e.g. the game name without the table number.")
        self.title_edit.textEdited.connect(lambda _text: self.radio_window.setChecked(True))
        title_row.addWidget(self.title_edit, 1)
        layout.addLayout(title_row)

        preview_row = QtWidgets.QHBoxLayout()
        self.test_button = QtWidgets.QPushButton("Test capture")
        self.test_button.clicked.connect(self.test_capture)
        preview_row.addWidget(self.test_button)
        self.status_label = QtWidgets.QLabel("")
        self.status_label.setWordWrap(True)
        preview_row.addWidget(self.status_label, 1)
        layout.addLayout(preview_row)
        self.preview_label = QtWidgets.QLabel()
        self.preview_label.setMinimumSize(*PREVIEW_SIZE)
        self.preview_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.preview_label.setStyleSheet("background: #222; color: #aaa;")
        self.preview_label.setText("Press 'Test capture' to see what the bot will capture")
        layout.addWidget(self.preview_label)

        buttons = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.StandardButton.Save
                                             | QtWidgets.QDialogButtonBox.StandardButton.Cancel)
        buttons.accepted.connect(self.save)
        buttons.rejected.connect(self.close)
        layout.addWidget(buttons)

    def _load(self, target):
        {MODE_FULL_SCREEN: self.radio_full, MODE_REGION: self.radio_region,
         MODE_WINDOW: self.radio_window}[target.mode].setChecked(True)
        if target.region:
            self.set_region(target.region, check=False)
        self.title_edit.setText(target.window_title)

    def set_region(self, region, check=True):
        for box, value in zip(self.region_boxes, region):
            box.blockSignals(True)
            box.setValue(int(value))
            box.blockSignals(False)
        if check:
            self.radio_region.setChecked(True)

    def refresh_windows(self):
        self.windows = list_windows(exclude_pid=os.getpid())
        self.window_combo.clear()
        for window in self.windows:
            self.window_combo.addItem(f"{window.title}  ({window.width} x {window.height})")
        if not self.windows:
            self.window_combo.addItem("No windows found")

    def _window_chosen(self, index):
        if 0 <= index < len(self.windows):
            self.title_edit.setText(self.windows[index].title)
            self.radio_window.setChecked(True)

    def current_target(self):
        """CaptureTarget from the dialog's inputs. Raises ValueError when they are invalid."""
        if self.radio_region.isChecked():
            return CaptureTarget(MODE_REGION, region=tuple(box.value() for box in self.region_boxes),
                                 window_title=self.title_edit.text().strip())
        if self.radio_window.isChecked():
            title = self.title_edit.text().strip()
            if not title:
                raise ValueError("Choose a window or enter part of its title.")
            region = tuple(box.value() for box in self.region_boxes)
            try:
                return CaptureTarget(MODE_WINDOW, region=region, window_title=title)
            except ValueError:
                return CaptureTarget(MODE_WINDOW, window_title=title)
        return CaptureTarget(MODE_FULL_SCREEN)

    # picking on screen
    def _hide_own_windows(self):
        self.hidden_widgets = [w for w in QtWidgets.QApplication.topLevelWidgets()
                               if w.isVisible() and not isinstance(w, ScreenPicker)]
        for widget in self.hidden_widgets:
            widget.hide()

    def _restore_own_windows(self):
        for widget in self.hidden_widgets:
            widget.show()
        self.hidden_widgets = []
        self.raise_()
        self.activateWindow()

    def start_picker(self, mode):
        self._hide_own_windows()
        QtCore.QTimer.singleShot(HIDE_DELAY_MS, lambda: self._show_picker(mode))

    def _show_picker(self, mode):
        try:
            screenshot = grab_image()
            windows = list_windows(exclude_pid=os.getpid()) if mode == PICK_WINDOW else []
        except Exception as exc:  # pylint: disable=broad-except
            self._restore_own_windows()
            QtWidgets.QMessageBox.warning(self, "Capture area", f"Unable to capture the screen: {exc}")
            return
        if mode == PICK_WINDOW and not windows:
            self._restore_own_windows()
            QtWidgets.QMessageBox.warning(self, "Capture area", "No windows found on this system.")
            return
        self.picker = ScreenPicker(screenshot, windows, mode)
        self.picker.picked.connect(self._picked)
        self.picker.cancelled.connect(self._restore_own_windows)
        self.picker.showFullScreen()

    def _picked(self, result):
        self._restore_own_windows()
        if isinstance(result, tuple):
            self.set_region(result)
        else:
            self.refresh_windows()
            self.title_edit.setText(result.title)
            self.radio_window.setChecked(True)
            for i, window in enumerate(self.windows):
                if window.handle == result.handle:
                    self.window_combo.setCurrentIndex(i)
        self.test_capture()

    # preview and save
    def test_capture(self):
        try:
            target = self.current_target()
            bbox = target.resolve_bbox(self.windows if target.mode == MODE_WINDOW else None)
        except (ValueError, CaptureTargetError) as exc:
            self.status_label.setText(str(exc))
            return
        self._hide_own_windows()
        QtCore.QTimer.singleShot(HIDE_DELAY_MS, lambda: self._test_capture(target, bbox))

    def _test_capture(self, target, bbox):
        try:
            start = time.perf_counter()
            image = grab_image(bbox)
            elapsed = time.perf_counter() - start
        except Exception as exc:  # pylint: disable=broad-except
            self._restore_own_windows()
            self.status_label.setText(f"Capture failed: {exc}")
            return
        self._restore_own_windows()
        pixmap = pil_to_pixmap(image)
        area = self.preview_label.size()
        self.preview_label.setPixmap(pixmap.scaled(min(area.width(), pixmap.width()), min(area.height(), pixmap.height()),
                                                   Qt.AspectRatioMode.KeepAspectRatio,
                                                   Qt.TransformationMode.SmoothTransformation))
        position = f" at {bbox[0]},{bbox[1]}" if bbox else ""
        self.status_label.setText(f"{target.describe()}: {image.size[0]} x {image.size[1]}{position}, "
                                  f"captured in {elapsed * 1000:.0f} ms")

    def save(self):
        try:
            target = self.current_target()
        except ValueError as exc:
            QtWidgets.QMessageBox.warning(self, "Capture area", str(exc))
            return
        if target.mode == MODE_WINDOW and not self._window_exists(target):
            answer = QtWidgets.QMessageBox.question(
                self, "Capture area",
                f"No visible window contains '{target.window_title}' in its title right now. Save anyway?")
            if answer != QtWidgets.QMessageBox.StandardButton.Yes:
                return
        target.save()
        self.target = target
        log.info(f"Capture area saved: {target.describe()}")
        self.saved.emit(target)
        self.close()

    def _window_exists(self, target):
        try:
            target.resolve_bbox()
            return True
        except CaptureTargetError:
            return False


def open_capture_target_dialog(parent=None, on_saved=None):
    """Show the dialog. `on_saved` is called with the saved CaptureTarget. Keep the returned dialog referenced."""
    dialog = CaptureTargetDialog(parent)
    if on_saved is not None:
        dialog.saved.connect(on_saved)
    dialog.show()
    return dialog
