"""Check the capture area feature on a real desktop (Windows, macOS or Linux).

Opens a stand-in game window that shows a table screenshot from poker/tests/screenshots at a known
screen position. The window runs in a separate process, like a real game client. Then checks:

1. listing: the window is found by its title, with the right position and size
2. capture: capturing the window returns the screenshot pixel for pixel
3. recognition: the bot's recognition, run through the real screen capture, finds the expected cards
   and amounts in window mode and in full screen mode, and the top left corner in screen coordinates
   (used for mouse clicks) equals the window position
4. moved window: after the window reopens at another position, window mode follows it
5. missing window: when the window is gone, the full screen is captured instead
6. timing: capture time and recognition time in both modes

No mouse clicks are made and config.ini is not written.

Usage (from the repository root)::

    python -m poker.tools.capture_check
    python -m poker.tools.capture_check --repeat 10 --out capture_check_output
    python -m poker.tools.capture_check --window-only 300   # only show the stand-in window for 300 seconds
"""
import argparse
import ctypes
import json
import logging
import os
import platform
import statistics
import subprocess
import sys
import time

import numpy as np
from PIL import Image, ImageChops

from poker.tools import capture_target
from poker.tools.capture_target import (MODE_FULL_SCREEN, MODE_WINDOW, CaptureTarget, CaptureTargetError,
                                        ensure_dpi_aware, find_window)
from poker.tools.helper import get_dir

# pylint: disable=import-outside-toplevel,too-many-locals,too-many-statements,c-extension-no-member

WINDOW_TITLE = "Capture Check - Fake Hold'em Table $0.01/$0.02"
POSITIONS = ((300, 150), (420, 230))
WAIT_FOR_WINDOW = 20  # seconds
PIXEL_TOLERANCE = 8  # per channel, for colour management differences

log = logging.getLogger(__name__)


# ---------------------------------------------------------------- stand-in game window (child process)

def serve_window(image_path, title, x, y, seconds):
    """Show image_path as a frameless always-on-top window at (x, y). Runs in its own process."""
    # one image pixel = one screen pixel, also with Windows display scaling above 100 %
    os.environ['QT_ENABLE_HIGHDPI_SCALING'] = '0'
    from PyQt6 import QtCore, QtGui, QtWidgets

    app = QtWidgets.QApplication([sys.argv[0]])
    label = QtWidgets.QLabel()
    label.setWindowFlags(QtCore.Qt.WindowType.FramelessWindowHint | QtCore.Qt.WindowType.WindowStaysOnTopHint)
    pixmap = QtGui.QPixmap(image_path)
    label.setPixmap(pixmap)
    label.setWindowTitle(title)
    label.setGeometry(x, y, pixmap.width(), pixmap.height())
    label.show()
    if seconds:
        QtCore.QTimer.singleShot(int(seconds * 1000), app.quit)
    app.exec()


class StandInWindow:
    """Starts the stand-in game window in a separate process and waits until it is listed."""

    def __init__(self, image_path, position, seconds=0):
        self.image_path = image_path
        self.position = position
        self.seconds = seconds
        self.process = None

    def __enter__(self):
        repo_root = os.path.dirname(get_dir('codebase'))
        env = dict(os.environ, PYTHONPATH=repo_root + os.pathsep + os.environ.get('PYTHONPATH', ''))
        self.process = subprocess.Popen(  # pylint: disable=consider-using-with
            [sys.executable, '-m', 'poker.tools.capture_check', '--serve-window', self.image_path, WINDOW_TITLE,
             str(self.position[0]), str(self.position[1]), str(self.seconds)], cwd=repo_root, env=env)
        deadline = time.time() + WAIT_FOR_WINDOW
        while time.time() < deadline:
            if self.process.poll() is not None:
                raise RuntimeError(f"The stand-in window process ended with code {self.process.returncode}")
            if find_window(WINDOW_TITLE) is not None:
                time.sleep(1.0)  # let it paint
                return self
            time.sleep(0.2)
        self.close()
        raise RuntimeError(f"The stand-in window did not appear within {WAIT_FOR_WINDOW} s")

    def close(self):
        if self.process and self.process.poll() is None:
            self.process.terminate()
            try:
                self.process.wait(10)
            except subprocess.TimeoutExpired:
                self.process.kill()
        deadline = time.time() + 10
        while time.time() < deadline and find_window(WINDOW_TITLE) is not None:
            time.sleep(0.2)

    def __exit__(self, *_exc):
        self.close()


# ---------------------------------------------------------------- checks

class Report:
    def __init__(self, out_dir):
        self.out_dir = out_dir
        self.checks = []
        self.timing = {}
        self.environment = {}

    def add(self, name, status, detail=''):
        self.checks.append({'name': name, 'status': status, 'detail': detail})
        print(f"[{status}] {name}: {detail}", flush=True)

    def save_image(self, image, name):
        os.makedirs(self.out_dir, exist_ok=True)
        path = os.path.join(self.out_dir, name)
        image.save(path)
        return path

    def failed(self):
        return any(c['status'] == 'FAIL' for c in self.checks)

    def write(self):
        os.makedirs(self.out_dir, exist_ok=True)
        path = os.path.join(self.out_dir, 'result.json')
        with open(path, 'w', encoding='utf-8') as fh:
            json.dump({'environment': self.environment, 'checks': self.checks, 'timing': self.timing}, fh,
                      indent=2, default=str)
        return path


def environment_info():
    info = {'platform': platform.platform(), 'python': sys.version.split()[0],
            'processor': platform.processor() or platform.machine()}
    try:
        import PIL
        from PyQt6.QtCore import QT_VERSION_STR
        info['pillow'] = PIL.__version__
        info['qt'] = QT_VERSION_STR
    except Exception:  # pylint: disable=broad-except
        pass
    info['primary_screen'] = capture_target.grab_image().size
    if sys.platform == 'win32':
        try:
            info['display_scale_percent'] = ctypes.windll.shcore.GetScaleFactorForDevice(0)
        except Exception:  # pylint: disable=broad-except
            pass
        try:
            info['monitors'] = ctypes.windll.user32.GetSystemMetrics(80)  # SM_CMONITORS
        except Exception:  # pylint: disable=broad-except
            pass
    return info


def compare_images(captured, expected):
    """Share of pixels that differ by more than the tolerance, and their bounding box."""
    if captured.size != expected.size:
        return 1.0, None
    diff = np.array(ImageChops.difference(captured.convert('RGB'), expected))
    mask = np.any(diff > PIXEL_TOLERANCE, axis=2)
    if not mask.any():
        return 0.0, None
    ys, xs = np.nonzero(mask)
    return float(mask.mean()), (int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1)


def recognize(context, target):
    """One recognition run through the real screen capture. Returns (table, total seconds, step timings)."""
    from poker.decisionmaker.current_hand_memory import History
    from poker.tools import timing_benchmark as tb

    strategy = tb.LocalStrategy()
    strategy.read_strategy()
    history = History()
    history.preflop_sheet = context['preflop_sheet']
    table, total, timings, _ = tb.run_recognition(context['table_dict'], target, strategy, history,
                                                  tb.StubGameLogger(), tb.Counters())
    return table, total, timings


def check_recognition(report, context, target, position, name):
    from poker.tools import timing_benchmark as tb
    try:
        table, _, _ = recognize(context, target)
    except Exception as exc:  # pylint: disable=broad-except
        report.add(name, 'FAIL', f"recognition failed: {exc}")
        return
    found, mismatches = tb.check_expected(table, context['spec']['expected'], position)
    if mismatches:
        report.add(name, 'FAIL', f"mismatches (found, expected): {mismatches}")
        report.save_image(table.entireScreenPIL, f"{name.replace(' ', '_')}_capture.png")
    else:
        report.add(name, 'PASS', f"cards {found['my_cards']} / {found['table_cards']}, "
                                 f"pot {found['total_pot']}, call {found['call_value']}, "
                                 f"raise {found['raise_value']}, mouse reference {found['screen_tlc']}")


def measure(report, context, target, label, repeat):
    grabs, totals, corners = [], [], []
    for _ in range(repeat):
        start = time.perf_counter()
        target.grab()
        grabs.append(time.perf_counter() - start)
        try:
            _, total, timings = recognize(context, target)
        except Exception as exc:  # pylint: disable=broad-except
            report.add(f"timing {label}", 'FAIL', str(exc))
            return
        totals.append(total)
        corners.append(timings['get_top_left_corner'])
    report.timing[label] = {
        'capture_ms': round(statistics.mean(grabs) * 1000, 1),
        'top_left_corner_search_ms': round(statistics.mean(corners) * 1000, 1),
        'recognition_ms': round(statistics.mean(totals) * 1000, 1),
        'recognition_median_ms': round(statistics.median(totals) * 1000, 1),
        'repeat': repeat}
    print(f"[TIME] {label}: {report.timing[label]}", flush=True)


def run_checks(args):
    import pandas as pd
    from poker.tools import timing_benchmark as tb

    ensure_dpi_aware()
    tb.use_direct_capture()
    report = Report(args.out)
    report.environment = environment_info()
    print(f"Environment: {report.environment}", flush=True)

    spec = tb.TABLE_SPECS[args.table]
    image_path = os.path.join(get_dir('codebase'), 'tests', 'screenshots', spec['screenshot'])
    expected_image = Image.open(image_path).convert('RGB')
    width, height = expected_image.size
    table_dict, _, _ = tb.build_table_fixture(spec)
    context = {'spec': spec, 'table_dict': table_dict,
               'preflop_sheet': pd.read_excel(os.path.join(get_dir('codebase'), 'decisionmaker', 'preflop.xlsx'),
                                              sheet_name=None, engine='openpyxl')}
    window_target = CaptureTarget(MODE_WINDOW, window_title=WINDOW_TITLE)

    first, second = POSITIONS
    with StandInWindow(image_path, first):
        # 1. listing
        window = find_window(WINDOW_TITLE)
        expected_rect = (first[0], first[1], first[0] + width, first[1] + height)
        if window.rect == expected_rect:
            report.add('window listing', 'PASS', f"found at {window.rect}")
        else:
            report.add('window listing', 'FAIL', f"found at {window.rect}, expected {expected_rect}. "
                                                 f"A size other than {width}x{height} points to display scaling.")

        # 2. pixel exact window capture
        captured, offset = window_target.grab()
        share, box = compare_images(captured, expected_image)
        detail = f"{captured.size[0]}x{captured.size[1]} at {offset}, differing pixels {share:.2%}"
        if share == 0 and offset == first:
            report.add('window capture', 'PASS', detail)
        elif share < 0.005 and offset == first:
            report.add('window capture', 'WARN', f"{detail}, area {box}")
        else:
            path = report.save_image(captured, 'window_capture.png')
            report.add('window capture', 'FAIL', f"{detail}, area {box}, saved {path}")

        # 3. recognition through the real capture
        check_recognition(report, context, window_target, first, 'recognition window mode')
        check_recognition(report, context, CaptureTarget(MODE_FULL_SCREEN), first, 'recognition full screen mode')

        # 6. timing
        measure(report, context, window_target, 'window mode', args.repeat)
        measure(report, context, CaptureTarget(MODE_FULL_SCREEN), 'full screen mode', args.repeat)

    # 4. the window reopens somewhere else
    with StandInWindow(image_path, second):
        check_recognition(report, context, window_target, second, 'moved window')

    # 5. the window is gone
    try:
        window_target.resolve_bbox()
        report.add('missing window', 'FAIL', 'the closed window is still found')
    except CaptureTargetError:
        image, offset = window_target.grab()
        full_size = capture_target.grab_image().size
        status = 'PASS' if offset == (0, 0) and image.size == full_size else 'FAIL'
        report.add('missing window', status, f"captured {image.size} at {offset} (full screen is {full_size})")

    path = report.write()
    print(f"\nResult: {'FAIL' if report.failed() else 'PASS'} -> {os.path.abspath(path)}")
    return 1 if report.failed() else 0


def main():
    parser = argparse.ArgumentParser(description=__doc__.split('\n', maxsplit=1)[0])
    parser.add_argument('--table', default='pokerstars_turn', help='table screenshot to show: pokerstars_turn or '
                                                                   'partypoker_preflop')
    parser.add_argument('--repeat', type=int, default=10, help='recognition runs per mode for the timing')
    parser.add_argument('--out', default='capture_check_output', help='folder for result.json and failure images')
    parser.add_argument('--window-only', type=float, metavar='SECONDS',
                        help='only show the stand-in window at the first position for this many seconds')
    parser.add_argument('--serve-window', nargs=5, metavar=('IMAGE', 'TITLE', 'X', 'Y', 'SECONDS'),
                        help=argparse.SUPPRESS)
    args = parser.parse_args()

    if args.serve_window:
        image, title, x, y, seconds = args.serve_window
        serve_window(image, title, int(x), int(y), float(seconds))
        return 0

    logging.basicConfig(level=logging.WARNING, format='%(levelname)s %(name)s: %(message)s')
    logging.disable(logging.INFO)  # the bot's loggers force DEBUG; keep warnings such as a missing window

    if args.window_only:
        from poker.tools import timing_benchmark as tb
        ensure_dpi_aware()
        spec = tb.TABLE_SPECS[args.table]
        image_path = os.path.join(get_dir('codebase'), 'tests', 'screenshots', spec['screenshot'])
        with StandInWindow(image_path, POSITIONS[0], seconds=args.window_only) as window:
            print(f"Showing '{WINDOW_TITLE}' at {POSITIONS[0]} for {args.window_only:.0f} s. Ctrl+C closes it.")
            try:
                window.process.wait()
            except KeyboardInterrupt:
                pass
        return 0

    return run_checks(args)


if __name__ == '__main__':
    sys.exit(main())
