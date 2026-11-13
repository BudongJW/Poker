"""Choose what the bot captures: the full screen, a fixed screen region or one game window.

The choice is stored in ``config.ini`` (section ``[main]``):

* ``capture_mode``: ``full_screen`` (default), ``region`` or ``window``
* ``capture_region``: ``x1,y1,x2,y2`` in screen pixels, used by ``region``
* ``capture_window_title``: text contained in the title of the game window, used by ``window``

The window is looked up again for every screenshot, so it can be moved while the bot runs.
Only the captured part is searched for the table, which also makes finding the top left
corner faster than searching the full screen.

Command line usage::

    python -m poker.tools.capture_target --list
    python -m poker.tools.capture_target --grab capture.png
    python -m poker.tools.capture_target --window "Hold'em" --save
    python -m poker.tools.capture_target --region 0,0,1280,900 --save
    python -m poker.tools.capture_target --full-screen --save
"""
import argparse
import ctypes
import logging
import os
import sys
import time
from dataclasses import dataclass

from PIL import ImageGrab

from poker.tools.helper import get_config

# pylint: disable=logging-fstring-interpolation

log = logging.getLogger(__name__)

MODE_FULL_SCREEN = 'full_screen'
MODE_REGION = 'region'
MODE_WINDOW = 'window'
MODES = (MODE_FULL_SCREEN, MODE_REGION, MODE_WINDOW)

CONFIG_SECTION = 'main'
CONFIG_MODE = 'capture_mode'
CONFIG_REGION = 'capture_region'
CONFIG_WINDOW_TITLE = 'capture_window_title'

MIN_SIZE = 10  # smallest accepted width or height of a region or window

_warned_messages = set()  # warnings that were already logged once


class CaptureTargetError(Exception):
    """Raised when the configured capture target cannot be used."""


@dataclass
class WindowInfo:
    """A visible top level window. ``rect`` is (left, top, right, bottom) in screen pixels."""
    handle: int
    title: str
    rect: tuple
    pid: int = 0

    @property
    def width(self):
        return self.rect[2] - self.rect[0]

    @property
    def height(self):
        return self.rect[3] - self.rect[1]

    def contains(self, x, y):
        left, top, right, bottom = self.rect
        return left <= x < right and top <= y < bottom


def parse_region(text):
    """Parse 'x1,y1,x2,y2' into a normalized tuple. Raises ValueError for invalid input."""
    parts = [p.strip() for p in str(text).replace(';', ',').split(',') if p.strip() != '']
    if len(parts) != 4:
        raise ValueError(f"Region needs four numbers x1,y1,x2,y2, got: {text!r}")
    x1, y1, x2, y2 = (int(round(float(p))) for p in parts)
    region = (min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2))
    if region[2] - region[0] < MIN_SIZE or region[3] - region[1] < MIN_SIZE:
        raise ValueError(f"Region is too small: {region}")
    return region


def format_region(region):
    return ','.join(str(int(v)) for v in region)


@dataclass
class CaptureTarget:
    """Which part of the screen is captured."""
    mode: str = MODE_FULL_SCREEN
    region: tuple = None
    window_title: str = ''

    def __post_init__(self):
        if self.mode not in MODES:
            raise ValueError(f"Unknown capture mode {self.mode!r}, use one of {MODES}")
        if self.region is not None:
            self.region = parse_region(format_region(self.region))

    @classmethod
    def from_config(cls, config=None):
        """Read the capture target from config.ini. Missing or broken entries mean full screen."""
        config = config or get_config()
        cfg = config.config
        mode = cfg.get(CONFIG_SECTION, CONFIG_MODE, fallback=MODE_FULL_SCREEN).strip() or MODE_FULL_SCREEN
        region_text = cfg.get(CONFIG_SECTION, CONFIG_REGION, fallback='').strip()
        title = cfg.get(CONFIG_SECTION, CONFIG_WINDOW_TITLE, fallback='').strip()
        region = None
        if region_text:
            try:
                region = parse_region(region_text)
            except ValueError as exc:
                log.warning(f"Ignoring invalid {CONFIG_REGION} in config.ini: {exc}")
        if mode not in MODES:
            log.warning(f"Unknown {CONFIG_MODE} '{mode}' in config.ini, capturing the full screen")
            mode = MODE_FULL_SCREEN
        if mode == MODE_REGION and region is None:
            log.warning("capture_mode is 'region' but no valid capture_region is set, capturing the full screen")
            mode = MODE_FULL_SCREEN
        if mode == MODE_WINDOW and not title:
            log.warning("capture_mode is 'window' but capture_window_title is empty, capturing the full screen")
            mode = MODE_FULL_SCREEN
        return cls(mode=mode, region=region, window_title=title)

    def save(self, config=None):
        """Write the capture target to config.ini."""
        config = config or get_config()
        cfg = config.config
        if not cfg.has_section(CONFIG_SECTION):
            cfg.add_section(CONFIG_SECTION)
        cfg.set(CONFIG_SECTION, CONFIG_MODE, self.mode)
        cfg.set(CONFIG_SECTION, CONFIG_REGION, format_region(self.region) if self.region else '')
        cfg.set(CONFIG_SECTION, CONFIG_WINDOW_TITLE, self.window_title)
        config.update_file()

    def describe(self):
        if self.mode == MODE_REGION:
            x1, y1, x2, y2 = self.region
            return f"Region {x1},{y1} - {x2},{y2} ({x2 - x1}x{y2 - y1})"
        if self.mode == MODE_WINDOW:
            return f"Window with title containing '{self.window_title}'"
        return "Full screen"

    def resolve_bbox(self, windows=None):
        """
        Screen rectangle to capture, or None for the full screen.

        Raises CaptureTargetError when the window cannot be found.
        """
        if self.mode == MODE_REGION:
            return self.region
        if self.mode == MODE_WINDOW:
            if windows is None:  # never capture the bot's own windows
                windows = list_windows(exclude_pid=os.getpid())
            window = find_window(self.window_title, windows)
            if window is None:
                raise CaptureTargetError(f"No visible window with a title containing '{self.window_title}'")
            return window.rect
        return None

    def grab(self):
        """
        Capture the target.

        Returns (image, offset), where offset is the screen position of the image's top left pixel.
        When the window cannot be found, the full screen is captured and a warning is logged once.
        """
        try:
            bbox = self.resolve_bbox()
        except CaptureTargetError as exc:
            if str(exc) not in _warned_messages:
                log.warning(f"{exc}. Capturing the full screen instead.")
                _warned_messages.add(str(exc))
            bbox = None
        image = grab_image(bbox)
        offset = (bbox[0], bbox[1]) if bbox else (0, 0)
        return image, offset


def grab_image(bbox=None):
    """Capture the screen, or only bbox=(left, top, right, bottom) in screen pixels."""
    if bbox is None:
        return ImageGrab.grab()
    if sys.platform == 'win32':
        # all_screens makes windows on a secondary monitor work; bbox is in virtual screen pixels
        return ImageGrab.grab(bbox=bbox, all_screens=True)
    return ImageGrab.grab(bbox=bbox)


def find_window(title, windows=None):
    """Return the top most visible window whose title equals or contains `title` (case insensitive)."""
    if not title:
        return None
    windows = list_windows() if windows is None else windows
    wanted = title.strip().lower()
    for window in windows:
        if window.title.strip().lower() == wanted:
            return window
    for window in windows:
        if wanted in window.title.lower():
            return window
    return None


def list_windows(exclude_pid=None):
    """
    Visible, not minimized top level windows with a title, top most first.

    Returns an empty list when listing windows is not supported on this system.
    """
    try:
        if sys.platform == 'win32':
            windows = _list_windows_win32()
        elif sys.platform == 'darwin':
            windows = _list_windows_mac()
        else:
            windows = _list_windows_x11()
    except Exception as exc:  # pylint: disable=broad-except
        log.warning(f"Unable to list windows: {exc}")
        return []
    return [w for w in windows
            if w.width >= MIN_SIZE and w.height >= MIN_SIZE and (exclude_pid is None or w.pid != exclude_pid)]


def ensure_dpi_aware():
    """
    On Windows, make window rectangles use physical pixels like the screenshots do.

    Qt applications are DPI aware already; call this in scripts without Qt.
    """
    if sys.platform != 'win32':
        return
    try:
        ctypes.windll.shcore.SetProcessDpiAwareness(2)  # per monitor DPI aware
    except Exception:  # pylint: disable=broad-except
        try:
            ctypes.windll.user32.SetProcessDPIAware()
        except Exception:  # pylint: disable=broad-except
            pass


def _list_windows_win32():
    from ctypes import wintypes  # pylint: disable=import-outside-toplevel

    user32 = ctypes.windll.user32
    dwmapi = ctypes.windll.dwmapi
    dwmwa_extended_frame_bounds = 9
    dwmwa_cloaked = 14
    windows = []

    def window_rect(hwnd):
        rect = wintypes.RECT()
        # the extended frame bounds leave out the invisible resize borders of Windows 10/11
        if dwmapi.DwmGetWindowAttribute(hwnd, dwmwa_extended_frame_bounds, ctypes.byref(rect),
                                        ctypes.sizeof(rect)) != 0:
            if not user32.GetWindowRect(hwnd, ctypes.byref(rect)):
                return None
        return rect.left, rect.top, rect.right, rect.bottom

    def is_cloaked(hwnd):
        cloaked = wintypes.DWORD()
        result = dwmapi.DwmGetWindowAttribute(hwnd, dwmwa_cloaked, ctypes.byref(cloaked), ctypes.sizeof(cloaked))
        return result == 0 and cloaked.value != 0

    def callback(hwnd, _lparam):
        if not user32.IsWindowVisible(hwnd) or user32.IsIconic(hwnd) or is_cloaked(hwnd):
            return True
        length = user32.GetWindowTextLengthW(hwnd)
        if length == 0:
            return True
        buffer = ctypes.create_unicode_buffer(length + 1)
        user32.GetWindowTextW(hwnd, buffer, length + 1)
        rect = window_rect(hwnd)
        if rect is None:
            return True
        pid = wintypes.DWORD()
        user32.GetWindowThreadProcessId(hwnd, ctypes.byref(pid))
        windows.append(WindowInfo(int(hwnd), buffer.value, rect, int(pid.value)))
        return True

    enum_proc = ctypes.WINFUNCTYPE(wintypes.BOOL, wintypes.HWND, wintypes.LPARAM)
    user32.EnumWindows(enum_proc(callback), 0)  # top most window first
    return windows


def _list_windows_mac():
    # pylint: disable=import-outside-toplevel,no-name-in-module,import-error
    from Quartz import (CGWindowListCopyWindowInfo, kCGNullWindowID,
                        kCGWindowListExcludeDesktopElements, kCGWindowListOptionOnScreenOnly)

    options = kCGWindowListOptionOnScreenOnly | kCGWindowListExcludeDesktopElements
    windows = []
    for info in CGWindowListCopyWindowInfo(options, kCGNullWindowID):  # front most window first
        if info.get('kCGWindowLayer', 0) != 0:
            continue
        owner = info.get('kCGWindowOwnerName', '') or ''
        name = info.get('kCGWindowName', '') or ''  # needs the screen recording permission
        title = f"{owner} - {name}" if name else owner
        if not title:
            continue
        bounds = info['kCGWindowBounds']
        left, top = int(bounds['X']), int(bounds['Y'])
        rect = (left, top, left + int(bounds['Width']), top + int(bounds['Height']))
        windows.append(WindowInfo(int(info.get('kCGWindowNumber', 0)), title, rect,
                                  int(info.get('kCGWindowOwnerPID', 0))))
    return windows


def _list_windows_x11():  # pylint: disable=too-many-locals
    from Xlib import X, display  # pylint: disable=import-outside-toplevel

    disp = display.Display()
    try:
        root = disp.screen().root
        utf8 = disp.intern_atom('UTF8_STRING')
        net_wm_name = disp.intern_atom('_NET_WM_NAME')
        net_wm_pid = disp.intern_atom('_NET_WM_PID')
        stacking = root.get_full_property(disp.intern_atom('_NET_CLIENT_LIST_STACKING'), X.AnyPropertyType)
        if stacking is not None:
            candidates = [disp.create_resource_object('window', wid) for wid in stacking.value]
        else:  # no window manager
            candidates = root.query_tree().children
        windows = []
        for win in reversed(candidates):  # stacking order is bottom to top
            try:
                if win.get_attributes().map_state != X.IsViewable:
                    continue
                name_prop = win.get_full_property(net_wm_name, utf8)
                title = name_prop.value.decode('utf-8', 'replace') if name_prop else win.get_wm_name()
                if isinstance(title, bytes):
                    title = title.decode('latin-1')
                if not title:
                    continue
                geometry = win.get_geometry()
                origin = root.translate_coords(win, 0, 0)
                pid_prop = win.get_full_property(net_wm_pid, X.AnyPropertyType)
                pid = int(pid_prop.value[0]) if pid_prop else 0
                rect = (origin.x, origin.y, origin.x + geometry.width, origin.y + geometry.height)
                windows.append(WindowInfo(int(win.id), title, rect, pid))
            except Exception:  # pylint: disable=broad-except
                continue  # window closed while listing
        return windows
    finally:
        disp.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__.split('\n', maxsplit=1)[0])
    parser.add_argument('--list', action='store_true', help='list visible windows')
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument('--full-screen', action='store_true', help='capture the full screen')
    mode.add_argument('--region', help='capture a region given as x1,y1,x2,y2')
    mode.add_argument('--window', help='capture the window whose title contains this text')
    parser.add_argument('--save', action='store_true', help='store the chosen capture target in config.ini')
    parser.add_argument('--grab', metavar='FILE', help='capture once and save the image to FILE')
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format='%(levelname)s %(message)s')
    ensure_dpi_aware()

    if args.list:
        for window in list_windows():
            left, top, right, bottom = window.rect
            print(f"{left:6d},{top:6d} {right - left:5d}x{bottom - top:<5d} pid {window.pid:<7d} {window.title}")

    if args.full_screen:
        target = CaptureTarget()
    elif args.region:
        target = CaptureTarget(MODE_REGION, region=parse_region(args.region))
    elif args.window:
        target = CaptureTarget(MODE_WINDOW, window_title=args.window)
    else:
        target = CaptureTarget.from_config()
    print(f"Capture target: {target.describe()}")

    if args.save:
        target.save()
        print("Saved to config.ini")

    if args.grab:
        start = time.perf_counter()
        image, offset = target.grab()
        elapsed = time.perf_counter() - start
        image.save(args.grab)
        print(f"Captured {image.size[0]}x{image.size[1]} at screen position {offset} "
              f"in {elapsed * 1000:.0f} ms -> {os.path.abspath(args.grab)}")


if __name__ == '__main__':
    main()
