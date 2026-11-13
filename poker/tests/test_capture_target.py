"""Tests for choosing the capture target (full screen, screen region or game window). No server needed."""
from configparser import ConfigParser

import pandas as pd
import pytest
from PIL import Image

from poker.tools import capture_target
from poker.tools.capture_target import (MODE_FULL_SCREEN, MODE_REGION, MODE_WINDOW, CaptureTarget, CaptureTargetError,
                                        WindowInfo, find_window, parse_region)

# pylint: disable=missing-function-docstring,redefined-outer-name,too-few-public-methods


class FakeConfig:
    """Same interface as CustomConfigParser, without touching config.ini"""

    def __init__(self, text=''):
        self.config = ConfigParser()
        self.config.optionxform = str
        self.config.read_string('[main]\n' + text)
        self.writes = 0

    def update_file(self):
        self.writes += 1


WINDOWS = [
    WindowInfo(1, 'Poker Lobby', (0, 0, 800, 600)),
    WindowInfo(2, "Table 12 - Hold'em", (200, 100, 1150, 800)),
    WindowInfo(3, "Table 7 - Hold'em", (400, 300, 1350, 1000)),
    WindowInfo(4, 'poker', (10, 10, 500, 400)),
]


@pytest.fixture
def grabbed(monkeypatch):
    """Replace the screen grabber and record the requested rectangles"""
    calls = []

    def fake_grab(bbox=None):
        calls.append(bbox)
        size = (1920, 1080) if bbox is None else (bbox[2] - bbox[0], bbox[3] - bbox[1])
        return Image.new('RGB', size)

    monkeypatch.setattr(capture_target, 'grab_image', fake_grab)
    return calls


def test_parse_region_normalizes_corners():
    assert parse_region('300, 200, 100, 50') == (100, 50, 300, 200)
    assert parse_region('0;0;1280;900') == (0, 0, 1280, 900)


@pytest.mark.parametrize('text', ['1,2,3', 'a,b,c,d', '0,0,5,500'])
def test_parse_region_rejects_invalid_input(text):
    with pytest.raises(ValueError):
        parse_region(text)


def test_missing_config_entries_mean_full_screen():
    assert CaptureTarget.from_config(FakeConfig()).mode == MODE_FULL_SCREEN


@pytest.mark.parametrize('text', [
    'capture_mode = region\ncapture_region =\n',
    'capture_mode = region\ncapture_region = 1,2\n',
    'capture_mode = window\ncapture_window_title =\n',
    'capture_mode = something_else\n',
])
def test_incomplete_config_falls_back_to_full_screen(text):
    assert CaptureTarget.from_config(FakeConfig(text)).mode == MODE_FULL_SCREEN


def test_save_and_load_round_trip():
    config = FakeConfig()
    CaptureTarget(MODE_WINDOW, region=(5, 6, 700, 800), window_title="Hold'em").save(config)
    assert config.writes == 1
    assert config.config.get('main', 'capture_region') == '5,6,700,800'
    loaded = CaptureTarget.from_config(config)
    assert loaded == CaptureTarget(MODE_WINDOW, region=(5, 6, 700, 800), window_title="Hold'em")


def test_find_window_prefers_exact_title_then_top_most_match():
    assert find_window('POKER', WINDOWS).handle == 4  # exact match, ignoring case
    assert find_window("hold'em", WINDOWS).handle == 2  # first (top most) window containing the text
    assert find_window('Omaha', WINDOWS) is None


def test_grab_full_screen(grabbed):
    image, offset = CaptureTarget().grab()
    assert grabbed == [None]
    assert image.size == (1920, 1080) and offset == (0, 0)


def test_grab_region_returns_its_position(grabbed):
    image, offset = CaptureTarget(MODE_REGION, region=(100, 50, 1100, 850)).grab()
    assert grabbed == [(100, 50, 1100, 850)]
    assert image.size == (1000, 800) and offset == (100, 50)


def test_grab_window_follows_the_window(grabbed, monkeypatch):
    monkeypatch.setattr(capture_target, 'list_windows', lambda exclude_pid=None: WINDOWS)
    target = CaptureTarget(MODE_WINDOW, window_title='Table 7')
    image, offset = target.grab()
    assert grabbed == [(400, 300, 1350, 1000)]
    assert image.size == (950, 700) and offset == (400, 300)

    moved = [WindowInfo(3, "Table 7 - Hold'em", (0, 0, 950, 700))]
    monkeypatch.setattr(capture_target, 'list_windows', lambda exclude_pid=None: moved)
    _, offset = target.grab()
    assert offset == (0, 0)


def test_missing_window_falls_back_to_full_screen(grabbed, monkeypatch):
    monkeypatch.setattr(capture_target, 'list_windows', lambda exclude_pid=None: WINDOWS)
    target = CaptureTarget(MODE_WINDOW, window_title='Omaha')
    with pytest.raises(CaptureTargetError):
        target.resolve_bbox()
    image, offset = target.grab()
    assert grabbed == [None]
    assert image.size == (1920, 1080) and offset == (0, 0)


def test_take_screenshot_with_offset_uses_the_target(grabbed):
    from poker.tools.screen_operations import take_screenshot_with_offset  # pylint: disable=import-outside-toplevel
    image, offset = take_screenshot_with_offset(target=CaptureTarget(MODE_REGION, region=(30, 40, 530, 440)))
    assert grabbed == [(30, 40, 530, 440)]
    assert image.size == (500, 400) and offset == (30, 40)


@pytest.mark.parametrize('capture', [MODE_FULL_SCREEN, MODE_WINDOW])
@pytest.mark.parametrize('table', ['pokerstars_turn', 'partypoker_preflop'])
def test_recognition_through_capture_target(capture, table, monkeypatch):
    """The table is not at the screen origin: values are recognized and mouse clicks use screen coordinates"""
    import poker.tools.timing_benchmark as tb  # pylint: disable=import-outside-toplevel
    from poker.decisionmaker.current_hand_memory import History  # pylint: disable=import-outside-toplevel

    spec = tb.TABLE_SPECS[table]
    table_dict, desktop, table_rect = tb.build_table_fixture(spec)
    monkeypatch.setattr(capture_target, 'grab_image', tb.FakeScreen(desktop).grab)
    strategy = tb.LocalStrategy()
    strategy.read_strategy()
    history = History()
    history.preflop_sheet = pd.read_excel(tb.os.path.join(tb.get_dir('codebase'), 'decisionmaker', 'preflop.xlsx'),
                                          sheet_name=None, engine='openpyxl')

    recognized, _, _, _ = tb.run_recognition(table_dict, tb.make_capture_target(capture, table_rect), strategy,
                                             history, tb.StubGameLogger(), tb.Counters())

    _, mismatches = tb.check_expected(recognized, spec['expected'])
    assert not mismatches
    assert recognized.screen_tlc == tb.TABLE_OFFSET
    expected_offset = table_rect[:2] if capture == MODE_WINDOW else (0, 0)
    assert recognized.capture_offset == expected_offset
