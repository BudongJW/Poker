"""Offline timing benchmark for screen recognition and decision calculation.

Measures how long one decision cycle of the bot takes, split into two phases:

* screen recognition: the ``ready`` chain in ``main.ThreadManager.run``
  (screenshot -> top left corner -> template matching -> OCR)
* calculation: ``run_montecarlo_wrapper`` (equity simulation) followed by
  ``Decision`` / ``Decision.make_decision``

The bot normally downloads table mappings and strategies from a remote server.
To measure without that server, this script builds table mappings from the
screenshots in ``poker/tests/screenshots``, uses a local strategy and replaces
GUI signals, mouse and game logger with stubs. As a result the following are
NOT part of the measured time and have to be added for a real setup:

* the HTTP request done by ``strategy.read_strategy()`` in every cycle
* the screen capture itself (``ImageGrab.grab``); the captured image is
  replaced by a desktop-sized image with the table placed in it
* mouse movements and GUI updates

The screenshot goes through the capture target code (poker.tools.capture_target):
``--capture full_screen`` captures the whole desktop image, ``--capture window``
only the table's window, like choosing the game window in the "Capture area" dialog.

Usage::

    python -m poker.tools.timing_benchmark --repeat 20 --json result.json
    python -m poker.tools.timing_benchmark --capture window --no-stages
"""
import argparse
import copy
import io
import json
import logging
import os
import platform
import statistics
import sys
import time
from unittest.mock import MagicMock

import numpy as np
import pandas as pd
from PIL import Image

from poker.decisionmaker.current_hand_memory import CurrentHandPreflopState, History
from poker.decisionmaker.decisionmaker import Decision
from poker.decisionmaker.montecarlo_python import run_montecarlo_wrapper
from poker.scraper.table_screen_based import TableScreenBased
from poker.scraper.table_setup_actions_and_signals import CARD_SUITES, CARD_VALUES
from poker.tools import capture_target
from poker.tools.capture_target import MODE_FULL_SCREEN, MODE_REGION, MODE_WINDOW, CaptureTarget
from poker.tools.helper import get_config, get_dir
from poker.tools.strategy_handler import StrategyHandler

# pylint: disable=unnecessary-lambda,too-many-locals,too-many-arguments,too-many-positional-arguments
# pylint: disable=too-few-public-methods

DESKTOP_SIZE = (1920, 1080)
# Screen position of the table window. It is not at the screen origin, so the
# conversion of the top left corner to screen coordinates for mouse clicks is checked.
TABLE_OFFSET = (200, 100)
CAPTURE_MODES = (MODE_FULL_SCREEN, MODE_WINDOW)

TABLE_SPECS = {
    'pokerstars_turn': {
        'title': 'PokerStars 6-max, Turn',
        'screenshot': 'ps473830744_Flop_1.png',
        'max_players': 6,
        'card_size': (22, 39),
        # visible cards: template box (x1, y1, x2, y2) cut from the screenshot
        'cards': {
            'JD': (410, 427, 432, 466), 'JS': (470, 427, 492, 466),
            '6C': (312, 241, 334, 282), 'JC': (375, 241, 397, 282),
            '9H': (439, 241, 461, 282), '8H': (503, 241, 525, 282),
        },
        # templates that are visible on the screenshot
        'templates': {
            'topleft_corner': (0, 0, 40, 20),
            'dealer_button': (364, 420, 390, 441),
            'covered_card': (52, 330, 95, 352),
            'my_turn': (522, 637, 570, 660),
            'call_button': (684, 628, 726, 648),
            'raise_button': (818, 628, 903, 648),
        },
        'areas': {
            'my_cards_area': (400, 418, 535, 475),
            'table_cards_area': (300, 230, 570, 330),
            'buttons_search_area': (470, 615, 945, 690),
            'my_turn_search_area': (470, 615, 945, 690),
            'lost_everything_search_area': (250, 150, 700, 450),
            'total_pot_area': (458, 215, 506, 234),
            'current_round_pot': (443, 343, 500, 362),
            'call_value': (672, 646, 735, 668),
            'raise_value': (830, 646, 892, 668),
            'game_number': (300, 0, 420, 18),
        },
        # per player, index 0 is the bot
        'player_areas': {
            'player_funds_area': [(462, 507, 520, 528), (55, 387, 112, 407), (90, 182, 140, 201),
                                  (418, 119, 473, 138), (795, 182, 852, 201), (825, 387, 880, 407)],
            'player_pot_area': [(330, 395, 400, 415), (238, 360, 287, 378), (190, 200, 240, 218),
                                (420, 145, 470, 163), (700, 200, 750, 218), (680, 365, 730, 383)],
            'button_search_area': [(340, 405, 410, 455), (190, 330, 260, 380), (190, 190, 260, 240),
                                   (520, 120, 590, 170), (650, 190, 720, 240), (680, 330, 750, 380)],
            'covered_card_area': [(380, 380, 540, 420), (30, 300, 190, 365), (30, 100, 190, 160),
                                  (360, 40, 520, 95), (700, 100, 900, 160), (720, 300, 930, 365)],
        },
        'expected': {
            'my_cards': ['JD', 'JS'], 'table_cards': ['6C', 'JC', '9H', '8H'], 'game_stage': 'Turn',
            'dealer_position': 0, 'players_in_game': [0, 1],
            'my_funds': 1.67, 'total_pot': 0.29, 'call_value': 0.04, 'raise_value': 0.08,
        },
    },
    'partypoker_preflop': {
        'title': 'PartyPoker 6-max, PreFlop',
        'screenshot': '53269218_PreFlop_0.png',
        'max_players': 6,
        'card_size': (20, 39),
        'cards': {
            'TS': (357, 349, 377, 388), 'JC': (414, 349, 434, 388),
        },
        'templates': {
            'topleft_corner': (0, 0, 40, 20),
            'dealer_button': (579, 158, 600, 171),
            'covered_card': (338, 47, 380, 70),
            'my_turn': (452, 537, 495, 558),
            'call_button': (582, 527, 618, 546),
            'raise_button': (695, 527, 760, 547),
        },
        'areas': {
            'my_cards_area': (350, 342, 470, 395),
            'table_cards_area': (240, 190, 570, 300),
            'buttons_search_area': (410, 520, 800, 575),
            'my_turn_search_area': (410, 520, 800, 575),
            'lost_everything_search_area': (250, 150, 700, 450),
            'total_pot_area': (398, 173, 450, 192),
            'current_round_pot': (380, 195, 450, 215),
            'call_value': (575, 547, 627, 567),
            'raise_value': (700, 547, 756, 567),
            'game_number': (700, 55, 795, 70),
        },
        'player_areas': {
            'player_funds_area': [(390, 410, 450, 434), (38, 327, 82, 347), (38, 160, 90, 180),
                                  (353, 93, 405, 113), (735, 160, 790, 180), (732, 327, 777, 347)],
            'player_pot_area': [(480, 330, 530, 348), (393, 322, 433, 340), (200, 180, 250, 198),
                                (428, 137, 470, 155), (560, 180, 610, 198), (560, 307, 600, 325)],
            'button_search_area': [(300, 350, 355, 400), (130, 280, 200, 330), (130, 150, 200, 200),
                                   (470, 85, 530, 130), (560, 140, 640, 190), (560, 280, 640, 330)],
            'covered_card_area': [(320, 330, 480, 360), (0, 265, 140, 310), (0, 95, 140, 140),
                                  (320, 35, 460, 80), (660, 95, 800, 140), (660, 265, 800, 310)],
        },
        'expected': {
            'my_cards': ['TS', 'JC'], 'table_cards': [], 'game_stage': 'PreFlop',
            'dealer_position': 4, 'players_in_game': [0, 3],
            'my_funds': 1.98, 'total_pot': 0.08, 'call_value': 0.03, 'raise_value': 0.08,
        },
    },
}

# Templates searched by the scraper that are not visible on the screenshots.
# They get a random image of a typical size so the search cost is still measured.
ABSENT_TEMPLATE_SIZES = {
    'check_button': (40, 20),
    'bet_button': (40, 20),
    'all_in_call_button': (60, 20),
    'im_back': (60, 20),
    'resume_hand': (60, 20),
    'lost_everything': (60, 30),
    'fast_fold_button': (60, 20),
}

BENCHMARK_STRATEGY = {
    'Strategy': 'Benchmark', 'computername': 'benchmark', 'pokerSite': 'PS',
    'bigBlind': 0.02, 'smallBlind': 0.01, 'initialFunds': 2.0, 'initialFunds2': 2.0,
    'PreFlopMinCallEquity': 0.5, 'PreFlopMinBetEquity': 0.6, 'PreFlopMaxBetEquity': 0.9,
    'PreFlopCallPower': 1, 'PreFlopBetPower': 1,
    'FlopMinCallEquity': 0.5, 'FlopMinBetEquity': 0.6, 'FlopCallPower': 1, 'FlopBetPower': 1,
    'TurnMinCallEquity': 0.5, 'TurnMinBetEquity': 0.6, 'TurnCallPower': 1, 'TurnBetPower': 1,
    'RiverMinCallEquity': 0.5, 'RiverMinBetEquity': 0.6, 'RiverCallPower': 1, 'RiverBetPower': 1,
    'FlopBluff': 0, 'FlopBluffMinEquity': 0.3, 'FlopBluffMaxEquity': 0.5,
    'TurnBluffMinEquity': 0.3, 'TurnBluffMaxEquity': 0.5,
    'RiverBluffMinEquity': 0.3, 'RiverBluffMaxEquity': 0.5,
    'FlopCheckDeceptionMinEquity': 0.9, 'TurnCheckDeceptionMinEquity': 0.9, 'RiverCheckDeceptionMinEquity': 0.9,
    'alwaysCallEquity': 0.99, 'always_call_low_stack_multiplier': 8, 'antibluff_percentage': 0,
    'betPotRiverEquity': 0.9, 'betPotRiverEquityMaxBBM': 5, 'secondRiverBetPotMinEquity': 0.9,
    'bullyDivider': 1, 'minBullyEquity': 0.5, 'maxBullyEquity': 0.6,
    'collusion': 0, 'considerLastGames': 1, 'differentiate_reverse_sheet': 1,
    'flop_betting_condidion_1': 1, 'turn_betting_condidion_1': 1, 'river_betting_condidion_1': 1,
    'flop_bluffing_condidion_1': 1, 'turn_bluffing_condidion_1': 0, 'turn_bluffing_condidion_2': 1,
    'river_bluffing_condidion_1': 0, 'river_bluffing_condidion_2': 1,
    'gather_player_names': 0, 'increased_preflop_betting': 0,
    'maxPotAdjustment': 0.2, 'maxPotAdjustmentPreFlop': 0.2, 'potAdjustment': 0, 'potAdjustmentPreFlop': 0,
    'max_abs_fundchange': 5, 'minimumLossForIteration': -10, 'minimum_bet_size': 3,
    'opponent_raised_without_initiative_flop': 1, 'opponent_raised_without_initiative_turn': 1,
    'opponent_raised_without_initiative_river': 1,
    'out_multiplier': 1,
    'pre_flop_equity_increase_if_bet': 0.1, 'pre_flop_equity_increase_if_call': 0.05,
    'pre_flop_equity_reduction_by_position': 0.02,
    'preflop_override': 1, 'range_of_range': 0,
    'range_preflop': 1, 'range_multiple_players': 0.25,
    'range_utg0': 0.2, 'range_utg1': 0.2, 'range_utg2': 0.25, 'range_utg3': 0.3, 'range_utg4': 0.4,
    'range_utg5': 0.5,
    'secondRoundAdjustment': 0.1, 'secondRoundAdjustmentPowerIncrease': 1, 'secondRoundAdjustmentPreFlop': 0.1,
    'strategyIterationGames': 1000, 'use_pot_multiples': 0, 'use_relative_equity': 0, 'BetPlusInc': 1,
}

# Board cards used for the calculation-only sweep over game stages
STAGE_BOARDS = {
    'PreFlop': [],
    'Flop': ['6C', 'JC', '9H'],
    'Turn': ['6C', 'JC', '9H', '8H'],
    'River': ['6C', 'JC', '9H', '8H', '2D'],
}

log = logging.getLogger(__name__)


class LocalStrategy(StrategyHandler):
    """Strategy handler that does not contact the strategy server."""

    def __init__(self):
        super().__init__()
        self.current_strategy = BENCHMARK_STRATEGY['Strategy']

    def read_strategy(self, strategy_override=''):
        self.selected_strategy = dict(BENCHMARK_STRATEGY)
        self.check_defaults()
        return True


class StubGameLogger:
    """Game logger without database access."""

    def get_game_count(self, *_args, **_kwargs):
        return 0

    def get_strategy_return(self, *_args, **_kwargs):
        return 0

    def get_flop_frequency_of_player(self, *_args, **_kwargs):
        return np.nan

    def get_collusion_cards(self, *_args, **_kwargs):
        return '', False

    def mark_last_game(self, *_args, **_kwargs):
        return None

    def upload_collusion_data(self, *_args, **_kwargs):
        return None


def _png_bytes(img):
    buf = io.BytesIO()
    img.save(buf, format='PNG')
    return buf.getvalue()


def _area(box):
    x1, y1, x2, y2 = box
    return {'x1': x1, 'y1': y1, 'x2': x2, 'y2': y2}


def build_table_fixture(spec, seed=0):
    """
    Build a table mapping and a desktop screenshot from a test screenshot.

    Returns table_dict, desktop image and the table's rectangle on the desktop.
    """
    path = os.path.join(get_dir('codebase'), 'tests', 'screenshots', spec['screenshot'])
    table_img = Image.open(path).convert('RGB')
    rng = np.random.default_rng(seed)

    def noise(size):
        width, height = size
        return Image.fromarray(rng.integers(0, 256, (height, width, 3), dtype=np.uint8))

    table_dict = {'table_name': spec['title'], 'max_players': {'value': spec['max_players']}}
    for name, box in spec['templates'].items():
        table_dict[name] = _png_bytes(table_img.crop(box))
    for name, size in ABSENT_TEMPLATE_SIZES.items():
        table_dict[name] = _png_bytes(noise(size))
    for value in CARD_VALUES:
        for suit in CARD_SUITES:
            card = value + suit
            if card in spec['cards']:
                template = table_img.crop(spec['cards'][card])
            else:
                template = noise(spec['card_size'])
            table_dict[card.lower()] = _png_bytes(template)
    for name, box in spec['areas'].items():
        table_dict[name] = _area(box)
    for name, boxes in spec['player_areas'].items():
        table_dict[name] = {str(i): _area(box) for i, box in enumerate(boxes)}

    background = rng.integers(20, 60, (DESKTOP_SIZE[1], DESKTOP_SIZE[0], 3), dtype=np.uint8)
    desktop = Image.fromarray(background)
    desktop.paste(table_img, TABLE_OFFSET)
    table_rect = (TABLE_OFFSET[0], TABLE_OFFSET[1],
                  TABLE_OFFSET[0] + table_img.size[0], TABLE_OFFSET[1] + table_img.size[1])
    return table_dict, desktop, table_rect


class FakeScreen:
    """Stands in for the screen: grabbing returns the desktop image or a part of it."""

    def __init__(self, desktop):
        self.desktop = desktop

    def grab(self, bbox=None):
        return self.desktop if bbox is None else self.desktop.crop(bbox)

    def install(self):
        capture_target.grab_image = self.grab
        return self


def make_capture_target(capture, table_rect):
    """Capture target for a capture mode. A window is captured like a region with the window's rectangle."""
    if capture == MODE_WINDOW:
        return CaptureTarget(MODE_REGION, region=table_rect)
    return CaptureTarget(MODE_FULL_SCREEN)


def recognition_steps(table, strategy, history, mouse, gui_signals):
    """Same order as the ready chain in main.ThreadManager.run (non neural network table)."""
    return [
        ('take_screenshot', lambda: table.take_screenshot(True, strategy)),
        ('get_top_left_corner', lambda: table.get_top_left_corner(strategy)),
        ('check_for_captcha', lambda: table.check_for_captcha(mouse)),
        ('get_lost_everything', lambda: table.get_lost_everything(history, table, strategy, gui_signals)),
        ('check_for_imback', lambda: table.check_for_imback(mouse)),
        ('check_for_resume_hand', lambda: table.check_for_resume_hand(mouse)),
        ('check_for_button_if_slow_table', lambda: table.check_for_button_if_slow_table(False)),
        ('get_my_cards', lambda: table.get_my_cards()),
        ('get_new_hand', lambda: table.get_new_hand(mouse, history, strategy)),
        ('get_table_cards', lambda: table.get_table_cards(history)),
        ('upload_collusion_wrapper', lambda: table.upload_collusion_wrapper(strategy, history)),
        ('get_dealer_position', lambda: table.get_dealer_position()),
        ('check_fast_fold', lambda: table.check_fast_fold(history, strategy, mouse)),
        ('check_for_button', lambda: table.check_for_button()),
        ('read_strategy', lambda: strategy.read_strategy()),
        ('get_round_number', lambda: table.get_round_number(history)),
        ('check_for_checkbutton', lambda: table.check_for_checkbutton()),
        ('init_get_other_players_info', lambda: table.init_get_other_players_info()),
        ('get_other_player_status', lambda: table.get_other_player_status(strategy, history)),
        ('get_other_player_names', lambda: table.get_other_player_names(strategy)),
        ('get_other_player_funds', lambda: table.get_other_player_funds(strategy)),
        ('get_total_pot_value', lambda: table.get_total_pot_value(history)),
        ('get_round_pot_value', lambda: table.get_round_pot_value(history)),
        ('check_for_call', lambda: table.check_for_call()),
        ('check_for_betbutton', lambda: table.check_for_betbutton()),
        ('check_for_allincall', lambda: table.check_for_allincall()),
        ('get_current_call_value', lambda: table.get_current_call_value(strategy)),
        ('get_current_bet_value', lambda: table.get_current_bet_value(strategy)),
    ]


class Counters:
    """Counts template searches and OCR calls during recognition."""

    def __init__(self):
        self.template = 0
        self.ocr = 0

    def install(self):
        import poker.scraper.table_scraper as ts_module  # pylint: disable=import-outside-toplevel
        import poker.tools.screen_operations as so_module  # pylint: disable=import-outside-toplevel
        counters = self
        orig_template = so_module.is_template_in_search_area
        orig_ocr = so_module.ocr

        def counted_template(*args, **kwargs):
            counters.template += 1
            return orig_template(*args, **kwargs)

        def counted_ocr(*args, **kwargs):
            counters.ocr += 1
            return orig_ocr(*args, **kwargs)

        ts_module.is_template_in_search_area = counted_template
        ts_module.ocr = counted_ocr
        so_module.ocr = counted_ocr  # used by get_bot_pot via a local import
        return self


def new_table(table_dict, target, strategy, game_logger, gui_signals):
    table = TableScreenBased(strategy, table_dict, gui_signals, game_logger, 0.0, None)
    table.capture_target = target
    return table


def run_recognition(table_dict, target, strategy, history, game_logger, counters):
    """Run the ready chain once. The screen must be replaced with FakeScreen(...).install() before."""
    gui_signals = MagicMock()
    gui_signals.exit_thread = False
    gui_signals.pause_thread = False
    mouse = MagicMock()
    table = new_table(table_dict, target, strategy, game_logger, gui_signals)
    counters.template = counters.ocr = 0
    timings = {}
    start = time.perf_counter()
    for name, step in recognition_steps(table, strategy, history, mouse, gui_signals):
        t0 = time.perf_counter()
        ok = step()
        timings[name] = time.perf_counter() - t0
        if not ok:
            raise RuntimeError(f"Recognition stopped at step {name}")
    total = time.perf_counter() - start
    return table, total, timings, gui_signals


def run_calculation(table, strategy, history, game_logger, gui_signals, preflop_state):
    config = get_config()
    t0 = time.perf_counter()
    m = run_montecarlo_wrapper(strategy, gui_signals, config, None, table, game_logger, preflop_state, history)
    t1 = time.perf_counter()
    d = Decision(table, history, strategy, game_logger)
    d.make_decision(table, history, strategy, game_logger)
    t2 = time.perf_counter()
    return {'montecarlo': t1 - t0, 'decision': t2 - t1, 'total': t2 - t0,
            'runs': int(m.runs), 'equity': float(table.abs_equity), 'decision_value': d.decision}


def check_expected(table, expected, screen_tlc=TABLE_OFFSET):
    """Compare recognized values and the top left corner in screen coordinates with the expected ones."""
    expected = dict(expected, screen_tlc=tuple(screen_tlc))
    found = {
        'screen_tlc': tuple(table.screen_tlc),
        'my_cards': sorted(table.mycards), 'table_cards': sorted(table.cardsOnTable),
        'game_stage': table.gameStage, 'dealer_position': table.dealer_position,
        'players_in_game': table.players_in_game, 'my_funds': table.myFunds,
        'total_pot': table.totalPotValue, 'call_value': table.currentCallValue,
        'raise_value': table.currentBetValue,
    }
    want = dict(expected, my_cards=sorted(expected['my_cards']), table_cards=sorted(expected['table_cards']))
    mismatches = {k: (found[k], v) for k, v in want.items() if found[k] != v}
    return found, mismatches


def stats(values):
    values = list(values)
    ordered = sorted(values)
    p95 = ordered[min(len(ordered) - 1, int(round(0.95 * (len(ordered) - 1))))]
    return {'mean': statistics.mean(values), 'median': statistics.median(values),
            'stdev': statistics.stdev(values) if len(values) > 1 else 0.0,
            'min': ordered[0], 'p95': p95, 'max': ordered[-1], 'n': len(values)}


def set_scenario(table, stage, opponents):
    """Copy a recognized table and change board cards and active opponents."""
    t = copy.copy(table)
    t.other_players = copy.deepcopy(table.other_players)
    t.cardsOnTable = list(STAGE_BOARDS[stage])
    t.gameStage = stage
    t.max_X = 1 if stage != 'PreFlop' else 0.95
    for i, player in enumerate(t.other_players):
        player['status'] = 1 if i < opponents else 0
    t.other_active_players = opponents
    t.playersAhead = 0
    t.playersBehind = opponents
    t.isHeadsUp = opponents < 2
    t.mt_tm = time.time()
    return t


def environment_info(repeat, warmup, capture):
    info = {
        'capture': capture, 'table_offset': TABLE_OFFSET,
        'python': sys.version.split()[0], 'platform': platform.platform(),
        'processor': platform.processor() or platform.machine(), 'cpu_count': os.cpu_count(),
        'numpy': np.__version__, 'pandas': pd.__version__,
        'desktop_size': DESKTOP_SIZE, 'repeat': repeat, 'warmup': warmup}
    try:
        import cv2  # pylint: disable=import-outside-toplevel
        import lmfit  # pylint: disable=import-outside-toplevel
        import tesserocr  # pylint: disable=import-outside-toplevel
        info['opencv'] = cv2.__version__  # pylint: disable=no-member
        info['lmfit'] = lmfit.__version__
        info['tesseract'] = tesserocr.tesseract_version().splitlines()[0]  # pylint: disable=no-member
    except Exception:  # pylint: disable=broad-except
        pass
    return info


def new_history(preflop_sheet):
    history = History()
    history.preflop_sheet = preflop_sheet
    return history


def benchmark_table(spec, context, repeat, warmup):
    """Full decision cycles (recognition + calculation) on one table screenshot."""
    table_dict, desktop, table_rect = build_table_fixture(spec)
    FakeScreen(desktop).install()
    target = make_capture_target(context['capture'], table_rect)
    samples = {'recognition': [], 'montecarlo': [], 'decision': [], 'calculation': [], 'cycle': []}
    step_times = {}
    table = calc = None
    for i in range(warmup + repeat):
        strategy = LocalStrategy()
        strategy.read_strategy()
        history = new_history(context['preflop_sheet'])
        table, recog_total, timings, gui_signals = run_recognition(
            table_dict, target, strategy, history, context['game_logger'], context['counters'])
        calc = run_calculation(table, strategy, history, context['game_logger'], gui_signals,
                               CurrentHandPreflopState())
        if i < warmup:
            continue
        samples['recognition'].append(recog_total)
        samples['montecarlo'].append(calc['montecarlo'])
        samples['decision'].append(calc['decision'])
        samples['calculation'].append(calc['total'])
        samples['cycle'].append(recog_total + calc['total'])
        for name, value in timings.items():
            step_times.setdefault(name, []).append(value)

    found, mismatches = check_expected(table, spec['expected'])
    result = {name: stats(values) for name, values in samples.items()}
    result.update({
        'title': spec['title'],
        'capture_size': list(table.entireScreenPIL.size),
        'recognition_without_fixed_sleep': stats(
            [r - s for r, s in zip(samples['recognition'], step_times['take_screenshot'])]),
        'steps': {name: stats(values) for name, values in step_times.items()},
        'counts': {'template_searches': context['counters'].template, 'ocr_calls': context['counters'].ocr},
        'recognized': found,
        'mismatches': mismatches,
        'last_calculation': calc,
    })
    return result, table


def benchmark_stage(base, stage, opponents, context, repeat, warmup):
    """Calculation only, on a copy of a recognized table with another board and number of opponents."""
    samples = {'montecarlo': [], 'decision': [], 'total': [], 'runs': []}
    players = set()
    for i in range(warmup + repeat):
        strategy = LocalStrategy()
        strategy.read_strategy()
        history = new_history(context['preflop_sheet'])
        history.round_number = 0
        table = set_scenario(base, stage, opponents)
        calc = run_calculation(table, strategy, history, context['game_logger'], MagicMock(),
                               CurrentHandPreflopState())
        if i < warmup:
            continue
        for name, values in samples.items():
            values.append(calc[name])
        players.add(int(table.assumedPlayers))
    result = {name: stats(values) for name, values in samples.items()}
    result.update({'stage': stage, 'opponents': opponents, 'assumed_players': sorted(players)})
    return result


def run_benchmark(repeat, warmup, capture=MODE_FULL_SCREEN, stages=True):
    context = {
        'capture': capture,
        'game_logger': StubGameLogger(),
        'preflop_sheet': pd.read_excel(os.path.join(get_dir('codebase'), 'decisionmaker', 'preflop.xlsx'),
                                       sheet_name=None, engine='openpyxl'),
        'counters': Counters().install(),
    }
    results = {'environment': environment_info(repeat, warmup, capture), 'tables': {}, 'stages': {}}

    recognized = {}
    for key, spec in TABLE_SPECS.items():
        results['tables'][key], recognized[key] = benchmark_table(spec, context, repeat, warmup)
    if not stages:
        return results

    base = recognized['pokerstars_turn']
    for stage in STAGE_BOARDS:
        # run_montecarlo_wrapper always assumes two players preflop
        for opponents in ((1,) if stage == 'PreFlop' else (1, 2, 4)):
            results['stages'][f'{stage}/{opponents}'] = benchmark_stage(base, stage, opponents, context,
                                                                       repeat, warmup)
    return results


def ms(value):
    return f"{value * 1000:8.1f}"


def print_report(results):
    env = results['environment']
    print("\n=== Environment ===")
    for key, value in env.items():
        print(f"{key:>14}: {value}")

    for key, res in results['tables'].items():
        print(f"\n=== {res['title']} ({key}) ===")
        print(f"captured image: {res['capture_size'][0]}x{res['capture_size'][1]}, "
              f"template searches: {res['counts']['template_searches']}, OCR calls: {res['counts']['ocr_calls']}")
        print(f"mismatches against expected values: {res['mismatches'] or 'none'}")
        print(f"{'phase':<36}{'mean ms':>9}{'median':>9}{'p95':>9}{'max':>9}")
        for phase in ('recognition', 'recognition_without_fixed_sleep', 'montecarlo', 'decision',
                      'calculation', 'cycle'):
            s = res[phase]
            print(f"{phase:<36}{ms(s['mean'])} {ms(s['median'])} {ms(s['p95'])} {ms(s['max'])}")
        print("-- recognition steps (mean ms) --")
        for name, s in sorted(res['steps'].items(), key=lambda kv: -kv[1]['mean']):
            if s['mean'] >= 0.0005:
                print(f"  {name:<34}{ms(s['mean'])}")

    if not results['stages']:
        return
    print("\n=== Calculation by game stage and active opponents ===")
    print(f"{'stage/opponents':<18}{'players':>8}{'MC runs':>9}{'MC mean':>9}{'dec mean':>9}"
          f"{'total':>9}{'p95':>9}")
    for key, res in results['stages'].items():
        print(f"{key:<18}{str(res['assumed_players']):>8}{res['runs']['mean']:9.0f}{ms(res['montecarlo']['mean'])} "
              f"{ms(res['decision']['mean'])} {ms(res['total']['mean'])} {ms(res['total']['p95'])}")


def main():
    parser = argparse.ArgumentParser(description=__doc__.split('\n', maxsplit=1)[0])
    parser.add_argument('--repeat', type=int, default=20, help='measured iterations per scenario')
    parser.add_argument('--warmup', type=int, default=2, help='iterations not counted')
    parser.add_argument('--capture', choices=CAPTURE_MODES, default=MODE_FULL_SCREEN,
                        help='capture the full desktop or only the table window')
    parser.add_argument('--no-stages', action='store_true', help='skip the calculation by game stage')
    parser.add_argument('--json', help='write raw results to this file')
    parser.add_argument('--verbose', action='store_true', help='show bot log output')
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO if args.verbose else logging.CRITICAL)
    if not args.verbose:
        logging.disable(logging.CRITICAL)

    results = run_benchmark(args.repeat, args.warmup, args.capture, stages=not args.no_stages)
    print_report(results)
    if args.json:
        with open(args.json, 'w', encoding='utf-8') as fh:
            json.dump(results, fh, indent=2, default=str)


if __name__ == '__main__':
    main()
