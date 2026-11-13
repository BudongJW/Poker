"""
Offline tests for the screen recognition: template matching, card recognition and OCR.

Unlike test_table_and_ocr.py these tests need neither the table database nor a poker
client. They use the screenshots in tests/screenshots. The coordinates below were measured
on these screenshots, the expected values are what the screenshots show.
"""
import importlib.util
import io
import os
from concurrent.futures import ThreadPoolExecutor

import cv2
import numpy as np
import pytest
from PIL import Image, ImageFilter, ImageOps

from poker.scraper.table_scraper import TableScraper, CARD_TEMPLATE_NAMES
from poker.tools import constants as const
from poker.tools.screen_operations import (binary_pil_to_cv2, check_cropping,
                                           crop_screenshot_with_topleft_corner,
                                           find_template_on_screen, find_templates_in_search_area,
                                           get_ocr_float, is_template_in_search_area, ocr,
                                           parse_ocr_number, pil_to_cv2)

TESTS_DIR = os.path.dirname(os.path.abspath(__file__))
SCREENSHOTS = os.path.join(TESTS_DIR, 'screenshots')

PARTY_POKER_1 = '53269218_PreFlop_0.png'
PARTY_POKER_2 = '988359671_PreFlop_0.png'
POKER_STARS = 'ps473830744_Flop_1.png'
GG_POKER = 'ggpk6ocr.png'
GG_POKER_2023 = 'Screenshot_2023-08-12_114453.png'

requires_tesseract = pytest.mark.skipif(importlib.util.find_spec('tesserocr') is None,
                                        reason="tesserocr is not installed")


def load_screenshot(name):
    return Image.open(os.path.join(SCREENSHOTS, name))


def png_bytes(image):
    """Encode an image the way templates are stored in the table mapping"""
    buffer = io.BytesIO()
    image.save(buffer, format='PNG')
    return buffer.getvalue()


def area(x1, y1, x2, y2):
    return {'x1': x1, 'y1': y1, 'x2': x2, 'y2': y2}


# ---------------------------------------------------------------------------------------
# Parsing of the text recognised by tesseract
# ---------------------------------------------------------------------------------------

@pytest.mark.parametrize('text, expected', [
    ('$0.29', 0.29),
    ('0.08\n', 0.08),
    ('$2', 2.0),
    ('£3.94', 3.94),
    ('€1.95', 1.95),
    ('12.5 BB', 12.5),
    ('$1,234.56', 1234.56),
    ('1.234,56', 1234.56),
    ('1,250', 1250.0),
    ('1,250,000', 1250000.0),
    ('21,793,801,830', 21793801830.0),
    ('21.793.801.830', 21793801830.0),
    ('0,05', 0.05),
    ('1,5', 1.5),
    ('3.94.', 3.94),
    ('.1.67', 1.67),
])
def test_parse_ocr_number(text, expected):
    assert parse_ocr_number(text) == expected


@pytest.mark.parametrize('text', [
    None,
    '',
    '$',
    '.29',  # dropped leading zero or speck: 29 would be a dangerous misread
    '1.2.3',
    '1,2,3',
    '1,23,456',
    '2,08.5',
    '1.234,567.89',
])
def test_parse_ocr_number_rejects_ambiguous_text(text):
    assert parse_ocr_number(text) is None


# ---------------------------------------------------------------------------------------
# Template matching
# ---------------------------------------------------------------------------------------

def test_object_is_counted_once_although_neighbouring_pixels_match():
    screenshot = pil_to_cv2(load_screenshot(POKER_STARS))
    template = screenshot[250:280, 400:440]

    # the template matches at more than one pixel position around the object ...
    raw = cv2.matchTemplate(screenshot, template, cv2.TM_SQDIFF_NORMED)
    assert np.count_nonzero(raw <= 0.01) > 1

    # ... which is still a single object on the screen
    count, points, best_fit, min_val = find_template_on_screen(template, screenshot, 0.01)
    assert count == 1
    assert points == [(400, 250)]
    assert tuple(best_fit) == (400, 250)
    assert min_val == pytest.approx(0, abs=1e-6)


def test_distinct_objects_are_counted_separately():
    rng = np.random.default_rng(1)
    canvas = Image.fromarray(rng.integers(0, 255, (120, 220, 3), dtype=np.uint8))
    card = Image.open(os.path.join(TESTS_DIR, 'test_cards', 'AH.png'))
    canvas.paste(card, (30, 20))
    canvas.paste(card, (120, 25))
    canvas.paste(card, (150, 70))  # touches the second card
    count, points, _, _ = find_template_on_screen(pil_to_cv2(card), pil_to_cv2(canvas), 0.01)
    assert count == 3
    assert sorted(points) == [(30, 20), (120, 25), (150, 70)]


def test_template_not_on_screen():
    screenshot = pil_to_cv2(load_screenshot(PARTY_POKER_1))
    card = pil_to_cv2(Image.open(os.path.join(TESTS_DIR, 'test_cards', 'AH.png')))
    count, points, _, _ = find_template_on_screen(card, screenshot, 0.01)
    assert count == 0
    assert points == []


@pytest.mark.parametrize('mode', ['RGB', 'RGBA', 'L', 'P'])
def test_pil_to_cv2_accepts_all_image_modes(mode):
    image = Image.new('RGB', (4, 3), (255, 0, 0)).convert(mode)
    converted = pil_to_cv2(image)
    assert converted.shape == (3, 4, 3)
    assert converted.dtype == np.uint8
    if mode != 'L':
        assert tuple(converted[0, 0]) == (0, 0, 255)  # red in BGR order


def test_binary_templates_are_decoded_once():
    data = png_bytes(Image.open(os.path.join(TESTS_DIR, 'test_cards', 'QS.png')))
    first = binary_pil_to_cv2(data)
    assert binary_pil_to_cv2(data) is first
    assert binary_pil_to_cv2(bytes(bytearray(data))) is first
    assert not first.flags.writeable  # shared array must not be changed by a caller


def desktop_with_tables(table, positions, size=(1400, 1000)):
    """Simulate a desktop with the poker table window(s) at the given positions"""
    desktop = Image.new('RGB', size, (0, 90, 160))
    for position in positions:
        desktop.paste(table.convert('RGB'), position)
    return desktop


def test_crop_table_from_desktop():
    table = load_screenshot(GG_POKER_2023)  # RGBA png
    top_left_corner = pil_to_cv2(table.crop((0, 0, 60, 30)))
    desktop = desktop_with_tables(table, [(137, 59)])

    cropped, tlc = crop_screenshot_with_topleft_corner(desktop, top_left_corner, useSleep=False)
    assert tlc == (137, 59)
    assert cropped.size == (const.CROP_WIDTH, const.CROP_HEIGHT)
    window = cropped.crop((0, 0) + table.size)
    assert np.array_equal(np.asarray(window), np.asarray(table.convert('RGB')))
    assert check_cropping([window], top_left_corner)
    assert not check_cropping([load_screenshot(POKER_STARS)], top_left_corner)


def test_crop_refuses_multiple_tables():
    table = load_screenshot(POKER_STARS)
    top_left_corner = pil_to_cv2(table.crop((0, 0, 60, 25)))
    desktop = desktop_with_tables(table, [(0, 0), (600, 400)], size=(1600, 1150))
    assert crop_screenshot_with_topleft_corner(desktop, top_left_corner, useSleep=False) == (None, None)


# ---------------------------------------------------------------------------------------
# Card recognition with the table scraper
# ---------------------------------------------------------------------------------------

# corner (rank and suit) of the cards visible on the GG Poker 2023 screenshot
GG_CARD_CORNERS = {
    'kc': (251, 257, 276, 303),
    '4c': (323, 257, 348, 303),
    '5h': (395, 257, 420, 303),
    '4s': (467, 257, 492, 303),
    '6d': (539, 257, 564, 303),
    'as': (371, 462, 396, 508),
    '3d': (414, 458, 440, 506),
}


def gg_table_dict():
    """Table mapping as it would be created in the table setup for this screenshot"""
    screenshot = load_screenshot(GG_POKER_2023).convert('RGB')
    rng = np.random.default_rng(7)
    table_dict = {
        'table_cards_area': area(240, 240, 610, 350),
        'my_cards_area': area(355, 440, 500, 520),
    }
    for card in CARD_TEMPLATE_NAMES:
        if card in GG_CARD_CORNERS:
            template = screenshot.crop(GG_CARD_CORNERS[card])
        else:  # a card that is not on the screenshot
            template = Image.fromarray(rng.integers(0, 255, (46, 25, 3), dtype=np.uint8))
        table_dict[card] = png_bytes(template)
    return table_dict, screenshot


def test_cards_are_recognized_in_screen_order():
    table_dict, screenshot = gg_table_dict()
    table_scraper = TableScraper(table_dict)
    table_scraper.screenshot = screenshot

    assert table_scraper.get_table_cards2()
    assert table_scraper.table_cards == ['KC', '4C', '5H', '4S', '6D']  # flop, turn, river

    table_scraper.get_my_cards2()
    assert table_scraper.my_cards == ['AS', '3D']


def test_similar_card_template_only_counts_once():
    table_dict, screenshot = gg_table_dict()
    # A template for the 4 of diamonds that looks almost like the 4 of clubs, as it can
    # happen with small or blurry templates.
    rng = np.random.default_rng(3)
    four_of_clubs = np.asarray(screenshot.crop(GG_CARD_CORNERS['4c'])).astype(np.int16)
    similar = np.clip(four_of_clubs + rng.integers(-4, 5, four_of_clubs.shape), 0, 255)
    table_dict['4d'] = png_bytes(Image.fromarray(similar.astype(np.uint8)))

    # both templates match the same card ...
    assert is_template_in_search_area(table_dict, screenshot, '4c', 'table_cards_area')
    assert is_template_in_search_area(table_dict, screenshot, '4d', 'table_cards_area')

    # ... but only the better fitting one is reported
    table_scraper = TableScraper(table_dict)
    table_scraper.screenshot = screenshot
    assert table_scraper.get_table_cards2()
    assert table_scraper.table_cards == ['KC', '4C', '5H', '4S', '6D']


def test_incomplete_flop_is_reported():
    table_dict, screenshot = gg_table_dict()
    table_dict['table_cards_area'] = area(240, 240, 390, 350)  # only two of the cards
    table_scraper = TableScraper(table_dict)
    table_scraper.screenshot = screenshot
    assert not table_scraper.get_table_cards2()
    assert table_scraper.table_cards == ['KC', '4C']


def test_missing_card_template_is_skipped():
    table_dict, screenshot = gg_table_dict()
    del table_dict['2c']
    matches = find_templates_in_search_area(table_dict, screenshot, CARD_TEMPLATE_NAMES, 'my_cards_area')
    assert [name for name, _, _, _ in matches] == ['as', '3d']


def test_search_area_smaller_than_template():
    table_dict, screenshot = gg_table_dict()
    table_dict['my_cards_area'] = area(355, 440, 370, 450)
    with pytest.raises(RuntimeError, match='too small'):
        find_templates_in_search_area(table_dict, screenshot, ['as'], 'my_cards_area')


# ---------------------------------------------------------------------------------------
# OCR of numbers on real screenshots
# ---------------------------------------------------------------------------------------

# (screenshot, crop box, value shown on the screenshot)
REAL_NUMBERS = [
    (POKER_STARS, (460, 216, 504, 234), 0.29),  # pot
    (POKER_STARS, (674, 646, 736, 668), 0.04),  # call button
    (POKER_STARS, (830, 646, 892, 668), 0.08),  # raise button
    (POKER_STARS, (466, 507, 515, 527), 1.67),  # my funds
    (POKER_STARS, (60, 387, 106, 407), 1.80),
    (POKER_STARS, (828, 387, 876, 407), 1.95),
    (POKER_STARS, (240, 360, 282, 378), 0.04),  # player bet
    (POKER_STARS, (445, 343, 492, 361), 0.25),
    (POKER_STARS, (800, 183, 846, 201), 4.41),  # dimmed funds of a sitting out player
    (POKER_STARS, (420, 120, 468, 138), 1.66),
    (POKER_STARS, (90, 183, 138, 201), 1.14),
    (GG_POKER, (418, 242, 462, 260), 0.08),  # pot, yellow text
    (GG_POKER, (611, 586, 657, 606), 0.05),  # call, white text on red
    (GG_POKER, (720, 586, 768, 606), 0.08),
    (GG_POKER, (380, 571, 430, 592), 1.78),  # blue text
    (GG_POKER, (382, 159, 430, 179), 3.48),
    (GG_POKER, (76, 219, 124, 239), 2.28),
    (GG_POKER, (688, 219, 735, 239), 2.12),
    (GG_POKER, (50, 466, 98, 486), 2.28),
    (GG_POKER, (186, 249, 228, 267), 0.01),
    (GG_POKER, (390, 221, 432, 239), 0.02),
    (GG_POKER, (584, 249, 626, 267), 0.05),
    (PARTY_POKER_1, (396, 173, 446, 192), 0.08),
    (PARTY_POKER_1, (576, 548, 624, 568), 0.03),
    (PARTY_POKER_1, (702, 548, 752, 568), 0.08),
    (PARTY_POKER_1, (396, 412, 446, 432), 1.98),  # gold text
    (PARTY_POKER_1, (354, 92, 406, 111), 4.74),
    (PARTY_POKER_1, (40, 160, 90, 180), 3.62),  # dark gold on black
    (PARTY_POKER_1, (708, 160, 758, 180), 8.68),
    (PARTY_POKER_1, (40, 328, 90, 348), 2.31),
    (PARTY_POKER_1, (708, 328, 758, 348), 3.96),
    (PARTY_POKER_1, (428, 137, 470, 153), 0.05),  # partly covered by chips
    (PARTY_POKER_1, (392, 323, 434, 339), 0.02),
    (PARTY_POKER_1, (562, 309, 604, 325), 0.01),
    (PARTY_POKER_1, (730, 58, 792, 70), 21793801830),  # game number with thousands separators
    (PARTY_POKER_2, (396, 173, 446, 192), 0.03),
    (PARTY_POKER_2, (576, 548, 624, 568), 0.01),
    (PARTY_POKER_2, (702, 548, 752, 568), 0.04),
    (PARTY_POKER_2, (396, 412, 446, 432), 1.95),
    (PARTY_POKER_2, (354, 92, 406, 111), 2.08),
    (PARTY_POKER_2, (40, 160, 90, 180), 4.86),
    (PARTY_POKER_2, (708, 160, 758, 180), 3.21),
    (PARTY_POKER_2, (40, 328, 90, 348), 2.31),
    (PARTY_POKER_2, (708, 328, 758, 348), 2),
    (PARTY_POKER_2, (200, 309, 242, 325), 0.02),
    (PARTY_POKER_2, (392, 323, 434, 339), 0.01),
    (PARTY_POKER_2, (730, 58, 792, 70), 21807665834),
]


def real_number_crops():
    return [(load_screenshot(name).crop(box), expected) for name, box, expected in REAL_NUMBERS]


@requires_tesseract
@pytest.mark.parametrize('name, box, expected', REAL_NUMBERS,
                         ids=[f'{name.split(".")[0]}-{expected}' for name, _, expected in REAL_NUMBERS])
def test_ocr_real_screenshots(name, box, expected):
    assert get_ocr_float(load_screenshot(name).crop(box)) == expected


@requires_tesseract
@pytest.mark.parametrize('file_name', ['num1.png', 'num2.png'])
def test_ocr_tricky_numbers(file_name):
    assert get_ocr_float(Image.open(os.path.join(TESTS_DIR, 'ocr', file_name))) == 3.94


def _scale(factor):
    def scale(image):
        size = (max(1, round(image.width * factor)), max(1, round(image.height * factor)))
        return image.resize(size, Image.BILINEAR)
    return scale


def _jpeg(image):
    buffer = io.BytesIO()
    image.convert('RGB').save(buffer, 'JPEG', quality=60)
    return Image.open(io.BytesIO(buffer.getvalue()))


def _noise(image):
    pixels = np.asarray(image.convert('RGB')).astype(np.int16)
    noise = np.random.default_rng(0).normal(0, 12, pixels.shape)
    return Image.fromarray(np.clip(pixels + noise, 0, 255).astype(np.uint8))


# Different screen scaling, compression and themes of the poker clients
DISTORTIONS = {
    'inverted colours': lambda image: ImageOps.invert(image.convert('RGB')),
    'scaled to 80%': _scale(0.8),
    'scaled to 150%': _scale(1.5),
    'jpeg compressed': _jpeg,
    'noise': _noise,
    'blurred': lambda image: image.convert('RGB').filter(ImageFilter.GaussianBlur(0.6)),
}


@requires_tesseract
def test_ocr_is_robust_to_distorted_screenshots():
    correct = wrong = total = 0
    for distort in DISTORTIONS.values():
        for crop, expected in real_number_crops():
            value = get_ocr_float(distort(crop))
            total += 1
            if value == expected:
                correct += 1
            elif value != -1:
                wrong += 1
    assert correct / total >= 0.97
    # a wrong number is worse than no number as the bot would act on it
    assert wrong / total <= 0.02


@requires_tesseract
def test_ocr_from_several_threads():
    crops = real_number_crops()[:20]
    with ThreadPoolExecutor(max_workers=4) as executor:
        results = list(executor.map(lambda item: get_ocr_float(item[0]), crops * 3))
    assert results == [expected for _, expected in crops] * 3


@requires_tesseract
def test_table_scraper_reads_numbers_from_desktop_screenshot():
    table = load_screenshot(POKER_STARS)
    table_dict = {
        'topleft_corner': png_bytes(table.crop((0, 0, 60, 25))),
        'total_pot_area': area(460, 216, 504, 234),
        'current_round_pot': area(445, 343, 492, 361),
        'call_value': area(674, 646, 736, 668),
        'raise_value': area(892, 668, 830, 646),  # corners marked in the other direction
        'player_funds_area': {
            '0': area(466, 507, 515, 527),
            '1': area(60, 387, 106, 407),
            '2': area(90, 183, 138, 201),
            '3': area(420, 120, 468, 138),
            '4': area(800, 183, 846, 201),
            '5': area(828, 387, 876, 407),
        },
    }
    table_scraper = TableScraper(table_dict)
    table_scraper.screenshot = desktop_with_tables(table, [(211, 97)])
    assert table_scraper.crop_from_top_left_corner() is not None
    assert table_scraper.tlc == (211, 97)

    table_scraper.get_pots()
    assert table_scraper.total_pot == 0.29
    assert table_scraper.current_round_pot == 0.25
    assert table_scraper.get_call_value() == 0.04
    assert table_scraper.get_raise_value() == 0.08

    table_scraper.get_my_funds2()
    assert table_scraper.player_funds == [1.67]
    table_scraper.get_players_funds()
    assert table_scraper.player_funds == [1.67, 1.80, 1.14, 1.66, 4.41, 1.95]
    assert ocr(table_scraper.screenshot, 'player_funds_area', table_dict, player=0) == 1.67
