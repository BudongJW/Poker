"""Operations to help identify items on screen"""
import functools
import io
import logging
import os
import re
import sys
import threading
from collections import Counter
from time import sleep

import cv2
import numpy as np
from PIL import Image, ImageGrab

from poker.tools.helper import memory_cache, get_dir
from poker.tools import constants as const

log = logging.getLogger(__name__)
is_debug = False  # used for saving images for debug purposes

if getattr(sys, 'frozen', False) and hasattr(sys, '_MEIPASS'):
    tesserpath = os.path.join(get_dir('codebase'), 'tessdata')
else:
    tesserpath = os.path.join(get_dir('codebase'), '..', 'tessdata')

# Characters tesseract is allowed to return when reading numbers. The comma is needed to
# read thousands separators ('21,793,801') instead of guessing another character for it.
OCR_WHITELIST = "0123456789.,$£€B"
# Text crops are scaled up so that the text is roughly this many pixels high before OCR
OCR_TARGET_HEIGHT = 60
# White margin added around the text, tesseract struggles with text touching the border
OCR_PADDING = 12
# Upper limit of distinct matches reported by find_template_on_screen
MAX_TEMPLATE_MATCHES = 100

# The tesseract api is not thread safe and is shared by the bot and the table setup
_tesseract_lock = threading.RLock()


@functools.lru_cache(maxsize=1)
def _tesseract_api():
    """Create the tesseract api on first use rather than at import time"""
    from tesserocr import PyTessBaseAPI, PSM, OEM
    api = PyTessBaseAPI(path=tesserpath,
                        psm=PSM.SINGLE_LINE,
                        oem=OEM.LSTM_ONLY)
    api.SetVariable("tessedit_char_whitelist", OCR_WHITELIST)
    return api


def _distinct_matches(result, threshold, width, height, max_matches=MAX_TEMPLATE_MATCHES):
    """
    Turn a cv2.matchTemplate (TM_SQDIFF_NORMED) result into distinct matches.

    A template usually matches not only at its exact position but also one or two pixels
    next to it. Positions that are closer than half the template size to a better match
    are therefore treated as the same object (non-maximum suppression).

    Returns:
        list of ((x, y), score) tuples, best match first

    """
    ys, xs = np.nonzero(result <= threshold)
    if len(xs) == 0:
        return []
    scores = result[ys, xs]
    order = np.argsort(scores, kind='stable')
    xs, ys, scores = xs[order], ys[order], scores[order]

    min_dx = max(1, width // 2)
    min_dy = max(1, height // 2)
    available = np.ones(len(xs), dtype=bool)
    matches = []
    while len(matches) < max_matches:
        i = int(np.argmax(available))
        if not available[i]:
            break
        matches.append(((int(xs[i]), int(ys[i])), float(scores[i])))
        available &= (np.abs(xs - xs[i]) >= min_dx) | (np.abs(ys - ys[i]) >= min_dy)
    return matches


def find_template_on_screen(template, screenshot, threshold, extended=False):  # pylint: disable=unused-argument
    """
    Find template on screen

    Args:
        template: cv2 image to look for
        screenshot: cv2 image to search in
        threshold: maximum normalized squared difference (0 is a perfect match)

    Returns:
        count: number of distinct places where the template was found. Neighbouring pixel
            positions that match the same object are only counted once.
        points: top left (x, y) corner of each of these places, best match first
        best_fit: top left (x, y) corner of the best match, even if it is above the threshold
        min_val: score of the best match

    """
    res = cv2.matchTemplate(screenshot, template, cv2.TM_SQDIFF_NORMED)
    min_val, _, min_loc, _ = cv2.minMaxLoc(res)
    matches = _distinct_matches(res, threshold, template.shape[1], template.shape[0])
    points = [point for point, _ in matches]
    return len(points), points, min_loc, min_val


@memory_cache
def load_table_template_cached(table_name):
    """Load template from mongodb as cv2 image"""
    from poker.tools.mongo_manager import MongoManager
    mongo = MongoManager()
    table = mongo.get_table(table_name=table_name)
    return table


def get_table_template_image(table_name='default', label='topleft_corner'):
    """Load template from mongodb as cv2 image"""
    from poker.tools.mongo_manager import MongoManager
    mongo = MongoManager()
    table = mongo.get_table(table_name=table_name)
    template_pil = Image.open(io.BytesIO(table[label]))
    return pil_to_cv2(template_pil)


def get_ocr_float(img_orig, fast=False):
    """Return float value from image. -1.0f when OCR failed"""
    return get_ocr_number(img_orig, fast)


def prepareImage(img_orig, binarize=True, threshold=76):
    """Prepare image for OCR"""

    def binarize_array_opencv(image, threshold):
        """Binarize image from gray channel with 76 as threshold"""
        img = np.asarray(image.convert('L'))

        _, thresh2 = cv2.threshold(img, threshold, 255, cv2.THRESH_BINARY_INV)
        return Image.fromarray(thresh2)

    basewidth = 300
    wpercent = (basewidth / float(img_orig.size[0]))
    hsize = int((float(img_orig.size[1]) * float(wpercent)))
    img_resized = img_orig.convert('L').resize(
        (basewidth, hsize), Image.LANCZOS)
    if binarize:
        img_resized = binarize_array_opencv(img_resized, threshold)

    if is_debug:
        pics_path = "log/pics"
        try:
            if not os.path.exists(pics_path):
                os.makedirs(pics_path)
        except OSError:
            log.error("Creation of the directory %s failed" % pics_path)
            sys.exit(1)

        img_orig.save('log/pics/img_orig.png')
        img_resized.save('log/pics/img_resized.png')

        log.debug("ocr images prepared")

    return img_resized


def get_ocr_number2(img_orig, fast=False):  # pylint: disable=unused-argument
    """Run tesseract on an already prepared PIL image and return the raw text"""
    with _tesseract_lock:
        api = _tesseract_api()
        api.SetImage(img_orig)
        return api.GetUTF8Text()


_GROUPED_BY_COMMA = re.compile(r'[1-9]\d{0,2}(?:,\d{3})+')
_GROUPED_BY_DOT = re.compile(r'[1-9]\d{0,2}(?:\.\d{3}){2,}')
_GROUPED_ANY = re.compile(r'(?:0|[1-9]\d{0,2}(?:[.,]\d{3})*)')


def parse_ocr_number(text):
    """
    Convert the text tesseract recognised (e.g. '$1,234.56') to a float.

    Currency symbols, 'BB' and whitespace are ignored. Thousands separators are told apart
    from decimal separators by their grouping: '1,250' is 1250 and '21,793,801,830' is
    21793801830, whereas '0,05' and '1,5' use the comma as decimal separator. When both
    separators appear, the last one is the decimal separator ('1.234,56' and '1,234.56').

    Returns:
        float, or None if the text is not an unambiguous number. Callers can then try
        another image preparation instead of acting on a misread value.

    """
    number = re.sub(r'[^\d.,]', '', text or '').rstrip('.,')
    if number[:1] in ('.', ','):
        # A separator in front of the number is usually a speck next to the text. It could
        # also be a dropped leading zero ('.29'), so the rest needs its own decimal separator.
        number = number.lstrip('.,')
        if '.' not in number and ',' not in number:
            return None
    number = _remove_thousands_separators(number)
    try:
        return float(number)
    except (TypeError, ValueError):
        return None


def _remove_thousands_separators(number):
    """'1,234.56' -> '1234.56', '0,05' -> '0.05'. None if the separators are ambiguous."""
    if ',' in number and '.' in number:
        decimal_pos = max(number.rfind(','), number.rfind('.'))
        integer_part = number[:decimal_pos]
        if number[decimal_pos] in integer_part or not _GROUPED_ANY.fullmatch(integer_part):
            return None
        return integer_part.replace(',', '').replace('.', '') + '.' + number[decimal_pos + 1:]
    if ',' in number:
        if _GROUPED_BY_COMMA.fullmatch(number):
            return number.replace(',', '')
        return number.replace(',', '.') if number.count(',') == 1 else None
    if number.count('.') > 1:
        return number.replace('.', '') if _GROUPED_BY_DOT.fullmatch(number) else None
    return number


def _to_rgb_array(img):
    """Convert a PIL image of any mode (RGBA, L, P...) or a numpy array to an RGB array"""
    if isinstance(img, Image.Image):
        if img.mode != 'RGB':
            img = img.convert('RGB')
        return np.asarray(img)
    arr = np.asarray(img)
    if arr.ndim == 2:
        return cv2.cvtColor(arr, cv2.COLOR_GRAY2RGB)
    if arr.shape[2] == 4:
        return np.ascontiguousarray(arr[:, :, :3])
    return arr


def _scale_to_text_height(gray):
    """Scale small crops up so that tesseract gets text of a comfortable size"""
    height = gray.shape[0]
    if height >= OCR_TARGET_HEIGHT:
        return gray
    factor = OCR_TARGET_HEIGHT / height
    return cv2.resize(gray, None, fx=factor, fy=factor, interpolation=cv2.INTER_CUBIC)


def _pad(binary):
    return cv2.copyMakeBorder(binary, OCR_PADDING, OCR_PADDING, OCR_PADDING, OCR_PADDING,
                              cv2.BORDER_CONSTANT, value=255)


def _border_pixels(img):
    return np.concatenate([img[0], img[-1], img[:, 0], img[:, -1]])


def prepare_image_otsu(img_orig):
    """
    Prepare an image for OCR with an automatic (Otsu) threshold.

    The result has black text on a white background independent of whether the poker
    client shows light text on a dark background or dark text on a light one. The border
    of a crop around a number is mostly background, which decides the polarity.
    """
    gray = cv2.cvtColor(_to_rgb_array(img_orig), cv2.COLOR_RGB2GRAY)
    gray = _scale_to_text_height(gray)
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    if _border_pixels(binary).mean() < 128:
        binary = 255 - binary
    return Image.fromarray(_pad(binary))


def prepare_image_color_distance(img_orig):
    """
    Prepare an image for OCR by the colour distance of each pixel to the background.

    This separates coloured text from a background of similar brightness (e.g. gold text
    on brown or green text on dark green), where a grey scale threshold has little contrast.
    """
    rgb = _to_rgb_array(img_orig).astype(np.float32)
    background = np.median(_border_pixels(rgb), axis=0)
    distance = np.sqrt(((rgb - background) ** 2).sum(axis=-1))
    distance = (255 * distance / max(float(distance.max()), 1.0)).astype(np.uint8)
    distance = _scale_to_text_height(distance)
    _, binary = cv2.threshold(distance, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    return Image.fromarray(_pad(binary))


# Image preparations for OCR, most reliable first. The fixed thresholds of the original
# implementation are kept as tie breakers.
OCR_PREPARATIONS = (
    prepare_image_otsu,
    prepare_image_color_distance,
    lambda img: prepareImage(img, binarize=True, threshold=125),
    lambda img: prepareImage(img, binarize=True, threshold=76),
)


def get_ocr_number(img_orig, fast=False):
    """
    Return float value from image. -1.0 when OCR failed

    The image is read with several preparations. As soon as two of them agree, that value
    is returned, so the usual case costs two tesseract calls. Otherwise the value read most
    often wins, ties go to the more reliable preparation. With fast=True only the first two
    preparations are used.
    """
    if img_orig is None or img_orig.size[0] < 1 or img_orig.size[1] < 1:
        log.warning("OCR called with an empty image. Please check the table mapping.")
        return -1.0

    preparations = OCR_PREPARATIONS[:2] if fast else OCR_PREPARATIONS
    votes = Counter()
    recognized = []
    for prepare in preparations:
        value = parse_ocr_number(get_ocr_number2(prepare(img_orig)))
        recognized.append(value)
        if value is None:
            continue
        votes[value] += 1
        if votes[value] >= 2:
            return value

    log.debug("OCR candidates: %s", recognized)
    if not votes:
        return -1.0
    most_votes = max(votes.values())
    return next(value for value in recognized if value is not None and votes[value] == most_votes)


def take_screenshot(virtual_box=False):
    """
    Take screenshot directly from screen or via virtualbox

    Args:
        virtual_box: bool

    Returns:
        PIL screenshot

    """
    if not virtual_box:
        log.debug("Calling screen grabber")
        screenshot = ImageGrab.grab()
        log.debug("Direct screenshot successful")

    else:  # virtual_box
        try:
            from poker.tools.vbox_manager import VirtualBoxController
            vb = VirtualBoxController()
            screenshot = vb.get_screenshot_vbox()
            log.debug("Screenshot taken from virtual machine")
        except:
            log.warning(
                "No virtual machine found. Press SETUP to re initialize the VM controller")
            # gui_signals.signal_open_setup.emit(p,L)
            screenshot = ImageGrab.grab()
    return screenshot

def normalize_rect(x1, y1, x2, y2):
    x1_ = min(x1, x2)
    x2_ = max(x1, x2)
    
    y1_ = min(y1,y2)
    y2_ = max(y1,y2)

    return x1_, y1_, x2_, y2_

def check_cropping(screenshot_list, top_left_corner_img):
    """Checks if screenshots are cropped and match the template 'icon'"""
    try:
        log.info("Checking cropping for '" + str(len(screenshot_list)) + "' images.")
        
        if len(screenshot_list) == 0: return False
        if top_left_corner_img.size == 0: return False

        any_too_big = any((s.width > const.CROP_WIDTH and s.height > const.CROP_HEIGHT) for s in screenshot_list)
        if any_too_big: return False
        
        for screenshot in screenshot_list:
            img = pil_to_cv2(screenshot)
            count, _, _, _ = find_template_on_screen(top_left_corner_img, img, 0.01)
            if count != 1: return False
    except Exception as e:
        log.exception(e)
        return False
    finally:
        log.info("Done.")

    return True

def crop_screenshot_with_topleft_corner(original_screenshot, topleft_corner, useSleep = True):
    log.debug("Cropping top left corner")
    img = pil_to_cv2(original_screenshot)
    count, points, _, _ = find_template_on_screen(topleft_corner, img, 0.01)

    if count == 1:
        tlc = points[0]
        log.debug(f"Found to left corner at {tlc}")
        cropped_screenshot = original_screenshot.crop(
            (tlc[0], tlc[1], tlc[0] + const.CROP_WIDTH, tlc[1] + const.CROP_HEIGHT))
        return cropped_screenshot, tlc
    elif count > 1:
        log.warning(
            "Multiple top left corners found. That doesn't work unfortunately at this point. Make sure only one table is visible.")
        return None, None
    else:
        log.warning("No top left corner found")
        if useSleep: sleep(5)
        return None, None


@functools.lru_cache(maxsize=512)
def _decode_template(img):
    template = pil_to_cv2(Image.open(io.BytesIO(img)))
    template.flags.writeable = False  # the cached array is shared between callers
    return template


def binary_pil_to_cv2(img):
    """
    Convert an image stored as png bytes in the table mapping to a cv2 (BGR) image.

    The same templates (e.g. all 52 cards) are checked on every screenshot, so the decoded
    images are cached instead of decoding the png files again each time.
    """
    return _decode_template(bytes(img))


def pil_to_cv2(img):
    """Convert a PIL image of any mode (RGB, RGBA, L, P...) to a cv2 (BGR) image"""
    return cv2.cvtColor(_to_rgb_array(img), cv2.COLOR_RGB2BGR)


def cv2_to_pil(img):
    return Image.fromarray(img)


def rotate_image(image, angle):
    image_center = tuple(np.array(image.shape[1::-1]) / 2)
    rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
    result = cv2.warpAffine(
        image, rot_mat, image.shape[1::-1], flags=cv2.INTER_LINEAR)
    return result


def check_if_image_in_range(img, screenshot, x1, y1, x2, y2, extended=False):
    cropped_screenshot = screenshot.crop((x1, y1, x2, y2))
    cropped_screenshot = pil_to_cv2(cropped_screenshot)
    count, _, _, _ = find_template_on_screen(
        img, cropped_screenshot, 0.01, extended=extended)
    return count >= 1


def is_template_in_search_area(table_dict, screenshot, image_name, image_area, player=None, extended=False):
    template_cv2 = binary_pil_to_cv2(table_dict[image_name])
    if player is not None:
        try:
            search_area = table_dict[image_area][str(player)]
        except KeyError as exc:
            raise KeyError(f"The table mapping is missing data for player {player} and {image_area}."
                           "Please fix the table mapping.") from exc
    else:
        search_area = table_dict[image_area]
    try:
        is_in_range = check_if_image_in_range(template_cv2, screenshot,
                                              search_area['x1'], search_area['y1'], search_area['x2'], search_area['y2'],
                                              extended=extended)
    except Exception as exc:
        x = search_area['x2'] - search_area['x1']
        y = search_area['y2'] - search_area['y1']
        xt = template_cv2.shape[1]
        yt = template_cv2.shape[0]
        if x < xt or y < yt:
            raise RuntimeError(f"Search area for {image_name} {player} is too small. It is {x}x{y} but the template is {xt}x{yt}."
                               ) from exc
        raise RuntimeError(f"The table has an missing template for {image_name}."
                           ) from exc

    return is_in_range


def _overlap_ratio(rect1, rect2):
    """Intersection of two (x, y, width, height) rectangles relative to the smaller one"""
    x1, y1, w1, h1 = rect1
    x2, y2, w2, h2 = rect2
    width = min(x1 + w1, x2 + w2) - max(x1, x2)
    height = min(y1 + h1, y2 + h2) - max(y1, y2)
    if width <= 0 or height <= 0:
        return 0.0
    return width * height / min(w1 * h1, w2 * h2)


def find_templates_in_search_area(table_dict, screenshot, image_names, image_area, threshold=0.01):
    """
    Find which of several templates (e.g. the 52 cards) are visible in a search area.

    Every template is expected at most once (like a card in a deck), so only its best match
    is used. The search area is cropped and converted only once for all templates. If
    templates of different names match the same spot (e.g. two cards with similar looking
    templates), only the best fitting one is kept instead of reporting both.

    Args:
        table_dict: table mapping
        screenshot: PIL image of the (cropped) table
        image_names: names of the templates in the table mapping
        image_area: name of the search area in the table mapping
        threshold: maximum normalized squared difference for a match

    Returns:
        list of (image_name, x, y, score) of the matches from left to right. x and y are
        relative to the search area.

    """
    search_area = table_dict[image_area]
    area = pil_to_cv2(screenshot.crop(normalize_rect(search_area['x1'], search_area['y1'],
                                                     search_area['x2'], search_area['y2'])))
    candidates = _best_match_per_template(table_dict, area, image_names, image_area, threshold)
    accepted = _drop_overlapping_matches(candidates, image_area)
    accepted.sort(key=lambda match: (match[2][0], match[2][1]))
    return [(image_name, rect[0], rect[1], score) for score, image_name, rect in accepted]


def _best_match_per_template(table_dict, area, image_names, image_area, threshold):
    """Returns list of (score, image_name, (x, y, width, height)) of the templates found"""
    candidates = []
    for image_name in image_names:
        if image_name not in table_dict:
            log.warning("The table mapping has no template for %s.", image_name)
            continue
        template = binary_pil_to_cv2(table_dict[image_name])
        height, width = template.shape[:2]
        if area.shape[0] < height or area.shape[1] < width:
            raise RuntimeError(f"Search area {image_area} is too small for {image_name}. It is "
                               f"{area.shape[1]}x{area.shape[0]} but the template is {width}x{height}.")
        result = cv2.matchTemplate(area, template, cv2.TM_SQDIFF_NORMED)
        matches = _distinct_matches(result, threshold, width, height, max_matches=1)
        if matches:
            (x, y), score = matches[0]
            candidates.append((score, image_name, (x, y, width, height)))
    return candidates


def _drop_overlapping_matches(candidates, image_area):
    """Of templates that match the same spot only keep the one with the best score"""
    accepted = []
    for score, image_name, rect in sorted(candidates, key=lambda candidate: candidate[0]):
        clash = next((other for other in accepted if _overlap_ratio(rect, other[2]) > 0.5), None)
        if clash:
            log.debug("Ignoring %s (score %.4f) in %s: same position as %s (score %.4f)",
                      image_name, score, image_area, clash[1], clash[0])
            continue
        accepted.append((score, image_name, rect))
    return accepted


def ocr(screenshot, image_area, table_dict, player=None, fast=False):
    """
    get ocr of area of screenshot

    Args:
        screenshot: pil image
        image_area: area name
        table_dict: table dict
        player: player number started from 0

    Returns:
        float

    """
    if player is not None:
        try:
            search_area = table_dict[image_area][str(player)]
        except KeyError:
            log.error(f"Missing table entry for {image_area} {player}. "
                      f"Please select it from the screenshot and press the corresponding button to add it to the "
                      f"table template. ")
            return 0
    else:
        search_area = table_dict[image_area]
    cropped_screenshot = screenshot.crop(
        normalize_rect(search_area['x1'], search_area['y1'], search_area['x2'], search_area['y2']))
    return get_ocr_float(cropped_screenshot, fast)
