"""Replay saved screenshots through the scraper and into the fly.

`FlyDecision` had never been run on a real scraped table - every number in doc/flybrain.md
comes from synthetic hands or from Kuhn. The gap that matters is not whether the fly can
choose an action, which the offline harness covers, but whether it can consume what the
*scraper* actually produces: a table object whose fields are missing, zero, or a string
where a float was expected, on a screen the fly has never seen.

Finding that out on a live table costs real hands and invalidates whatever record was
being collected up to that point. The repository already ships screenshots and the
scraper runs on them headlessly, so it can be found out for free.

What this checks, and what it does not
--------------------------------------
It runs the real `TableScraper` over a real screenshot, wraps the result in the attribute
surface the fly reads, and asks the fly for a decision. So it covers perception ->
features -> LIF -> readout -> a legal action.

It does not check that the decision is *good*. Equity is not scraped - it comes from the
Monte Carlo downstream - so a replay supplies it, and a replayed decision is therefore
conditional on a value the scraper never produced. Treat the output as evidence the path
works end to end, not as a measurement of play.
"""

# pylint: disable=too-many-arguments,too-many-positional-arguments,too-many-instance-attributes
import logging
import os

from poker.flybrain import decoding
from poker.flybrain.encoding import TableView

log = logging.getLogger(__name__)

# Screenshots shipped with the repository, and the table definition each was captured on.
# A table definition is a set of pixel coordinates, so it only reads a screenshot taken
# from the client, theme and resolution it was mapped for; pairing them wrongly produces
# a scrape full of zeroes rather than an error.
FIXTURES = (
    ('53269218_PreFlop_0.png', 'Official Party Poker', 'PreFlop'),
    ('988359671_PreFlop_0.png', 'Official Party Poker', 'PreFlop'),
    ('ps473830744_Flop_1.png', 'Official Poker Stars', 'Flop'),
    ('ggpk6ocr.png', 'Official GG Poker', 'Flop'),
)


class ScrapedTable(TableView):
    """A scraped table wearing the surface `features_from_table` and the fly expect.

    The scraper fills a `TableScraper`, not a `TableView`, and the names only partly
    overlap. Mapping here rather than teaching the encoder about the scraper keeps the
    offline tasks and the live path reading the same attribute surface.
    """

    def __init__(self, scraper, equity=0.5, stage='PreFlop', big_blind=1.0):
        total_pot = float(getattr(scraper, 'total_pot', 0.0) or 0.0)
        call_value = float(getattr(scraper, 'call_value', 0.0) or 0.0)
        funds = getattr(scraper, 'player_funds', None) or [0.0]
        super().__init__(equity, total_pot, call_value, float(funds[0] or 0.0), stage,
                         opponents=max(1, len(getattr(scraper, 'players_in_game', []) or [1]) - 1),
                         big_blind=big_blind)
        # Buttons decide what is legal, so they come from the scrape rather than a guess.
        self.checkButton = bool(getattr(scraper, 'check_button', False))
        self.callButton = bool(getattr(scraper, 'call_button', False))
        self.betButton = bool(getattr(scraper, 'raise_button', False))
        self.allInCallButton = bool(getattr(scraper, 'all_in_call_button', False))
        self.round_pot_value = float(getattr(scraper, 'round_pot', 0.0) or 0.0)
        self.my_cards = list(getattr(scraper, 'my_cards', []) or [])
        self.table_cards = list(getattr(scraper, 'table_cards', []) or [])


def scrape_screenshot(screenshot_path, table_name, table_dict=None):
    """Run the scraper over one screenshot and return the populated TableScraper."""
    from PIL import Image  # pylint: disable=import-outside-toplevel
    from poker.scraper.table_scraper import TableScraper  # pylint: disable=import-outside-toplevel
    from poker.tools.mongo_manager import MongoManager  # pylint: disable=import-outside-toplevel

    if table_dict is None:
        table_dict = MongoManager().get_table(table_name)
    scraper = TableScraper(table_dict)
    scraper.screenshot = Image.open(screenshot_path)
    scraper.crop_from_top_left_corner()

    # Each of these populates part of the table. They are called individually rather than
    # through main.py's chain because that chain also drives the mouse and the GUI.
    for step in ('is_my_turn', 'lost_everything', 'get_my_cards2', 'get_table_cards2',
                 'get_dealer_position2', 'get_players_in_game', 'get_pots',
                 'get_players_funds', 'get_call_value', 'get_raise_value',
                 'has_all_in_call_button', 'has_call_button', 'has_raise_button'):
        try:
            getattr(scraper, step)()
        except Exception as exc:  # pylint: disable=broad-except
            # One unreadable region should not hide what the rest of the scrape found.
            log.warning("scrape step %s failed on %s (%s)", step,
                        os.path.basename(screenshot_path), exc)
    return scraper


def replay_one(brain, screenshot_path, table_name, equity=0.5, stage='PreFlop',
               table_dict=None):
    """Scrape one screenshot and take a fly decision on it.

    Returns a dict of what was read and what was chosen, for logging or assertion.
    """
    scraper = scrape_screenshot(screenshot_path, table_name, table_dict=table_dict)
    table = ScrapedTable(scraper, equity=equity, stage=stage)
    allowed = decoding.legal_actions(table)
    observation = brain.decide(table, allowed=allowed)
    brain.abandon_hand()        # a replay is not a hand; nothing to reinforce

    return {
        'screenshot': os.path.basename(screenshot_path),
        'table_name': table_name,
        'my_cards': table.my_cards,
        'table_cards': table.table_cards,
        'total_pot': table.totalPotValue,
        'call_value': table.minCall,
        'my_funds': table.myFunds,
        'buttons': {'check': table.checkButton, 'call': table.callButton,
                    'raise': table.betButton, 'allin_call': table.allInCallButton},
        'allowed': list(allowed),
        'action': observation.action,
        'kc_active': float((observation.kc_counts > 0).mean()),
        'mbon_max': float(observation.mbon_rates.max()),
    }


def format_replay(row):
    """One line per replayed screenshot."""
    return (f"{row['screenshot']:28s} cards={','.join(row['my_cards']) or '-':8s} "
            f"pot={row['total_pot']:>7.2f} call={row['call_value']:>6.2f} "
            f"KC={row['kc_active']:.3f} MBON={row['mbon_max']:>7.2f} "
            f"-> {row['action']}  (legal: {', '.join(row['allowed'])})")
