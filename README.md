# Overview
This Python script uses a technique based on sample weighting to analyze Magic: The Gathering games from 17lands.com event data.
It essentially manipulates the sample weights so that the data "behaves" like the decks had a specified property.
For example, if you wanted the data to behave like a 16 land deck
the weights are higher for games where the player is mana screwed and lower for games where the user gets flooded.

The primary method for doing this analysis is ``weighted_sample_analysis`` which has the following signature:
```Python
def weighted_sampling_analysis(
        cards_of_interest: CardFilter,
        start_range: int,
        end_range: int,
        replacement_cards: Optional[CardFilter] = None,
        game_filter: Optional[GameFilter] = None,
        expansion: str = "BLB",
        data_path: Optional[Path] = None,
        card_path: Path = PATH / "cards.csv",
        abilities_path: Path = PATH / "abilities.csv",
        show_plots: bool = False
) -> WeightedSamplingResult:
```
* `cards_of_interest: CardFilter` this is a function that defines what types of cards you are testing.
For example, if you wanted to test how decks do with different number of lands in the deck
you would pass in `LandFilter` which returns `True` if the card is a land.
* `start_range: int, end_range: int` these two define the range on the number of cards of interest in the deck distributions you are targeting.
For example if you want to see what the results for 15, 16, 17, and 18 land decks look like use `start_range = 15` and `end_range = 18`.
*  `replacement_cards: Optional[CardFilter] = None` this card filter defines what we replace the cards of interest with in the targeted distribution.
Suppose, we want to replace lands with creatures and we are looking at games for a deck 17 lands and 10 creatures.
Then if we are targeting 16 lands we would weight the games so that they behave as if the deck had 11 creatures.
And if we are targeting 18 lands we would weight the games so that they behave as if the deck had 9 creatures.
By default this is set to other non-land cards, meaning any non-land that does not pass the `cards_of_interest` filter.
* `expansion: str = "BLB", data_path: Optional[Path] = None` these define where the script will look for the 17lands data is to analyze.
If `data_path` is set it will look there for the data.
Otherwise it will look in the same directory as the script for a file named `replay_data_public.{expansion}.TradDraft.csv`.
* `card_path: Path = PATH / "cards.csv", abilities_path: Path = PATH / "abilities.csv"`
these define where the script will look for the card and abilities data from 17lands.
By default it will look in the same directory as the script for files named `cards.csv` and `abilities.csv`.
* `show_plots: bool = False` lets the script know if you want plots to pop up when the data is done, false by default.
* `WeightedSamplingResult` contains `single_coi_count_results` which is a dict mapping number of cards of interest (e.g. 16 lands, 17 lands, 18 lands, etc.) to `SingleCardsOfInterestCountResult`.
`SingleCardsOfInterestCountResult` contains:
  * `total_win_rate: float` the weighted total winrate for the targeted number of cards of interest.
  * `cards_of_interest_seen_rates: np.ndarray` a 2D array representing weighted the percentage of games that end with a given number of total cards seen and given number of cards of interest seen.
    The first axis is the total number of cards drawn in the game (index 0 is 7 cards seen) and the second axis is the number of cards of interest seen.
  * `win_rates_per_cards_of_interest_seen: np.ndarray` a 2D array winrate of games that end with a given number of total cards seen and given number of cards of interest seen.
    Again the first axis is the total number of cards drawn in the game (index 0 is 7 cards seen) and the second axis is the number of cards of interest seen.
  * `mulligan_rates: np.ndarray` a 1D array showing weighted percentage of games with 0 mulligans, 1 mulligan, 2 mulligans, etc.

# CardFilter and DeckFilter

`CardFilter` and `GameFilter` are two abstract classes that filter out cards or games, respectively, that you are not interested in.
In both cases you need to implement the `__call__` function.
For `CardFilter` the function takes in a `Card` object and for `GameFilter` it takes in a `GameData` object and both return a Boolean.
The inputs contain data related to the card and game respectively.

Both abstract classes implement a `negate` function that returns a new filter that does the opposite of the filter it is called on.
They also have a `also` function which takes in another filter and returns a new filter that only returns true if both of the original filters return true.

# Getting the 17lands data

There are four data files you'll probably need to download from 17lands in order to use this code.

1. Replay data: Found at https://www.17lands.com/public_datasets.
In the DataSets table at the top find the expansion you want to analyze and find the row with "TradDraft" in the "Format" column.
Then click the link under the "Replay Data" column.
This should download a ZIP file which you should extract to the same directory you have the script saved.
2. Card data: Also found at https://www.17lands.com/public_datasets.
At the very bottom of the page in the "Cards List" table click the "Cards" link.
Move the csv to the same directory you have the script saved.
3. Abilities data: Also found https://www.17lands.com/public_datasets.
Again at the very bottom of the page in the "Cards List" table click the "Abilities" link.
Again move the csv to the same directory you have the script saved.
4. Card winrate data: This should only be necessary if you are using the `ReplacementLevelNonLands` card filter which should be used for analyzing land counts.
Found at https://www.17lands.com/card_data.
Choose the format you want to analyze (I used PremierDraft data for no real reason but if you use TradDraft you probably should change the `replacement_level_winrate`).
Click "Export data" and "Download as CSV".
Move downloaded file to same directory as script and rename file to "card_ratings.{format}.CSV".

# Examples

## BLB Land Analysis

```Python
class LandFilter(CardFilter):
    def __call__(self, card: Card) -> bool:
        return "Land" in card.types

class BLBNoDeckManipulation(GameFilter):
  deck_manipulation_cards = {"Carrot Cake", "Diresight", "Fabled Passage", "Fountainport Bell",
                             "Glarb, Calamity's Augur", "Gossip's Talent", "Heaped Harvest", "Hidden Grotto",
                             "Lightshell Duo", "Lilypad Village", "Mind Drill Assailant", "Mindwhisker",
                             "Psychic Whorl", "Rabbit Response", "Spellgyre", "Starlit Soothsayer",
                             "Thornvault Forager", "Valley Questcaller", "Veteran Guardmouse"}

  def __call__(self, data: GameData) -> bool:
      return all(
          card.name not in self.deck_manipulation_cards
          for card in data.deck
      )

result = weighted_sampling_analysis(
    cards_of_interest=LandFilter(),
    start_range=12,
    end_range=20,
    replacement_cards=ReplacementLevelNonLands(),
    game_filter=BLBNoDeckManipulation(),
    show_plots=True,
)
```

## BLB Land Analysis for BG decks

```Python
class DeckColorFilter(GameFilter):
    def __init__(self, colors: str):
        self.colors = colors

    def __call__(self, data: GameData) -> bool:
        return sorted(data.colors) == sorted(self.colors)

result = weighted_sampling_analysis(
    cards_of_interest=LandFilter(),
    start_range=12,
    end_range=20,
    replacement_cards=ReplacementLevelNonLands(),
    game_filter=BLBNoDeckManipulation().also(DeckColorFilter("GB")),
    show_plots=True,
)
```

## BLB Two Drop Analysis

```Python
class TwoDropFilter(CardFilter):
    def __call__(self, card: Card) -> bool:
        return "Creature" in card.types and card.mana_value == 2

result = weighted_sampling_analysis(
    cards_of_interest=TwoDropFilter(),
    start_range=2,
    end_range=8,
    game_filter=BLBNoDeckManipulation(),
    show_plots=True,
)
```

## DMU Land Analysis

```Python
class DMUNoDeckManipulation(GameFilter):
    deck_manipulation_cards = {"Automatic Librarian", "Crystal Grotto", "Djinn of the Fountain", "Furious Bellow",
                               "Guardian of New Benalia", "Herd Migration", "Impede Momentum", "Jaya's Firenado",
                               "Joint Exploration", "Lagomos, Hand of Hatred", "Micromancer", "Phyrexian Vivisector",
                               "Runic Shot", "Samite Herbalist", "Scout the Wilderness", "Shadow-Rite Priest",
                               "Shield-Wall Sentinel", "Slimefoot's Survey", "Sprouting Goblin", "The Cruelty of Gix",
                               "The Weatherseed Treaty", "Threats Undetected", "Tidepool Turtle",
                               "Urza Assembles the Titans", "Uurg, Spawn of Turg", "Weatherlight Compleated"}

    def __call__(self, data: GameData) -> bool:
        return all(
            card.name not in self.deck_manipulation_cards
            for card in data.deck
        )

result = weighted_sampling_analysis(
    cards_of_interest=LandFilter(),
    start_range=12,
    end_range=20,
    replacement_cards=ReplacementLevelNonLands(expansion="DMU"),
    game_filter=DMUNoDeckManipulation(),
    expansion="DMU"
    show_plots=True,
)
```

# Pitfalls

* The 17lands event data doesn't contain information about the result of scrying, surveiling, or searching your library.
The results of these effects would need to be taken into account in order to get accurate results.
Initially to solve this problem, I used a general `NoDeckManipulation` filter which filtered out games where one of these effects happened.
However this meant that shorter games were less likely to be filtered out.
So, instead I created filters, like `BLBNoDeckManipulation`, that filter out all games where the deck has a card with one of these effects.
The issue is that the actual card text isn't in the data so I just used a scryfall search to creaete these filters manually.
This process should probably be automated but I didn't do that.
Unfortunately, this filter significantly lowers the number of games analyzed.
* The data needs to be somewhat similar to the deck distributions you are targeting.
This method can't add weight to games that don't exist.
For example, suppose you are testing a very narrow set of cards of interest where most decks have maybe 1 or 2.
The analysis is going to under-represent games where 3 cards of interest are drawn.
In this case consider filtering by decks that contain significant numbers of your cards of interest.
Also the more your target deck differs from real decks in the data the higher the weights are going to be, effectively lowering your sample rate.
