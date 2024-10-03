import csv
import math
import scrython
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import List, Iterable, Dict, Tuple, Optional, Set, Container

import numpy as np

import matplotlib.pyplot as plt

PATH = Path(__file__).parent
DEFAULT_EXPANSION = "BLB"


@dataclass(frozen=True)
class Card:
    name: str
    rarity: str
    color_identity: str
    mana_value: int
    types: str


class CardFilter(ABC):
    @abstractmethod
    def __call__(
            self,
            card: Card,
    ) -> bool:
        pass

    def negate(self) -> "CardFilter":
        neg = type("NegationCardFilter", (CardFilter, object), {"__call__": lambda inner_self, card: not self(card)})
        return neg()

    def also(self, other_filter: "CardFilter") -> "CardFilter":
        new_filter = type(
            "UnionCardFilter",
            (CardFilter, object),
            {"__call__": lambda inner_self, card: self(card) and other_filter(card)}
        )
        return new_filter()


class SpecifiedCardsFilter(CardFilter):
    def __init__(self, cards: Set[str]):
        self.cards = cards

    def __call__(self, card: Card) -> bool:
        return card.name in self.cards


class LandFilter(CardFilter):
    def __call__(self, card: Card) -> bool:
        return "Land" in card.types


class CreatureFilter(CardFilter):
    def __call__(self, card: Card) -> bool:
        return "Creature" in card.types


class TwoDropFilter(CardFilter):
    def __call__(self, card: Card) -> bool:
        return "Creature" in card.types and card.mana_value == 2


class BigCardFilter(CardFilter):
    def __call__(self, card: Card) -> bool:
        return card.mana_value >= 5


class OtherNonLands(CardFilter):
    def __init__(self, original_filter: CardFilter):
        self.original_filter = original_filter

    def __call__(self, card: Card) -> bool:
        return "Land" not in card.types and not self.original_filter(card)


@dataclass
class GameData:
    win: bool
    on_play: bool
    deck: List[Card]
    colors: str
    splashes: str
    cards_of_interest_in_deck: int
    replacement_cards_in_deck: int
    mulligans: int
    candidate_hands: List[List[Card]]
    cards_of_interest_in_candidates: List[int]
    replacement_cards_in_candidates: List[int]
    cards_drawn: List[Card]
    cards_of_interest_drawn: int
    replacement_cards_drawn: int
    abilities_used: List[str]


def extract_game_data(
        data: List[str],
        column_name_dict: Dict[str, int],
        card_deck_indices: Dict[str, int],
        card_id_dict: Dict[int, Card],
        card_name_dict: Dict[str, Card],
        cards_of_interest: CardFilter,
        replacement_cards: CardFilter,
        extract_abilities: bool = False,
        abilities: Optional[Dict[int, str]] = None,
) -> GameData:
    win = data[column_name_dict["won"]] == "True"
    on_play = data[column_name_dict["on_play"]] == "True"
    deck = get_deck_list(data, card_deck_indices, card_name_dict)
    colors = data[column_name_dict["main_colors"]]
    splashes = data[column_name_dict["splash_colors"]]
    cards_of_interest_in_deck = get_filter_count(deck, cards_of_interest)
    replacement_cards_in_deck = get_filter_count(deck, replacement_cards)

    # handle mulligans
    mulligans = int(data[column_name_dict["num_mulligans"]])
    candidate_hands = []
    cards_of_interest_in_candidates = []
    replacement_cards_in_candidates = []
    for i in range(mulligans):
        index = column_name_dict[f"candidate_hand_{i + 1}"]
        cards = parse_cards_from_ids(data[index], card_id_dict)
        candidate_hands.append(cards)
        cards_of_interest_in_candidates.append(get_filter_count(cards, cards_of_interest))
        replacement_cards_in_candidates.append(get_filter_count(cards, replacement_cards))

    # handle cards drawn
    cards_drawn = parse_cards_from_ids(data[column_name_dict[f"candidate_hand_{mulligans + 1}"]], card_id_dict)
    for i in range(1, 31):
        cards_drawn += parse_cards_from_ids(data[column_name_dict[f"user_turn_{i}_cards_drawn"]], card_id_dict)
    cards_of_interest_drawn = get_filter_count(cards_drawn, cards_of_interest)
    replacement_cards_drawn = get_filter_count(cards_drawn, replacement_cards)

    # handle abilities
    abilities_used = []
    if extract_abilities and abilities is not None:
        for i in range(1, 31):
            abilities_used += parse_abilities_from_ids(
                data[column_name_dict[f"user_turn_{i}_user_abilities"]],
                abilities
            )

    return GameData(
        win=win,
        on_play=on_play,
        deck=deck,
        colors=colors,
        splashes=splashes,
        cards_of_interest_in_deck=cards_of_interest_in_deck,
        replacement_cards_in_deck=replacement_cards_in_deck,
        mulligans=mulligans,
        candidate_hands=candidate_hands,
        cards_of_interest_in_candidates=cards_of_interest_in_candidates,
        replacement_cards_in_candidates=replacement_cards_in_candidates,
        cards_drawn=cards_drawn,
        cards_of_interest_drawn=cards_of_interest_drawn,
        replacement_cards_drawn=replacement_cards_drawn,
        abilities_used=abilities_used,
    )


class GameFilter(ABC):
    @abstractmethod
    def __call__(self, data: GameData) -> bool:
        pass

    def negate(self) -> "GameFilter":
        neg = type("NegationGameFilter", (GameFilter, object), {"__call__": lambda inner_self, data: not self(data)})
        return neg()

    def also(self, other_filter: "GameFilter") -> "GameFilter":
        new_filter = type(
            "UnionGameFilter",
            (GameFilter, object),
            {"__call__": lambda inner_self, data: self(data) and other_filter(data)}
        )
        return new_filter()


class OnPlayGameFilter(GameFilter):
    def __call__(self, data: GameData) -> bool:
        return data.on_play


class TwoColorFilter(GameFilter):
    def __call__(self, data: GameData) -> bool:
        return len(data.colors) == 2 and data.splashes == ""


class MultiColorFilter(GameFilter):
    def __call__(self, data: GameData) -> bool:
        result = len(data.colors) > 2 or (len(data.colors) == 2 and data.splashes != "")
        return result


class ContainsCards(GameFilter):
    def __init__(self, card_counts: Dict[str, int]):
        self.card_counts = card_counts

    def __call__(self, data: GameData) -> bool:
        for card, count in self.card_counts.items():
            if count != len([c for c in data.deck if c.name == card]):
                return False
        return True


class ExcludeCards(GameFilter):
    def __init__(self, card_names: Container[str]):
        self.card_names = card_names

    def __call__(self, data: GameData) -> bool:
        return not any(card.name in self.card_names for card in data.deck)


class DeckColorFilter(GameFilter):
    def __init__(self, colors: str):
        self.colors = colors

    def __call__(self, data: GameData) -> bool:
        return sorted(data.colors) == sorted(self.colors)


class DeckSizeFilter(GameFilter):
    def __init__(self, size: int):
        self.size = size

    def __call__(self, data: GameData) -> bool:
        return len(data.deck) == self.size


class NoDeckManipulation(GameFilter):
    def __init__(self, expansion: str = DEFAULT_EXPANSION):
        excluded_cards = scrython.cards.search.Search(
            q=f"set:{expansion} (o:scry or o:surveil or o:\"search your library\")"
        )
        excluded_card_names = {card["name"] for card in excluded_cards.scryfallJson["data"]}
        self.delegate = ExcludeCards(excluded_card_names)

    def __call__(self, data: GameData) -> bool:
        return self.delegate(data)


def generate_card_dicts(
        path: Path = PATH / "cards.csv"
) -> Tuple[Dict[int, Card], Dict[str, Card]]:
    id_result: Dict[int, Card] = dict()
    name_result: Dict[str, Card] = dict()
    with open(path, mode="r") as file:
        csv_reader = csv.reader(file)
        column_names = next(csv_reader)
        for card_data in csv_reader:
            card = Card(
                name=card_data[2],
                rarity=card_data[3],
                color_identity=card_data[4],
                mana_value=int(card_data[5]),
                types=card_data[6],
            )
            id_result[int(card_data[0])] = card
            name_result[card.name] = card
    return id_result, name_result


def generate_card_winrates(
        expansion: str = DEFAULT_EXPANSION,
        path: Optional[Path] = None,
) -> Dict[str, Optional[float]]:
    if path is None:
        path = PATH / f"card-ratings.{expansion}.csv"
    result = dict()
    with open(path, mode="r") as file:
        csv_reader = csv.reader(file)
        column_names = next(csv_reader)
        for data in csv_reader:
            gihwr = data[15]
            result[data[0]] = None if gihwr == "" else float(gihwr.strip('%')) / 100
    return result


class ReplacementLevelNonLands(CardFilter):
    def __init__(
            self,
            replacement_level_winrate: float = 0.55,
            expansion: str = DEFAULT_EXPANSION,
            card_ratings_path: Optional[Path] = None,
    ):
        if card_ratings_path is None:
            card_ratings_path = PATH / f"card-ratings.{expansion}.csv"
        self.card_winrates = generate_card_winrates(path=card_ratings_path)
        self.replacement_level_winrate = replacement_level_winrate

    def __call__(self, card: Card) -> bool:
        return "Land" not in card.types and (
                self.card_winrates[card.name] is None
                or self.card_winrates[card.name] <= self.replacement_level_winrate)


def generate_abilities_dict(
        path: Path = PATH / "abilities.csv"
) -> Dict[int, str]:
    result = dict()
    with open(path, mode="r") as file:
        csv_reader = csv.reader(file)
        column_names = next(csv_reader)
        for ability_data in csv_reader:
            result[int(ability_data[0])] = ability_data[1]
    return result


def get_event_data_generator(
        expansion: str = DEFAULT_EXPANSION,
        path: Path = None,
) -> Iterable[List[str]]:
    if path is None:
        path = PATH / f"replay_data_public.{expansion}.TradDraft.csv"
    with open(path, mode="r") as file:
        csv_reader = csv.reader(file)
        for line in csv_reader:
            yield line


def convert_col_names_dict(col_names: List[str]) -> Dict[str, int]:
    return {name: i for i, name in enumerate(col_names)}


def get_deck_list(
        data: List[str],
        card_deck_indices: Dict[str, int],
        card_name_dict: Dict[str, Card],
) -> List[Card]:
    result: List[Card] = []
    for name, index in card_deck_indices.items():
        count = int(data[index])
        result += [card_name_dict[name]] * count
    return result


def parse_cards_from_ids(
        ids: str,
        card_id_dict: Dict[int, Card],
) -> List[Card]:
    if not ids or ids == "-1":
        return []
    return [card_id_dict[int(card_id)] for card_id in ids.split(sep="|")]


def parse_abilities_from_ids(
        ids: str,
        abilities: Dict[int, str],
) -> List[str]:
    if not ids or ids == "-1":
        return []
    return [abilities.get(int(ability_id), "") for ability_id in ids.split(sep="|")]


def get_filter_count(cards: List[Card], card_filter: CardFilter) -> int:
    return sum(card_filter(card) for card in cards)


def get_prob_ratios_from_counts(
        total_cards: int,
        total_cards_of_interest: int,
        total_replacement_cards: int,
        sample_size: int,
        sample_cards_of_interest: int,
        sample_replacement_cards: int,
        start_range: int,
        end_range: int,
) -> np.array:
    if sample_size - sample_cards_of_interest - sample_replacement_cards \
            > total_cards - total_cards_of_interest - total_replacement_cards \
            or sample_cards_of_interest > total_cards_of_interest \
            or sample_replacement_cards > total_replacement_cards:
        return np.array([0.0] * (end_range - start_range + 1))
    denominator = math.comb(total_cards - total_cards_of_interest - total_replacement_cards,
                            sample_size - sample_cards_of_interest - sample_replacement_cards) \
                  * math.comb(total_cards_of_interest, sample_cards_of_interest) \
                  * math.comb(total_replacement_cards, sample_replacement_cards)
    result = []
    for i in range(start_range, end_range + 1):
        adjusted_replacement_cards_in_deck = total_replacement_cards + total_cards_of_interest - i
        if i < sample_cards_of_interest \
                or adjusted_replacement_cards_in_deck < sample_replacement_cards \
                or 40 - i - adjusted_replacement_cards_in_deck \
                < sample_size - sample_cards_of_interest - sample_replacement_cards:
            result.append(0.0)
        else:
            combos = math.comb(40 - i - adjusted_replacement_cards_in_deck,
                               sample_size - sample_cards_of_interest - sample_replacement_cards) \
                     * math.comb(i, sample_cards_of_interest) \
                     * math.comb(adjusted_replacement_cards_in_deck, sample_replacement_cards)
            result.append(combos)
    return np.array(result) / denominator


def get_prob_ratios_from_data(
        data: GameData,
        start_range: int,
        end_range: int,
        total_games: int,
        coi_in_decks: List[int],
) -> np.array:
    result = get_prob_ratios_from_counts(
        len(data.deck),
        data.cards_of_interest_in_deck,
        data.replacement_cards_in_deck,
        len(data.cards_drawn),
        data.cards_of_interest_drawn,
        data.replacement_cards_drawn,
        start_range,
        end_range,
    )

    for i in range(data.mulligans):
        result *= get_prob_ratios_from_counts(
            len(data.deck),
            data.cards_of_interest_in_deck,
            data.replacement_cards_in_deck,
            len(data.candidate_hands[i]),
            data.cards_of_interest_in_candidates[i],
            data.replacement_cards_in_candidates[i],
            start_range,
            end_range
        )

    max_coi_seen = max(max(data.cards_of_interest_in_candidates, default=0), data.cards_of_interest_drawn)
    max_non_coi_seen = max(
        max([7 - c for c in data.cards_of_interest_in_candidates], default=0),
        len(data.cards_drawn) - data.cards_of_interest_drawn
    )
    impossible_games = sum(coi_in_decks[:max_coi_seen]) + sum(coi_in_decks[41 - max_non_coi_seen:])
    if impossible_games != total_games:
        result *= total_games / (total_games - impossible_games)

    return result


@dataclass
class SingleCardsOfInterestCountResult:
    total_win_rate: float
    cards_of_interest_seen_rates: np.ndarray
    win_rates_per_cards_of_interest_seen: np.ndarray
    mulligan_rates: np.ndarray


@dataclass
class WeightedSamplingResult:
    single_coi_count_results: Dict[int, SingleCardsOfInterestCountResult]


def plot_result(result: WeightedSamplingResult) -> None:
    x, y = [], []
    for num_cards_of_interest, single_coi_count_result in iter(sorted(result.single_coi_count_results.items())):
        x.append(num_cards_of_interest)
        y.append(single_coi_count_result.total_win_rate)

    plt.bar(x, y)
    for x_val, y_val in zip(x, y):
        plt.text(x_val - 0.5, y_val, f"{y_val * 100:.2f}%")

    plt.title("Cards of Interest Winrates")
    plt.xlabel("Cards of Interest")
    plt.ylabel("Winrate")

    plt.show()


def weighted_sampling_analysis(
        cards_of_interest: CardFilter,
        start_range: int,
        end_range: int,
        replacement_cards: Optional[CardFilter] = None,
        game_filter: Optional[GameFilter] = None,
        expansion: str = DEFAULT_EXPANSION,
        data_path: Optional[Path] = None,
        card_path: Path = PATH / "cards.csv",
        extract_abilities: bool = False,
        abilities_path: Optional[Path] = None,
        show_plots: bool = False
) -> WeightedSamplingResult:
    if data_path is None:
        data_path = PATH / f"replay_data_public.{expansion}.TradDraft.csv"
    if replacement_cards is None:
        replacement_cards = OtherNonLands(cards_of_interest)
    if extract_abilities and abilities_path is None:
        abilities_path = PATH / "abilities.csv"

    card_id_dict, card_name_dict = generate_card_dicts(card_path)
    if extract_abilities:
        abilities = generate_abilities_dict(abilities_path)
    else:
        abilities = dict()
    data_generator = get_event_data_generator(path=data_path)
    range_size = end_range - start_range + 1
    total_weights = np.array([0.0 for _ in range(range_size)])
    win_weights = np.array([0.0 for _ in range(range_size)])
    cards_of_interest_seen_weights = np.array(
        [[[0.0 for _ in range(range_size)] for _ in range(end_range + 1)] for _ in range(34)])
    win_weights_per_cards_of_interest_seen = np.array(
        [[[0.0 for _ in range(range_size)] for _ in range(end_range + 1)] for _ in range(34)])
    mulligan_weights = np.array(
        [[0.0 for _ in range(range_size)] for _ in range(7)])
    column_names = next(data_generator)
    column_name_dict: Dict[str, int] = {name: i for i, name in enumerate(column_names)}
    card_deck_indices = {name[5:]: i for name, i in column_name_dict.items() if name.startswith("deck_")}

    total_games = 0
    count = 0
    games = []
    coi_in_decks = [0] * 41
    for data in data_generator:
        game_data = extract_game_data(
            data=data,
            column_name_dict=column_name_dict,
            card_deck_indices=card_deck_indices,
            card_id_dict=card_id_dict,
            card_name_dict=card_name_dict,
            cards_of_interest=cards_of_interest,
            replacement_cards=replacement_cards,
            extract_abilities=extract_abilities,
            abilities=abilities,
        )
        count += 1
        if count % 10000 == 0:
            print(f"Games loaded: {count}")
        if len(game_data.deck) != 40 or (game_filter and not game_filter(game_data)):
            continue
        games.append(game_data)
        total_games += 1
        coi_in_decks[game_data.cards_of_interest_in_deck] += 1
    print(f"{total_games}/{count} games extracted into data set")

    count = 0
    for game_data in games:
        weights = get_prob_ratios_from_data(
            data=game_data,
            start_range=start_range,
            end_range=end_range,
            total_games=total_games,
            coi_in_decks=coi_in_decks,
        )
        total_weights += weights
        mulligan_weights[game_data.mulligans] += weights
        if len(game_data.cards_drawn) <= 40 and game_data.cards_of_interest_drawn <= end_range:
            cards_of_interest_seen_weights[len(game_data.cards_drawn) - 7, game_data.cards_of_interest_drawn] += weights
        if game_data.win:
            win_weights += weights
            if len(game_data.cards_drawn) <= 40 and game_data.cards_of_interest_drawn <= end_range:
                win_weights_per_cards_of_interest_seen[
                    len(game_data.cards_drawn) - 7, game_data.cards_of_interest_drawn] += weights
        count += 1
        if count % 10000 == 0:
            print(f"Games analyzed: {count}")

    win_rate_per_cards_of_interest_seen = np.divide(win_weights_per_cards_of_interest_seen,
                                                    cards_of_interest_seen_weights,
                                                    out=np.zeros_like(win_weights_per_cards_of_interest_seen),
                                                    where=cards_of_interest_seen_weights != 0)
    result = WeightedSamplingResult(
        single_coi_count_results={
            i + start_range: SingleCardsOfInterestCountResult(
                total_win_rate=win_weights[i] / total_weights[i],
                cards_of_interest_seen_rates=cards_of_interest_seen_weights[:, :, i] / total_weights[i],
                win_rates_per_cards_of_interest_seen=win_rate_per_cards_of_interest_seen[:, :, i],
                mulligan_rates=mulligan_weights[:, i] / total_weights[i],
            )
            for i in range(range_size)
        }
    )
    print(f"Done! Total games analyzed: {count}")

    if show_plots:
        plot_result(result)

    return result


if __name__ == "__main__":
    weighted_sampling_analysis(
        cards_of_interest=LandFilter(),
        start_range=12,
        end_range=20,
        replacement_cards=ReplacementLevelNonLands(),
        game_filter=NoDeckManipulation(),
        show_plots=True,
    )
