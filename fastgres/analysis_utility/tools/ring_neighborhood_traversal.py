import enum
import warnings
from typing import Optional

import numpy as np

from fastgres.analysis_utility import tool
from fastgres.analysis_utility.tool import TimeGranularity as Tg
from fastgres.analysis_utility.tool import Tool
from fastgres.baseline.hint_sets import Hint
from fastgres.baseline.utility import int_to_binary, chunks


class RingNeighborhoodTraversal(Tool):

    class OperationMode(enum.Enum):
        SUB = 1
        ADD = 0

    def __init__(self, archive_path: str, query_path: str, query_names: list[str], hints: list[Hint],
                 time_granularity: Tg = Tg.SECONDS, op_mode: OperationMode = OperationMode.SUB,
                 use_pseudo_dict: bool = True, spread: int = 1):
        super().__init__(archive_path, query_path)
        self._time_factor = None
        self._time_granularity = None
        self.time_granularity = time_granularity
        self._aggregated_archive = None
        self._query_names = query_names
        self._hints = hints
        self._plot = None
        self._op_mode = op_mode
        self.spread = spread
        if use_pseudo_dict:
            self._used_dict = tool.get_pseudo_labeled_dict(self.archive)
        else:
            self._used_dict = self.archive

    @property
    def op_mode(self):
        return self._op_mode

    @op_mode.setter
    def op_mode(self, new_mode: OperationMode):
        self._op_mode = new_mode

    @property
    def time_granularity(self):
        return self._time_granularity

    @time_granularity.setter
    def time_granularity(self, new_granularity: Tg):
        self._time_granularity = new_granularity
        if self.time_granularity == Tg.SECONDS:
            self._time_factor = 1.0
        elif self.time_granularity == Tg.MINUTES:
            self._time_factor = 60.0
        elif self.time_granularity == Tg.HOURS:
            self._time_factor = 3600.0

    @property
    def query_names(self):
        return self._query_names

    @query_names.setter
    def query_names(self, new_names: list[str]):
        self._query_names = new_names
        self._aggregated_archive = None
        self._plot = None

    @property
    def hints(self):
        return self._hints

    @hints.setter
    def hints(self, new_hints: list[Hint]):
        self._hints = new_hints
        self._aggregated_archive = None
        self._plot = None

    def _calculate_agg_archive(self):
        agg_archive = dict()
        for hint_value in range(2 ** len(self.hints)):
            agg_archive[hint_value] = round(sum([self._used_dict[query_name][str(hint_value)]
                                                 for query_name in self.query_names]) / len(self.query_names), 4)
        return agg_archive

    @property
    def aggregated_archive(self):
        if self._aggregated_archive is None:
            self._aggregated_archive = self._calculate_agg_archive()
        return self._aggregated_archive

    def calculate_one_ring(self, hint_set_int: int, reduced_space: list[int] = None) -> list[int]:
        # [1, 0, 0, 1, 1, 0]
        binary_rep = int_to_binary(hint_set_int)
        if self.op_mode.value not in binary_rep:
            # we are done in the last step
            return []
        # find indices to flip in reversed order as the lowest index has the highest int value
        # [1, 2, 5] -> idx
        # convert back to int
        # [2, 4, 32] -> 2**idx
        # op mode decides whether we add or subtract to reach our neighborhood
        factor = -1 if self.op_mode == self.OperationMode.SUB else 1
        result = [hint_set_int+(factor*2**i) for i, x in enumerate(reversed(binary_rep)) if x == self.op_mode.value]
        # allows to get one ring neighborhoods of reduced search hint set spaces
        if reduced_space is not None:
            result = list(set(result).intersection(set(reduced_space)))
        return result

    def calculate_one_ring_of_order(self, order: list[Hint], reduced_space: list[int] = None) -> list[int]:
        int_order = [hint.value for hint in order]
        # initialization 0 - 63
        current_value = 2**len(self.hints)-1 if self.op_mode == self.OperationMode.SUB else 0
        one_ring = [current_value]
        for idx in range(len(int_order)):
            one_ring.extend(self.calculate_one_ring(current_value, reduced_space))
            # subtract current order item
            current_value = current_value - int_order[idx] if self.op_mode == self.OperationMode.SUB \
                else current_value + int_order[idx]
        return one_ring

    def traverse_hint_sets(self) -> dict[int, float]:
        current_start = [2**len(self.hints)-1] if self.op_mode == self.OperationMode.SUB else [0]
        results = {current_start[0]: self.aggregated_archive[current_start[0]]}
        while True:
            # iterations allow spreading factors to be used
            current_one_ring = list()
            for start in current_start:
                [current_one_ring.append(entry) if entry not in current_one_ring else current_one_ring
                 for entry in self.calculate_one_ring(start)]
            current_one_ring = current_one_ring
            # break condition -> empty one ring, we reached 0 or 63
            if not current_one_ring:
                break

            current_archive = {key: self.aggregated_archive[key] for key in current_one_ring}

            # find minimum within one ring archive
            # min_cumulative_hint_set = min(current_archive, key=current_archive.get)
            ordered_archive = dict(sorted(current_archive.items(), key=lambda item: item[1]))
            min_sets = list(ordered_archive.keys())[:self.spread]

            for min_cumulative_hint_set in min_sets:
                results[min_cumulative_hint_set] = current_archive[min_cumulative_hint_set]
            # adjustment
            # current_start = min_cumulative_hint_set
            current_start = min_sets
        return results

    def traversed_order(self, traversed_hint_sets: dict[int, float]) -> list[int]:
        entries = list(traversed_hint_sets.keys())
        traversed_order = []
        spread_chunks = [_ for _ in chunks(entries[1:], self.spread)]
        for idx in range(len(spread_chunks)):
            current_chunk = spread_chunks[idx]
            if idx == 0:
                appends = entries[0] - current_chunk[0] if self.op_mode == self.OperationMode.SUB \
                    else current_chunk[0] - entries[0]
                traversed_order.append(appends)
            else:
                last_chunk = spread_chunks[idx-1]
                appends = last_chunk[0] - current_chunk[0] if self.op_mode == self.OperationMode.SUB \
                    else current_chunk[0] - last_chunk[0]
                traversed_order.append(appends)
        return traversed_order

    def sort_by_inactive_hints(self, hint_set_dict: dict[int, float], desc: bool = True):
        hints_set_ints = list(hint_set_dict.keys())
        # these are always there and do not count as a choice as long as we do not define other starting positions
        hints_set_ints.remove(2**len(self.hints)-1)
        hints_set_ints.remove(0)
        occurrences = np.subtract(len(hints_set_ints),
                                  np.sum(
                                      np.array([int_to_binary(hint_set_int) for hint_set_int in hints_set_ints]
                                               ), axis=0))
        hint_dict = dict(zip(self.hints, occurrences))
        return dict(sorted(hint_dict.items(), key=lambda item: item[1], reverse=desc))

    def print_results(self):
        print("")

    @tool.font(tool.default_fonts(21))
    def plot(self):
        if self._plot is not None:
            return self._plot

        return self._plot

    @tool.font(tool.default_fonts(21))
    def save_results(self, path: str):
        raise NotImplementedError
