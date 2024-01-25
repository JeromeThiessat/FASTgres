import abc
import enum
import itertools
import os

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap

from fastgres.baseline.utility import load_json
from fastgres.workloads.workload import Workload


class TimeGranularity(enum.Enum):
    SECONDS = "s"
    MINUTES = "m"
    HOURS = "h"


def set_fonts(size):
    # https://stackoverflow.com/a/14971193
    plt.rc('font', size=size)
    plt.rc('axes', titlesize=size)
    plt.rc('axes', labelsize=size)
    plt.rc('xtick', labelsize=size)
    plt.rc('ytick', labelsize=size)
    plt.rc('legend', fontsize=size)
    plt.rc('figure', titlesize=size)


def resent_fonts():
    set_fonts(10)


def get_rgb_colors():
    return [
        "#F8D35E",
        "#BFBFBF",
        "#F47264",
        "#BCE3F5",
        "#4DB5E6",
        "#1982B3",
        "#868AD1",
        "#ece133",
        "#fbafe4"
    ]


def get_colormap() -> dict[int, ListedColormap]:
    rgb_colors = get_rgb_colors()
    colors = ListedColormap(rgb_colors)
    color_map = {i: colors(i) for i in range(len(rgb_colors))}
    return color_map


def get_i_rgb_colors(i: int) -> list[str]:
    rgb_colors = get_rgb_colors()
    if i > len(rgb_colors):
        i = len(rgb_colors)
    return rgb_colors[:i]


def get_default_colors():
    return plt.rcParams['axes.prop_cycle'].by_key()['color']


def get_baseline(archive: dict, queries: list[str]) -> np.ndarray:
    return np.array([archive[query_name]["63"] for query_name in queries])


def get_opt(archive: dict, queries: list[str]) -> np.ndarray:
    return np.array([archive[query_name][str(archive[query_name]["opt"])] for query_name in queries])


def get_opt_hint_sets(archive: dict, queries: list[str]) -> np.ndarray:
    return np.array([archive[query_name]["opt"] for query_name in queries])


def get_speedup(baseline: np.ndarray, comparison: np.ndarray) -> np.ndarray:
    return np.divide(np.array(baseline), np.array(comparison))


def get_min_max_over_runs(run_results: np.ndarray):
    return np.min(run_results, axis=1), np.max(run_results, axis=1)


def get_min_max_in_run(run_results: np.ndarray):
    return np.min(run_results, axis=0), np.max(run_results, axis=0)


def get_hint_set_combinations(flexible_hints: list[int], number_of_hints: int = 6) -> list[int]:
    """
    :param flexible_hints: hints which are marked to be traversed (able to be switched on or off)
    :param number_of_hints: search space limitation
    :return: hint sets in integer form, which can be traversed in the current search space and current flexible hints
    """
    # for easily mapping to integer hints by condition
    temp = {1: True, 0: False}
    # repeat decides how long the final combinations are
    bin_comb = list(itertools.product([0, 1], repeat=len(flexible_hints)))
    # same but with boolean values now
    bool_comb = [[temp[_[i]] for i in range(len(_))] for _ in bin_comb]
    combinations = list()
    cap = (2 ** number_of_hints) - 1
    # now reduce the combinations to integer values that can be used for hinting
    for comb in bool_comb:
        combinations.append(cap - sum(np.array(flexible_hints)[np.array(comb)]))
    return combinations


def get_combinations_of_order(order: list[int]) -> list[list[int]]:
    """
    :param order: The hint set order that should be used to calculate the individual hint optimization performance
    :return: list of combinations that are possible upon enabling each hint of the given order sequentially
    """
    current_order = list()
    all_combinations = list()
    for hint in order:
        current_order.append(hint)
        combinations = get_hint_set_combinations(current_order)
        all_combinations.append(combinations)
    return all_combinations


def get_pseudo_labeled_dict(archive: dict) -> dict:
    """
    This function builds a pseudo-complete labeling dict. As full labeling is infeasible especially for large amounts
    of hints, we can use the aggressive timeout strategy that was used to determine an optimistic solution. This pseudo
    labeled dict does not represent the real world scenario completely as we assume the best possible scenario of the
    interrupted query being just a little slower than the set timeout. We can use this pseudo-complete dictionary to
    get a query performance estimation without having to relabel each time there would be a missing value. However,
    this dictionary should not be used to measure performance as it does not reflect proper times for missing
    query-hint-set combinations.
    :param archive: true but incomplete labeled data
    :return: pseudo-complete dictionary
    """
    missing_values = dict()
    for query_name in archive.keys():
        # this hint is always labeled as it is the postgres default
        current_best_time = archive[query_name]["63"]
        for int_idx in reversed(range(63)):
            try:
                query_time = archive[query_name][str(int_idx)]
                if query_time < current_best_time:
                    current_best_time = query_time
            except KeyError:
                # this means the query got timed out during labeling
                # it minimally took the previous best time for it to run
                # TODO: add a small offset as overhead to avoid confusion in later evaluation if needed
                query_time = current_best_time
                try:
                    missing_values[query_name]
                except KeyError:
                    missing_values[query_name] = dict()
                missing_values[query_name][str(int_idx)] = query_time
    # lastly merge with missing values
    pseudo_dict = dict()
    for query_name in missing_values:
        pseudo_dict[query_name] = archive[query_name] | missing_values[query_name]
        print(pseudo_dict[query_name])
    if not pseudo_dict:
        return archive
    return pseudo_dict


def get_optimal_solution_of_combination(combination: list[int], archive: dict, query_names: list[str]) -> dict:
    """
    :param combination: search space of hint sets to traverse
    :param archive: archive with labeled data
    :param query_names: queries to find the optimal hint set for
    :return: dictionary containing queries, their optimal hint sets, and their respective time
    """
    sorted_comb = list(sorted(combination))
    best_dict = dict()
    for query_name in query_names:
        optimal_hint, optimal_time = 63, archive[query_name]["63"]
        for hint_set in reversed(sorted_comb):

            # check archive for information, no info i.e., KeyError indicates this combination received a timeout
            try:
                time = archive[query_name][str(hint_set)]
            except KeyError:
                continue

            # check if better solution has been found
            if time < optimal_time:
                optimal_hint = hint_set
                optimal_time = time

        # set info about the best hint within the given combination range
        best_dict[query_name] = dict()
        best_dict[query_name]["opt"] = optimal_hint
        best_dict[query_name][str(optimal_hint)] = optimal_time
    return best_dict


def get_labeling_time_of_combination(combination: list[int], archive: dict, query_name: str) -> float:
    """
    :param combination: search space of hint sets to traverse
    :param archive: archive with labeled data
    :param query_name: query to measure labeling time of
    :return: time it took to label a certain query
    """
    labeling_time = archive[query_name]["63"]
    timeout_time = labeling_time
    for j in reversed(list(sorted(combination))):
        try:
            t = archive[query_name][str(j)]
        except KeyError:
            labeling_time += timeout_time
            continue
        if t < timeout_time:
            timeout_time = t
        labeling_time += timeout_time
    return labeling_time


class Tool(abc.ABC):

    def __init__(self, archive_path: str, workload_path: str):
        self._archive_path = archive_path
        self._archive = None
        self._workload_path = workload_path
        self._workload = None
        self._queries = None

    @property
    def archive(self):
        if not os.path.exists(self._archive_path):
            raise ValueError("Archive path: {} does not exist.")
        if self._archive is None:
            self._archive = load_json(self._archive_path)
        return self._archive

    @property
    def workload(self):
        if not os.path.exists(self._workload_path):
            raise ValueError("Query path: {} does not exist.")
        if self._workload is None:
            self._workload = Workload(self._workload_path, f'wl_{__class__.__name__}')
        return self._workload

    @property
    def queries(self):
        if self._queries is None:
            self._queries = self.workload.queries
        return self._queries

    def print_results(self):
        raise NotImplementedError

    def plot(self):
        raise NotImplementedError

    def save_results(self, path: str):
        raise NotImplementedError
