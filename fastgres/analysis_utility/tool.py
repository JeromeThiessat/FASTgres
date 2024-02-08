import abc
import enum
import itertools
import math
import os
import time

import matplotlib.pyplot as plt
import numpy as np
import warnings
from matplotlib.colors import ListedColormap
from copy import deepcopy

from fastgres.baseline.utility import load_json, binary_to_int
from fastgres.workloads.workload import Workload


class TimeGranularity(enum.Enum):
    SECONDS = "s"
    MINUTES = "m"
    HOURS = "h"


class FontParameter(enum.Enum):
    FONT = "font.size"
    AXES_TITLE = "axes.titlesize"
    AXES_LABEL = "axes.labelsize"
    X_TICK_LABEL = "xtick.labelsize"
    Y_TICK_LABEL = "ytick.labelsize"
    LEGEND_FONT = "legend.fontsize"
    FIGURE_TITLE = "figure.titlesize"


def default_fonts(size):
    return {i: size for i in FontParameter}


def font(font_params: dict[FontParameter, int]):
    def decorator(f):
        def inner(*args, **kwargs):
            with plt.rc_context({fp.value: size for fp, size in font_params.items()}):
                result = f(*args, **kwargs)
            return result

        return inner

    return decorator

    # # https://stackoverflow.com/a/14971193
    # plt.rc('font', size=size)
    # plt.rc('axes', titlesize=size)
    # plt.rc('axes', labelsize=size)
    # plt.rc('xtick', labelsize=size)
    # plt.rc('ytick', labelsize=size)
    # plt.rc('legend', fontsize=size)
    # plt.rc('figure', titlesize=size)


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


def get_colormap() -> dict[int, tuple[float, float, float, float]]:
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
    hints = list(set(list(archive[list(archive.keys())[0]].keys())) - {"opt"})
    top_hint = max([int(_) for _ in hints])
    hint_count = int(math.log2(top_hint+1))
    return np.array([archive[query_name][str(2**hint_count-1)] for query_name in queries])


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


def get_reduced_archive(archive: dict, flexible_hints: list[int], number_of_hints: int = 6) -> dict:
    """
    This function reduces a given archive to match a fixed amounts of used hints. This should only be used if
    :param archive: archive to reduce from
    :param flexible_hints: hints used in reduction, i.e., the flexible ones
    :param number_of_hints: overall amount of hints, i.e., flexible and rigid hints
    :return: archive containing only flexible hint set combinations
    """
    reduced = {key: dict() for key in archive}
    combinations = get_hint_set_combinations(flexible_hints, number_of_hints)
    # for query_name in reduced:
    #     reduced[query_name] = {hint_set: archive[query_name][str(hint_set)] for hint_set in combinations}
    #     reduced["opt"] = archive[query_name]["opt"] if int(archive[query_name]["opt"]) in combinations else
    for query_name in archive:
        opt_set, opt_t = 63, archive[query_name]['63']
        old_opt = int(archive[query_name]['opt'])
        for hint_set in combinations:
            try:
                hint_set_time = archive[query_name][str(hint_set)]
                reduced[query_name][str(hint_set)] = hint_set_time
                if hint_set_time < opt_t:
                    opt_set = hint_set
                    opt_t = hint_set_time
            except KeyError:
                continue
        reduced[query_name]['opt'] = opt_set if old_opt not in combinations else str(old_opt)

    return reduced


def get_hint_set_reduced_archive(archive: dict, hint_sets_to_use: list[int]) -> dict:
    """
    This function reduces a given archive to match a fixed amounts of used hints. This should only be used if
    :param archive: archive to reduce from
    :param hint_sets_to_use: hint sets to reduce to
    :return: archive containing only flexible hint set combinations
    """
    reduced = {key: dict() for key in archive}
    for query_name in archive:
        opt_set, opt_t = 63, archive[query_name]['63']
        old_opt = int(archive[query_name]['opt'])
        for hint_set in hint_sets_to_use:
            try:
                hint_set_time = archive[query_name][str(hint_set)]
                reduced[query_name][str(hint_set)] = hint_set_time
                if hint_set_time < opt_t:
                    opt_set = hint_set
                    opt_t = hint_set_time
            except KeyError:
                continue
        reduced[query_name]['opt'] = opt_set if old_opt not in hint_sets_to_use else str(old_opt)
    return reduced


def sanity_check_archive(archive: dict, threshold: float = 0.001) -> [dict, bool]:
    """
    Performs sanity checks on labeled archives such that no floating point errors in time measurement remain,
    causing issues when calculating with runtimes of e.g., 0.0
    :param archive: input archive to check
    :param threshold: threshold to consider while checking
    :return: cleaned archive and boolean if cleaning was performed
    """
    cleaned_dict = dict()
    cleaned = False
    for query_name in archive:
        new_dict = dict()
        for key in archive[query_name]:
            # except opt from checks
            if key == 'opt':
                continue
            time = float(archive[query_name][key])
            if time <= threshold:
                cleaned = True
                new_dict[key] = threshold
        # overwrite erroneous entries
        cleaned_dict[query_name] = archive[query_name] | new_dict
    return cleaned_dict, cleaned


def get_pseudo_labeled_dict(archive: dict, interpolate_default: bool = False) -> dict:
    """
    This function builds a pseudo-complete labeling dict. As full labeling is infeasible especially for large amounts
    of hints, we can use the aggressive timeout strategy that was used to determine an optimistic solution. This pseudo
    labeled dict does not represent the real world scenario completely as we assume the best possible scenario of the
    interrupted query being just a little slower than the set timeout. We can use this pseudo-complete dictionary to
    get a query performance estimation without having to relabel each time there would be a missing value. However,
    this dictionary should not be used to measure performance as it does not reflect proper times for missing
    query-hint-set combinations. Currently only supports 6 hints.
    :param interpolate_default: Switch to interpolate using pg default values
    :param archive: true but incomplete labeled data
    :return: pseudo-complete dictionary
    """
    pseudo_dict = deepcopy(archive)
    faulty_entries = 0
    hints = list(set(list(archive[list(archive.keys())[0]].keys())) - {"opt"})
    top_hint = max([int(_) for _ in hints])
    hint_count = int(math.log2(top_hint+1))
    # print(f"using {hint_count} hints")
    top_hint_set_int = binary_to_int([1]*hint_count)
    for query_name in archive:
        pg_def = archive[query_name][str(top_hint_set_int)]
        current_best_time = archive[query_name][str(top_hint_set_int)]
        for int_idx in reversed(range(top_hint_set_int)):
            try:
                query_time = archive[query_name][str(int_idx)]
                if query_time < current_best_time:
                    current_best_time = query_time
            except (KeyError, ValueError):
                query_time = current_best_time if not interpolate_default else pg_def
                query_time = 0.001 if query_time <= 0.001 else query_time
                pseudo_dict[query_name][str(int_idx)] = query_time
            if query_time < pseudo_dict[query_name][str(pseudo_dict[query_name]["opt"])]:
                faulty_entries += 1
    if faulty_entries > 0:
        warnings.warn(f"Archive contained: {faulty_entries} faulty entries")
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


def get_order_from_hint_sets(hint_sets: list[int], sub: bool = True) -> list[int]:
    order = list()
    for i in range(1, len(hint_sets)):
        if sub:
            order.append(hint_sets[i - 1] - hint_sets[i])
        else:
            order.append(hint_sets[i] - hint_sets[i - 1])
    return order


class Tool(abc.ABC):

    def __init__(self, archive_path: str, workload_path: str, sanity_check: bool = True):
        self._archive_path = archive_path
        self._archive = None
        self._workload_path = workload_path
        self._workload = None
        self._workload_queries = None
        self._sanity_check = sanity_check

    @property
    def archive(self):
        if not os.path.exists(self._archive_path):
            raise ValueError("Archive path: {} does not exist.")
        if self._archive is None:
            self._archive = load_json(self._archive_path)
            if self._sanity_check:
                self._archive, cleaned = sanity_check_archive(self._archive)
                if cleaned:
                    warnings.warn("Input archive contained erroneous entries and was cleaned")
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
        if self._workload_queries is None:
            self._workload_queries = self.workload.queries
        return self._workload_queries

    # def print_results(self):
    #     raise NotImplementedError

    def plot(self):
        raise NotImplementedError

    def save_results(self, path: str):
        raise NotImplementedError
