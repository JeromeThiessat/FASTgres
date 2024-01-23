import matplotlib
import matplotlib.pyplot as plt
import numpy as np

from fastgres.analysis_utility import tool
from fastgres.analysis_utility.tool import Tool
from fastgres.analysis_utility.tool import TimeGranularity as Tg


class SingleHintOptimizationInfluence(Tool):

    def __init__(self, archive_path: str, query_path: str, query_names: list[str], hint_set_order: list[int],
                 time_granularity: Tg = Tg.SECONDS, use_pseudo_dict: bool = True):
        super().__init__(archive_path, query_path)
        # queries to evaluate not workload.queries (all queries)
        self._query_names = query_names
        self._hint_set_order = hint_set_order
        self._results = None
        self._time_factor = None
        self._time_granularity = None
        self.time_granularity = time_granularity
        self.properties = None
        self._plot = None
        if use_pseudo_dict:
            self._used_dict = tool.get_pseudo_labeled_dict(self.archive)
        else:
            self._used_dict = self.archive

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
        # invalidate current plot
        self._plot = None

    @property
    def query_names(self):
        return self._query_names

    @query_names.setter
    def query_names(self, new_queries: list[str]):
        self._query_names = new_queries
        if self._results is not None:
            self._results = self._calculate_result()
        # invalidate current plot
        self._plot = None

    @property
    def hint_set_order(self):
        return self._hint_set_order

    @hint_set_order.setter
    def hint_set_order(self, new_order: list[int]):
        self._hint_set_order = new_order
        if self._results is not None:
            self._results = self._calculate_result()
        # invalidate current plot
        self._plot = None

    @property
    def results(self):
        if self._results is None:
            self._results = self._calculate_result()
        return self._results

    def _calculate_result(self) -> dict[str, np.ndarray]:
        """
        :return: dictionary with labeling times, optimal times within a combination, and their respective default time
        """
        combinations = tool.get_combinations_of_order(self.hint_set_order)
        labeling_times = list()
        optimal_times = list()
        default_times = list()
        for combination in combinations:
            result_dict = tool.get_optimal_solution_of_combination(combination, self._used_dict, self.query_names)
            # time it would have taken to label this combination
            labeling_time = np.divide(
                np.sum([tool.get_labeling_time_of_combination(combination, self._used_dict, query_name)
                        for query_name in self.query_names]),
                self._time_factor)
            # use the result dict as reference as under restricted hint flexibility, we obtain new optimal hint sets
            optimal_time = np.divide(
                np.sum(tool.get_opt(result_dict, self.query_names)),
                self._time_factor)
            # default time stays the same
            default_time = np.divide(
                np.sum(tool.get_baseline(self.archive, self.query_names)),
                self._time_factor)
            labeling_times.append(labeling_time)
            optimal_times.append(optimal_time)
            default_times.append(default_time)
        return {
            "labeling_times": np.array(labeling_times),
            "optimal_times": np.array(optimal_times),
            "default_times": np.array(default_times)
        }

    # def _calculate_results(self) -> list[dict[str, list[float]]]:
    #     results = list()
    #     for hint_set_order in self.hint_set_orders:
    #         results.append(self._calculate_result(hint_set_order))
    #     return results

    def set_properties(self, **kwargs):
        self.properties = kwargs

    def _apply_properties(self):
        if self.properties is None:
            return

        title_left = self.properties.get("title_left", None)
        title_right = self.properties.get("title_right", None)
        y_label_left = self.properties.get("ylabel_left", None)
        y_label_right = self.properties.get("ylabel_right", None)
        y_ticks_left = self.properties.get("yticks_left", None)
        y_ticks_right = self.properties.get("yticks_right", None)
        y_lim_left = self.properties.get("ylim_left", None)
        y_lim_right = self.properties.get("ylim_right", None)
        _, axs = self.plot
        speedup_ax, labeling_ax = axs
        if title_left is not None: speedup_ax.set_title(title_left)
        if title_right is not None: labeling_ax.set_title(title_right)
        if y_label_left is not None: speedup_ax.set_ylabel(y_label_left)
        if y_label_right is not None: labeling_ax.set_ylabel(y_label_right + f" [{self.time_granularity.value}]")
        if y_ticks_left is not None: speedup_ax.set_yticks(y_ticks_left)
        if y_ticks_right is not None: labeling_ax.set_yticks(y_ticks_right)
        if y_lim_left is not None: speedup_ax.set_ylim(y_lim_left)
        if y_lim_right is not None: labeling_ax.set_ylim(y_lim_right)

    @staticmethod
    def _get_names(hint_set_order: list[int]):
        name_dict = {32: 'hash',
                     16: 'merge',
                     8: 'nl',
                     4: 'idx-s',
                     2: 'seq-s',
                     1: 'idxo-s'}
        # names = [name_dict[idx] for idx in combination]
        return ["+" + name_dict[hint_set_order[idx]] if idx > 0
                else name_dict[hint_set_order[idx]] for idx in range(len(hint_set_order))]

    def set_bar_text(self, bar, ax=None):
        if ax is None:
            _, ax = self.plot
        for rect in bar:
            height = rect.get_height()
            ax.text(rect.get_x() + rect.get_width() / 2.0,
                    height / 2,
                    '%.2f' % round(height, 2),
                    ha='center',
                    va='bottom')
            # ax.text(i, y=display_time / 2, s=int(round(display_time, 1)), ha="center", va="center", c="white")
        return

    def print_results(self):
        print("")

    @property
    def plot(self):
        if self._plot is not None:
            return self._plot

        # possible speedup
        y_values = tool.get_speedup(self.results["default_times"], self.results["optimal_times"])
        x_values = [i for i in range(len(y_values))]

        # speedup_differences = np.ediff1d(y_values, to_begin=y_values[0])
        labeling_times = self.results["labeling_times"]
        # labeling_time_differences = np.ediff1d(labeling_times, to_begin=labeling_times[0])

        self._plot = plt.subplots(figsize=(16, 6), ncols=2)

        _, axs = self._plot
        ax_speedup, ax_labeling_time = axs
        bar_speedup = ax_speedup.bar(x_values, y_values)
        bar_labeling = ax_labeling_time.bar(x_values, labeling_times)

        tick_names = self._get_names(self.hint_set_order)
        ax_speedup.set_xticks(x_values, tick_names, rotation=-45)
        ax_labeling_time.set_xticks(x_values, tick_names, rotation=-45)

        # self.set_bar_text(bar_speedup, ax_speedup)
        # self.set_bar_text(bar_labeling, ax_speedup)
        tool.set_fonts(22)

        return self._plot

    def show(self):
        fig, _ = self.plot
        self._apply_properties()
        plt.subplots_adjust(wspace=0.23)
        plt.show()
        self._plot = None

    def save_results(self, path: str, dpi: int = 600):
        matplotlib.rcParams['pdf.use14corefonts'] = True
        fig, _ = self.plot
        self._apply_properties()
        fig.savefig(path, dpi=dpi, bbox_inches="tight")
