
import matplotlib
import matplotlib.pyplot as plt
import numpy as np

from fastgres.analysis_utility import tool
from fastgres.analysis_utility.tool import Tool
from fastgres.analysis_utility.tool import TimeGranularity as Tg


class HintOptimizationPotential(Tool):

    def __init__(self, archive_path: str, query_path: str,
                 time_granularity: Tg = Tg.SECONDS):
        super().__init__(archive_path, query_path)
        # tool.set_fonts(16)
        self._x = ["PG Default", "Opt"]
        self._y = None
        self._plot = None
        self.properties = None
        self._time_factor = None
        self._time_granularity = None
        self.time_granularity = time_granularity

    @property
    def y(self):
        if self._y is None:
            self._y = self.calculate_results()
        return self._y

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

    def calculate_results(self):
        default_time = np.divide(np.sum(tool.get_baseline(self.archive, self.queries)),
                                 self._time_factor)
        optimal_time = np.divide(np.sum(tool.get_opt(self.archive, self.queries)),
                                 self._time_factor)
        return default_time, optimal_time

    def print_results(self):
        print(f"Evaluation of {len(self.queries)} queries: Default: {self.y[0]} / Optimal: {self.y[1]}")

    def set_bar_text(self, bar):
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

    @property
    def plot(self):
        if self._plot is not None:
            return self._plot
        self._plot = plt.subplots()
        fig, ax = self._plot
        bar = ax.bar(self._x, self.y, color=tool.get_i_rgb_colors(len(self._x)))
        self.set_bar_text(bar)
        return self._plot

    def set_properties(self, **kwargs):
        self.properties = kwargs

    def _apply_properties(self):
        if self.properties is None:
            return
        title = self.properties.get("title", None)
        y_label = self.properties.get("ylabel", None)
        x_label = self.properties.get("xlabel", None)
        x_ticks = self.properties.get("xticks", None)
        y_ticks = self.properties.get("yticks", None)
        x_lim = self.properties.get("xlim", None)
        y_lim = self.properties.get("ylim", None)
        _, ax = self.plot
        if title is not None: ax.set_title(title)
        if x_label is not None: ax.set_xlabel(x_label)
        if y_label is not None: ax.set_ylabel(y_label + f" [{self.time_granularity.value}]")
        if x_ticks is not None: ax.set_xticks(x_ticks)
        if y_ticks is not None: ax.set_yticks(y_ticks)
        if x_lim is not None: ax.set_xlim(x_lim)
        if y_lim is not None: ax.set_ylim(y_lim)

    @tool.font(tool.default_fonts(16))
    def show(self):
        fig, _ = self.plot
        self._apply_properties()
        plt.show()
        self._plot = None

    @tool.font(tool.default_fonts(16))
    def save_results(self, path: str, dpi: int = 600):
        matplotlib.rcParams['pdf.use14corefonts'] = True
        fig, _ = self.plot
        self._apply_properties()
        fig.savefig(path, dpi=dpi, bbox_inches="tight")
        # plt.close(fig)
