
import enum
import warnings
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from fastgres.analysis_utility import tool
from fastgres.analysis_utility.tool import TimeGranularity as Tg
from fastgres.analysis_utility.tool import Tool
from fastgres.baseline.hint_sets import Hint
from fastgres.baseline.utility import int_to_binary, chunks


class HintSetDistribution(Tool):

    def __init__(self, archive_path: str, query_path: str, query_names: list[str], hints: list[Hint],
                 use_pseudo_dict: bool = False):
        super().__init__(archive_path, query_path)

        self._query_names = query_names
        self._hints = hints
        self._plot = None
        self.properties = None
        if use_pseudo_dict:
            self._used_dict = tool.get_pseudo_labeled_dict(self.archive)
        else:
            self._used_dict = self.archive

    @property
    def query_names(self):
        return self._query_names

    @query_names.setter
    def query_names(self, new_names: list[str]):
        self._query_names = new_names

    @property
    def hints(self):
        return self._hints

    @hints.setter
    def hints(self, new_hints: list[Hint]):
        self._used_dict = tool.get_reduced_archive(self._used_dict,
                                                   [hint.value for hint in self.hints],
                                                   len(self.hints))
        self._hints = new_hints
        self._plot = None

    def get_opt_histogram(self):
        sets = 2**len(self.hints)
        opt_sets = tool.get_opt_hint_sets(self._used_dict, self.query_names)
        df = pd.DataFrame({"query_name": self.query_names, "opt": opt_sets})
        aggregated_df = df.groupby(by=["opt"]).count()
        empty_df = pd.DataFrame(index=[i for i in range(sets)])
        aggregated_df = aggregated_df.join(empty_df, how="right")
        aggregated_df.fillna(0, inplace=True)
        aggregates = [_[0] for _ in aggregated_df.to_numpy().astype(int)]
        return list(aggregated_df.index), aggregates

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
        if y_label is not None: ax.set_ylabel(y_label)
        if x_ticks is not None: ax.set_xticks(x_ticks)
        if y_ticks is not None: ax.set_yticks(y_ticks)
        if x_lim is not None: ax.set_xlim(x_lim)
        if y_lim is not None: ax.set_ylim(y_lim)
        return

    @property
    def plot(self):
        if self._plot is not None:
            return self._plot
        histogram = self.get_opt_histogram()
        self._plot = plt.subplots(figsize=(6, 3))
        fig, ax = self._plot
        ax.yaxis.grid(color='gray', linestyle='dotted', linewidth=0.5)
        ax.bar(histogram[0], histogram[1], edgecolor="black", linewidth=0.05, color="#0173b2", zorder=3)
        return self._plot

    @tool.font(tool.default_fonts(14))
    def show(self):
        fig, _ = self.plot
        self._apply_properties()
        plt.show()
        self._plot = None

#
# # Do Once
# # hint_dist_combined_axs[0, 0].legend(handles=patches, loc="lower center", bbox_to_anchor=(1.1, -1.6), ncol=2)
# hint_dist_combined_axs[0, 1].set_title("JOB")
# hint_dist_combined_axs[0, 0].set_ylabel("PG12\nOccurrence [Count]")
# hint_dist_combined_axs[0, 0].set_title("Stack")
# hint_dist_combined_axs[1, 0].set_ylabel("PG14\nOccurrence [Count]")
# hint_dist_combined_axs[1, 0].set_xlabel("Hint Set [Int]")
# hint_dist_combined_axs[1, 1].set_xlabel("Hint Set [Int]")
#
# matplotlib.rcParams['pdf.use14corefonts'] = True
# # hint_dist_combined_fig.savefig("revision/distributions/hint_set_distribution_combined.pdf", dpi=1200, bbox_inches = "tight")
# plt.show()