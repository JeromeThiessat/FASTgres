import matplotlib
import numpy as np
import pandas as pd
import networkx as nx
from matplotlib import pyplot as plt

from fastgres.analysis_utility.tools.hint_set_order import HintSetOrder as HsO
from fastgres.analysis_utility import tool
from fastgres.baseline.hint_sets import Hint
from fastgres.baseline.utility import int_to_binary, get_first_mismatch


class QueryOrderTraversal(HsO):

    def __init__(self, archive_path: str, query_path: str, query_names: list[str], hints: list[Hint],
                 use_pseudo_dict: bool = True, op_mode: HsO.OperationMode = HsO.OperationMode.SUB):
        super().__init__(archive_path, query_path, query_names, hints, op_mode=op_mode, use_pseudo_dict=use_pseudo_dict)
        self._query_results = None

    #     self._reduced = True
    #
    # @property
    # def reduced(self):
    #     return self._reduced
    #
    # @reduced.setter
    # def reduced(self, new_val: bool):
    #     self._reduced = new_val
    #     self._query_results = None

    @property
    def query_results(self):
        if self._query_results is not None:
            return self.query_results
        self._query_results = self.calculate_reduced_result()
        return self._query_results

    def calculate_reduced_result(self):
        queries_to_use = self.eval_config.queries_to_use
        # top_k = self.eval_config.top_k
        top_k_res = self.get_top_k_results()
        orders = top_k_res["orders"]
        # including 63 and 0
        order_dict = {i: set() for i in range(len(self.hints) + 1)}
        edges = set()
        for order in orders:
            edges = edges.union(self.edges_from_order(order))
            for i in range(len(self.hints) + 1):
                order_dict[i].add(order[i])
        # print("Order Dict:", order_dict)
        # print("Edges: ", edges)
        result_dict = dict()
        for level in order_dict:
            result_dict[level] = dict()
            for hint_set in order_dict[level]:
                run_time = sum([self.used_dict[query_name][str(hint_set)] for query_name in queries_to_use])
                default_time = sum([self.used_dict[query_name]["63"] for query_name in queries_to_use])
                result_dict[level][str(hint_set)] = round(default_time / run_time, 1)
        result_dict["edges"] = edges
        return result_dict

    def print_results(self):
        print("")

    def get_max_order(self):
        res = self.calculate_reduced_result()
        max_order = list()
        edges = res["edges"]
        for level_idx in range(len(self.hints) + 1):
            level_hint_sets = res[level_idx]
            hint_sets, speedups = list(level_hint_sets.keys()), list(level_hint_sets.values())
            sorted_sets = sorted(zip(hint_sets, speedups), key=lambda x: x[1], reverse=True)
            s_h_sets = [_[0] for _ in sorted_sets] if isinstance(sorted_sets, list) else sorted_sets[0]
            for max_hint in s_h_sets:
                if not max_order or (max_order[-1], int(max_hint)) in edges:
                    max_order.append(int(max_hint))
                    break
        return max_order

    def get_evaluated_hint_sets(self, order: tuple):
        hint_order = tool.get_order_from_hint_sets(list(order))
        return self.calculate_one_ring_of_order([Hint(i) for i in hint_order])

    def get_best_of_order(self, query_name: str, order: tuple, early_stopping: bool = False):
        if early_stopping:
            current_best, current_best_time = order[0], self.used_dict[query_name][str(order[0])]
            for i in range(1, len(order)):
                new_set, new_t = order[i], self.used_dict[query_name][str(order[i])]
                if new_t < current_best_time:
                    current_best = new_set
                    current_best_time = new_t
                else:
                    # break as we stop once we have no time decrease
                    break
            return current_best, current_best_time
        else:
            sorted_list = sorted([(order[i], self.used_dict[query_name][str(order[i])]) for i in range(len(order))],
                                 key=lambda x: x[1])
            sorted_sets = [_[0] for _ in sorted_list]
            sorted_time = [_[1] for _ in sorted_list]
            return sorted_sets[0], sorted_time[0]

    def draw_default(self, order: list[int], ax, break_level: int = None):
        edges = set()
        default = 2**len(self.hints)-1
        pos = {default: np.array([round((len(self.hints)-1)/2, 1), 0]),
               str(default): np.array([round((len(self.hints)-1)/2, 1), 0])}
        graph = nx.DiGraph()
        labels = {default: f"{str(default)}\n(1.0)", f"{default}": f"{str(default)}\n(1.0)"}
        order = [default] + order + [0]
        for level_idx in range(len(order)):
            last_order_item = order[level_idx]
            int_list_last_order = int_to_binary(last_order_item, len(self.hints))
            one_ring = self.calculate_one_ring(last_order_item)
            for i in range(len(one_ring)):
                hint_set = one_ring[i]

                hint_set_int_list = int_to_binary(hint_set, len(self.hints))
                index = get_first_mismatch(int_list_last_order, hint_set_int_list)

                speedup = np.round(
                    tool.get_speedup(
                        tool.get_baseline(self.used_dict, self.query_names),
                        np.array([self.used_dict[q_name][str(hint_set)] for q_name in self.query_names])
                    ), 1)[0]
                # mean = round(len(one_ring) / 2, 1)
                # pos[hint_set] = np.array([i - mean, -level_idx])
                # pos[str(hint_set)] = np.array([i - mean, -level_idx])
                pos[hint_set] = np.array([index, -level_idx])
                pos[str(hint_set)] = np.array([index, -level_idx])
                labels[hint_set] = f"{hint_set}\n({speedup})"
                labels[str(hint_set)] = f"{hint_set}\n({speedup})"
                edges.add((last_order_item, hint_set))
                # graph.add_node(hint_set, pos=(i - mean, -level_idx))
                graph.add_node(hint_set, pos=(index, -level_idx))

            if break_level is not None and level_idx >= break_level:
                break
        # print(edges)
        # print(pos)
        top_order_edges = self.edges_from_order(tuple(order))
        self._draw_variables = dict()
        self._draw_variables["edges"] = edges
        self._draw_variables["top_order_edges"] = top_order_edges
        self._draw_variables["pos"] = pos
        self._draw_variables["labels"] = labels
        self._draw_variables["graph"] = graph
        return self.draw_graph(ax)

    def draw_reduced(self, ax):
        res = self.calculate_reduced_result()
        max_order = self.get_max_order()
        print(max_order)
        # fig, ax = plt.subplots()
        graph = nx.DiGraph()
        pos = dict()
        labels = dict()
        edges = res["edges"]
        for level_idx in range(len(self.hints) + 1):
            level_hint_sets = res[level_idx]
            hint_sets, speedups = list(level_hint_sets.keys()), list(level_hint_sets.values())
            for i in range(len(hint_sets)):
                hint_set = hint_sets[i]
                speedup = speedups[i]
                mean = round(len(hint_sets) / 2, 1)
                graph.add_node(hint_set, pos=(i - mean, -level_idx))
                # needs both
                labels[hint_set] = f"{hint_set}\n({speedup})"
                labels[str(hint_set)] = f"{hint_set}\n({speedup})"
                pos[hint_set] = np.array([i - mean, -level_idx])
                pos[str(hint_set)] = np.array([i - mean, -level_idx])

        top_order_edges = self.edges_from_order(tuple(max_order))

        self._draw_variables = dict()
        self._draw_variables["edges"] = edges
        self._draw_variables["top_order_edges"] = top_order_edges
        self._draw_variables["pos"] = pos
        self._draw_variables["labels"] = labels
        self._draw_variables["graph"] = graph
        return self.draw_graph(ax)

    @property
    def plot(self):
        if self._plot is not None:
            return self._plot
        fig, ax = plt.subplots()
        ax = self.draw_reduced(ax)
        self._plot = fig, ax
        return self._plot

    def show(self):
        fig, _ = self.plot
        plt.show()
        self._plot = None

    def save_results(self, path: str, dpi=600):
        fig, _ = self.plot
        matplotlib.rcParams['pdf.use14corefonts'] = True
        plt.savefig(path, dpi=dpi, bbox_inches="tight")
