
import numpy as np
import pandas as pd
import networkx as nx
from matplotlib import pyplot as plt

from fastgres.analysis_utility.tools.hint_set_order import HintSetOrder as HsO
from fastgres.analysis_utility import tool
from fastgres.baseline.hint_sets import Hint


class QueryOrderTraversal(HsO):

    def __init__(self, archive_path: str, query_path: str, query_names: list[str], hints: list[Hint],
                 use_pseudo_dict: bool = True, op_mode: HsO.OperationMode = HsO.OperationMode.SUB):
        super().__init__(archive_path, query_path, query_names, hints, op_mode=op_mode, use_pseudo_dict=use_pseudo_dict)
        self._query_results = None

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
        order_dict = {i: set() for i in range(len(self.hints)+1)}
        edges = set()
        for order in orders:
            edges = edges.union(self.edges_from_order(order))
            for i in range(len(self.hints)+1):
                order_dict[i].add(order[i])
        # print("Order Dict:", order_dict)
        # print("Edges: ", edges)
        result_dict = dict()
        for level in order_dict:
            result_dict[level] = dict()
            for hint_set in order_dict[level]:
                run_time = sum([self._used_dict[query_name][str(hint_set)] for query_name in queries_to_use])
                default_time = sum([self._used_dict[query_name]["63"] for query_name in queries_to_use])
                result_dict[level][str(hint_set)] = round(default_time/run_time, 1)
        result_dict["edges"] = edges
        return result_dict

    def print_results(self):
        print("")

    def get_max_order(self):
        res = self.calculate_reduced_result()
        max_order = list()
        edges = res["edges"]
        for level_idx in range(len(self.hints)+1):
            level_hint_sets = res[level_idx]
            hint_sets, speedups = list(level_hint_sets.keys()), list(level_hint_sets.values())
            sorted_sets = sorted(zip(hint_sets, speedups), key=lambda x: x[1], reverse=True)
            s_h_sets = [_[0] for _ in sorted_sets] if isinstance(sorted_sets, list) else sorted_sets[0]
            for max_hint in s_h_sets:
                if not max_order or (max_order[-1], int(max_hint)) in edges:
                    max_order.append(int(max_hint))
                    break
        return max_order

    def get_evaluated_hint_sets(self, query_name: str, order: tuple, early_stopping: bool = False):
        self.calculate_one_ring_of_order()

    def get_best_of_order(self, query_name: str, order: tuple, early_stopping: bool = False):
        if early_stopping:
            current_best, current_best_time = order[0], self._used_dict[query_name][str(order[0])]
            for i in range(1, len(order)):
                new_set, new_t = order[i], self._used_dict[query_name][str(order[i])]
                if new_t < current_best_time:
                    current_best = new_set
                else:
                    break
            return current_best, current_best_time
        else:
            sorted_list = sorted([(order[i], self._used_dict[query_name][str(order[i])]) for i in range(len(order))],
                                 key=lambda x: x[1])
            sorted_sets = [_[0] for _ in sorted_list]
            sorted_time = [_[1] for _ in sorted_list]
            return sorted_sets[0], sorted_time[0]

    @property
    def plot(self):
        if self._plot is not None:
            return self._plot

        res = self.calculate_reduced_result()
        fig, ax = plt.subplots()
        graph = nx.DiGraph()
        pos = dict()
        labels = dict()
        edges = res["edges"]
        for level_idx in range(len(self.hints)+1):
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

        max_order = self.get_max_order()
        top_order_edges = self.edges_from_order(tuple(max_order))
        for edge in edges:
            if edge in top_order_edges:
                color = "#009900"
                pen_width = 1.6
            else:
                color = "#000000"
                pen_width = 0.8
            graph.add_edge(str(edge[0]), str(edge[1]), color=color, weight=pen_width)

        colors = [graph[u][v]['color'] for u, v in graph.edges()]
        color_map = ['w' for _ in graph]
        # labels = [str(node) for node in graph]
        weights = [float(graph[u][v]['weight']) for u, v in graph.edges()]
        nx.draw_networkx_nodes(graph, pos=pos, node_color=color_map, label=labels, alpha=0.1,
                               linewidths=0.1, ax=ax)
        nx.draw_networkx_labels(graph, pos, ax=ax, labels=labels)
        index = 0
        for edge in graph.edges(data=True):
            nx.draw_networkx_edges(graph, pos=pos, edgelist=[(edge[0], edge[1])], edge_color=colors[index],
                                   arrowsize=weights[index] * 6, width=weights[index], ax=ax, node_size=1000)
            index += 1
        self._plot = fig, ax
        return self._plot

    def show(self):
        fig, _ = self.plot
        plt.show()
        self._plot = None

    def save_results(self, path: str):
        fig, _ = self.plot
        plt.savefig(path)
