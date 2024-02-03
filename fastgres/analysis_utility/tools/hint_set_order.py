import numpy as np
import pandas as pd
import networkx as nx
from matplotlib import pyplot as plt

from fastgres.analysis_utility.tools.ring_neighborhood_traversal import RingNeighborhoodTraversal as RnT
from fastgres.analysis_utility import tool
from fastgres.baseline.hint_sets import Hint


class HintSetOrder(RnT):
    class EvaluationConfig:

        def __init__(self, top_k: int, queries_to_use: list[str]):
            self.top_k = top_k
            self.queries_to_use = queries_to_use

    def __init__(self, archive_path: str, query_path: str, query_names: list[str], hints: list[Hint],
                 use_pseudo_dict: bool = True,  op_mode: RnT.OperationMode = RnT.OperationMode.SUB):
        super().__init__(archive_path, query_path, query_names, hints, op_mode=op_mode, use_pseudo_dict=use_pseudo_dict)
        self._result_df = None
        self._graph_results = None
        self._eval_config = None
        self._results = None

    @property
    def results(self):
        if self._results is not None:
            return self._results
        self._results = self._calculate_sorted()
        return self._results

    @property
    def eval_config(self):
        if self._eval_config is None:
            raise ValueError("Evaluation Config not set yet.")
        return self._eval_config

    @eval_config.setter
    def eval_config(self, new_config: EvaluationConfig):
        self._eval_config = new_config
        self._results = None

    def _calculate_sorted(self):
        reduced_df = self.result_df[self.result_df["query_name"].isin(self.query_names)]
        count_aggregate = reduced_df.groupby(by=[f"level_{i}" for i in range(len(self.hints) + 1)]).count()
        orders = count_aggregate.index.to_numpy()
        occ = count_aggregate["time_0"].to_numpy()

        idx = [i for i in range(len(occ))]
        sorted_index_tuples = sorted(list(zip(idx, occ)), key=lambda x: x[1], reverse=True)
        sorted_idx = [_[0] for _ in sorted_index_tuples]
        sorted_occ = [_[1] for _ in sorted_index_tuples]
        sorted_orders = [orders[i] for i in sorted_idx]
        result = {
            "sorted_idx": sorted_idx,
            "sorted_occ": sorted_occ,
            "sorted_orders": sorted_orders
        }
        return result

    def get_top_k_results(self):
        res = self.results
        sorted_orders = res["sorted_orders"]
        sorted_occ = res["sorted_occ"]
        top_orders = sorted_orders[:self.eval_config.top_k]
        top_occ = sorted_occ[:self.eval_config.top_k]
        coverage = round(100 * sum(top_occ) / sum(sorted_occ), 1)
        res = {
            "top_k": self.eval_config.top_k,
            "idx": res["sorted_idx"][:self.eval_config.top_k],
            "occ": top_occ,
            "orders": top_orders,
            "coverage": coverage
        }
        return res

    def _calculate_result_df(self):
        cols = ["query_name"]
        levels = [f"level_{i}" for i in range(len(self.hints) + 1)]
        times = [f"time_{i}" for i in range(len(self.hints) + 1)]
        cols.extend(levels)
        cols.extend(times)
        res = pd.DataFrame(columns=cols)
        all_query_names = self.query_names
        for query_name in self.query_names:
            self.query_names = [query_name]
            traversed_hint_sets = self.traverse_hint_sets()
            entry = [query_name]
            entry.extend(list(traversed_hint_sets.keys()))
            entry.extend(list(traversed_hint_sets.values()))
            res.loc[len(res.index)] = entry
        # reset to old default
        # TODO: fix ugly workaround
        self.query_names = all_query_names
        return res

    @property
    def result_df(self):
        if self._result_df is not None:
            return self._result_df
        self._result_df = self._calculate_result_df()
        return self._result_df

    def print_results(self):
        print("")

    @staticmethod
    def edges_from_order(order: tuple) -> set[tuple]:
        edge_tuples = set()
        for i in range(1, len(order)):
            edge_tuples.add((order[i - 1], order[i]))
        return edge_tuples

    @property
    def plot(self):
        if self._plot is not None:
            return self._plot

        res = self.results
        top_k_res = self.get_top_k_results()
        top_orders = top_k_res["orders"]
        orders = res["sorted_orders"]

        top_order_edges = set()
        for order_pair in top_orders:
            for i in range(1, len(order_pair)):
                top_order_edges.add((order_pair[i - 1], order_pair[i]))

        level_uniques = [[63]]
        for level in range(1, len(self.hints) + 1):
            single_level_uniques = list(set([order[level] for order in orders]))
            level_uniques.append(single_level_uniques)

        edge_tuples = set()
        for order_pair in orders:
            edge_tuples = edge_tuples.union(self.edges_from_order(order_pair))
            # for i in range(1, len(order_pair)):
            #     edge_tuples.add((order_pair[i - 1], order_pair[i]))

        fig, ax = plt.subplots()
        graph = nx.DiGraph()
        pos = dict()
        for level_idx in range(len(self.hints) + 1):
            for i in range(len(level_uniques[level_idx])):
                unique = level_uniques[level_idx][i]
                mean = round(len(level_uniques[level_idx]) / 2, 1)
                graph.add_node(unique, pos=(i - mean, -level_idx))
                # needs both
                pos[unique] = np.array([i - mean, -level_idx])
                pos[str(unique)] = np.array([i - mean, -level_idx])

        for edge in edge_tuples:
            if edge in top_order_edges:
                color = "#009900"
                pen_width = 1.6
            else:
                color = "#000000"
                pen_width = 0.8
            graph.add_edge(str(edge[0]), str(edge[1]), color=color, weight=pen_width)

        colors = [graph[u][v]['color'] for u, v in graph.edges()]
        color_map = ['w' for _ in graph]
        labels = [str(node) for node in graph]
        weights = [float(graph[u][v]['weight']) for u, v in graph.edges()]
        nx.draw_networkx_nodes(graph, pos=pos, node_color=color_map, label=labels, alpha=0.1, linewidths=0.1, ax=ax)
        nx.draw_networkx_labels(graph, pos, ax=ax)
        index = 0
        for edge in graph.edges(data=True):
            nx.draw_networkx_edges(graph, pos=pos, edgelist=[(edge[0], edge[1])], edge_color=colors[index],
                                   arrowsize=weights[index] * 6, width=weights[index], ax=ax)
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
