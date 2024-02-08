import argparse
import os
import pstats
import time

import numpy as np

from fastgres.baseline.database_connection import DatabaseConnection
from fastgres.baseline.hint_sets import HintSet
from fastgres.definitions import PathConfig
from fastgres.baseline.utility import load_json, save_json, binary_to_int, int_to_binary
from fastgres.workloads.workload import Workload
from fastgres.baseline.log_utils import Logger, get_logger
from fastgres.analysis_utility.tools.ring_neighborhood_traversal import RingNeighborhoodTraversal as RnT
from fastgres.analysis_utility import tool
from tqdm import trange

# pg_hints = [("ASYNC_APPEND", "enable_async_append"), ("BITMAP_SCAN", "enable_bitmapscan"),
#             ("GATHER_MERGE", "enable_gathermerge"), ("HASH_AGG", "enable_hashagg"),
#             ("INC_SORT", "enable_incremental_sort"), ("MATERIALIZATION", "enable_material"),
#             ("MEMOIZE", "enable_memoize"), ("PARA_APPEND", "enable_parallel_append"),
#             ("PARA_HASH", "enable_parallel_hash"), ("PART_PRUNING", "enable_partition_pruning"),
#             ("PART_JOIN", "enable_partitionwise_join"), ("PART_AGG", "enable_partitionwise_aggregate"),
#             ("PRESORT_AGG", "enable_presorted_aggregate"), ("SORT", "enable_sort"),
#             ("TID_SCAN", "enable_tidscan "), ("HASH_JOIN", "enable_hashjoin"),
#             ("MERGE_JOIN", "enable_mergejoin"), ("NESTED_LOOP_JOIN", "enable_nestloop"),
#             ("INDEX_SCAN", "enable_indexscan"), ("SEQ_SCAN", "enable_seqscan"),
#             ("INDEX_ONLY_SCAN", "enable_indexonlyscan")]
# hints = [(list(reversed(pg_hints))[i][0], 2 ** i) for i in range(len(pg_hints))]
#
# PostgresOperator = enum.Enum("PostgresOperator", pg_hints)
# Hint = enum.Enum("Hint", hints)

stop_level = 7
use_hint_removal = True


def calculate_one_ring(hint_set: HintSet, reduced_space: list[int] = None,
                       op_mode: RnT.OperationMode = RnT.OperationMode.SUB,
                       ignored_hints: list[int] = None) -> list[int]:
    ignored = [0] * len(hint_set.operators) if ignored_hints is None else ignored_hints
    binary_rep = hint_set.get_binary()
    # ignored sets are internally mapped to 0/1 depending on op mode
    binary_rep = [1 - op_mode.value if ignored[i] == 1 else binary_rep[i] for i in range(len(hint_set.operators))]
    if op_mode.value not in binary_rep:
        return []
    factor = -1 if op_mode == op_mode.SUB else 1
    result = [hint_set.get_int() + (factor * 2 ** i) for i, x in enumerate(reversed(binary_rep)) if x == op_mode.value]
    if reduced_space is not None:
        result = list(set(result).intersection(set(reduced_space)))
    return result


class Labeling:

    def __init__(self, connection_string: str, archive: dict, workload: Workload,
                 base_timeout: float = 300, use_extension: bool = False):
        self.dbc = DatabaseConnection(connection_string, "labeling_connection", use_extension)
        self.archive = archive
        self.workload = workload
        self.base_timeout = base_timeout
        self.read_queries = self.workload.read_queries()
        self.starting_hint_set = HintSet(binary_to_int([1 for _ in range(len(HintSet.all_operators))]),
                                         operators=HintSet.all_operators)
        self.op_mode = RnT.OperationMode.SUB

    def save_archive(self, path: str):
        save_json(self.archive, path)

    def get_hints_to_ignore(self, query_name: str, previous_hint_set: int, neighborhood: list[int],
                            time_threshold: float) -> list[int]:
        ignored = 0
        for hint_set_int in neighborhood:
            hint_int = previous_hint_set - hint_set_int
            hint_set_time = self.archive[query_name][str(hint_set_int)]
            # speedup = tool.get_speedup(tool.get_baseline(self.archive, [query_name]), np.array([hint_set_time]))
            if hint_set_time > time_threshold:
                ignored += hint_int
        return int_to_binary(ignored, len(self.starting_hint_set.operators))

    def get_best_hint_set(self, query_name: str, query: str):
        timeout = self.base_timeout
        current_hint_set = self.starting_hint_set
        current_time = self.dbc.evaluate_hinted_query(query, current_hint_set, timeout)

        opt_hint_set = current_hint_set
        opt_time = current_time

        self.archive[query_name][str(current_hint_set.get_int())] = current_time
        # self.archive[query_name]["order"] = [current_hint_set.get_int()]
        level = 1
        hints_to_ignore = [0] * len(self.starting_hint_set.operators)
        while current_hint_set.get_int() != 0:
            neighborhood = calculate_one_ring(current_hint_set, None, self.op_mode,
                                              ignored_hints=hints_to_ignore)
            if not neighborhood:
                break
            best_neighborhood_hint_set = None
            best_neighborhood_time = None

            logger.info(f"{query_name} neighborhood level: {level}, {len(neighborhood)}")
            for hint_set_int in neighborhood:
                hint_set = HintSet(hint_set_int, operators=HintSet.all_operators)
                query_hint_time = self.dbc.evaluate_hinted_query(query, hint_set, timeout)
                self.archive[query_name][str(hint_set.get_int())] = query_hint_time

                if best_neighborhood_time is None or query_hint_time < best_neighborhood_time:
                    best_neighborhood_hint_set = hint_set
                    best_neighborhood_time = query_hint_time
            neighborhood_ignored = self.get_hints_to_ignore(query_name, current_hint_set.get_int(), neighborhood,
                                                            1.5*best_neighborhood_time)
            hints_to_ignore = [0 if neighborhood_ignored[i] == 0 and hints_to_ignore[i] == 0
                               else 1 for i in range(len(neighborhood_ignored))]

            current_hint_set = best_neighborhood_hint_set
            current_time = best_neighborhood_time
            # self.archive[query_name]["order"].append(current_hint_set.get_int())

            if current_time < opt_time:
                opt_hint_set = current_hint_set
                opt_time = current_time

            # TODO: Delete after analysis
            if level >= stop_level:
                break
            level += 1

        self.archive[query_name]["opt"] = opt_hint_set.get_int()
        return


def run_single_query(labeling: Labeling, query_names: list[str], idx: int, save_path: str):
    query_name = query_names[idx]
    try:
        # Check if we can skip queries
        opt_val = labeling.archive[query_name]['opt']
        logger.info("Found optimum for query {}, {}".format(query_name, opt_val))
        return
    except KeyError:
        # Evaluate like default
        pass
    logger.info('Evaluating query: {}, {} / {}'.format(query_name, idx + 1, len(query_names)))
    labeling.archive[query_name] = dict()
    labeling.get_best_hint_set(query_name, labeling.read_queries[idx])
    labeling.save_archive(save_path)


def run(labeling: Labeling, save_path: str):
    t0 = time.time()
    labeling.dbc.establish_connection()
    query_names = labeling.workload.queries
    query_names = query_names[:3]
    for i in trange(len(query_names), desc="Labeling queries"):
        run_single_query(labeling, query_names, i, save_path)
    labeling.dbc.close_connection()
    t1 = time.time() - t0
    logger.info(f'Finished labeling {len(query_names)} queries in {int(t1 / 60)}min {int(t1 % 60)}s.')
    return


if __name__ == "__main__":
    # pr = cProfile.Profile()
    # pr.enable()
    parser = argparse.ArgumentParser(description="Generate physical operator labels for input queries and save to json")
    parser.add_argument("queries", default=None, help="Directory in which .sql-queries are located")
    parser.add_argument("-o", "--output", default=None, help="Output dictionary save name")
    parser.add_argument("-c", "--config", default=None, help="Path to config file.")
    parser.add_argument("-db", "--database", default=None, help="Database name (e.g., imdb, stack, tpch."
                                                                "Reads from config.ini parameter.")
    parser.add_argument("-e", "--extension", default="True", help="Whether to enable pg_hint_plan or not. "
                                                                  "Useful when evaluating UES optimized queries.")
    args = parser.parse_args()

    args_query_path = args.queries
    args_save_path = args.output
    args_db_string = args.database
    args_config = args.config

    if not os.path.exists(args_config):
        raise argparse.ArgumentError(args_config, "Unknown config path.")

    path_config = PathConfig(args_config)
    _ = Logger(path_config, "labeling.log")
    logger = get_logger()

    if args_db_string == "imdb":
        args_db_string = path_config.PG_IMDB
    elif args_db_string == "stack":
        args_db_string = path_config.PG_STACK_OVERFLOW
    elif args_db_string == "tpch":
        args_db_string = path_config.PG_TPC_H

    if args_query_path is None:
        raise argparse.ArgumentError(args.queries, "No query path provided")
    args_wl = Workload(args_query_path, "labeling_workload")

    if args_save_path is None:
        raise argparse.ArgumentError(args.queries, "No output path provided")
    elif os.path.exists(args_save_path):
        logger.info("Loading existing Evaluation Dict")
        args_archive = load_json(args_save_path)
    else:
        logger.info("Creating new Evaluation Dict")
        args_archive = dict()

    if args.extension not in ["True", "False"]:
        raise ValueError(f"{args.extention} is not a valid option for -e.")
    else:
        args_use_extension = True if args.extension == "True" else False

    args_labeling = Labeling(args_db_string, args_archive, args_wl, use_extension=args_use_extension)
    logger.info(f"\nRunning Labeling on:\n {args_labeling.dbc.version()}.\n")

    try:
        run(args_labeling, args_save_path)
        logger.info("Finished Label Generation")
    except KeyboardInterrupt:
        args_labeling.dbc.close_connection()

    # pr.disable()
    # ps = pstats.Stats(pr)
    # ps.dump_stats("fastgres/logs/profile2.pstat")
