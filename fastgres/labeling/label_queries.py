import argparse
import os
import pstats
import time

from fastgres.baseline.database_connection import DatabaseConnection
from fastgres.baseline.hint_sets import HintSet
from fastgres.analysis_utility.tool import get_hint_set_combinations
from fastgres.definitions import PathConfig
from fastgres.baseline.utility import load_json, save_json
from fastgres.workloads.workload import Workload
from fastgres.baseline.log_utils import Logger, get_logger
from tqdm.contrib import telegram as tg
from tqdm import trange


class ChatInfo:
    def __init__(self, token: str, chat_id: str):
        self.token = token
        self.chat_id = chat_id


class Labeling:

    def __init__(self, connection_string: str, archive: dict, workload: Workload,
                 hints: list[int], base_timeout: float = 300, chat_info: ChatInfo = None, use_extension: bool = False):
        self.dbc = DatabaseConnection(connection_string, "labeling_connection", use_extension)
        self.archive = archive
        self.workload = workload
        self.base_timeout = base_timeout
        self.hints = hints
        self.chat_info = chat_info
        self.read_queries = self.workload.read_queries()
        self.iteration_list = [int(_) for _ in list(sorted(get_hint_set_combinations(self.hints)))]

    def save_archive(self, path: str):
        save_json(self.archive, path)

    def get_best_hint_set(self, query_name: str, query: str):
        # base timeout of 5 minutes should suffice
        timeout = self.base_timeout
        best_hint = None

        for hint_set_int in reversed(self.iteration_list):
            # logger.info("Evaluating Hint Set {}".format(hint_set_int))
            if hint_set_int in self.archive[query_name]:
                # logger.info('Found query entry')
                query_hint_time = self.archive[query_name][hint_set_int]
                if timeout is None or query_hint_time < timeout:
                    timeout = query_hint_time
                    best_hint = hint_set_int
                # logger.info('Found query but timed out')
                continue
            else:
                # logger.info('Evaluating Query')
                hint_set = HintSet(hint_set_int)
                query_hint_time = self.dbc.evaluate_hinted_query(query, hint_set, timeout)

            if query_hint_time is None:
                # timed out queries can not be counted
                # logger.info('Timed out query')
                continue
            else:
                # update dictionary
                hint_set_evaluations = self.archive[query_name]
                hint_set_evaluations[hint_set_int] = query_hint_time
                self.archive[query_name] = hint_set_evaluations

                # update timeout
                if timeout is None or query_hint_time < timeout:
                    timeout = query_hint_time
                    best_hint = hint_set_int

            # logger.info('Adjusted Timeout with Query: {}, Hint Set: {}, Time: {}'
            #             .format(query_name, int_to_binary(hint_set_int), query_hint_time))
        self.archive[query_name]['opt'] = best_hint


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

    labeling.get_best_hint_set(query_name, labeling.read_queries[idx])
    labeling.save_archive(save_path)


def run(labeling: Labeling, save_path: str):
    t0 = time.time()
    labeling.dbc.establish_connection()
    query_names = labeling.workload.queries
    if labeling.chat_info is not None:
        for i in tg.trange(len(query_names),
                           token=labeling.chat_info.token, chat_id=labeling.chat_info.chat_id,
                           desc="Labeling queries"):
            run_single_query(labeling, query_names, i, save_path)
    else:
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
    parser.add_argument("-hs", "--hints",
                        nargs="+", default=["32", "16", "8", "4", "2", "1"], help="hints to evaluate.")
    parser.add_argument("-ch", "--chat", nargs="+", default=None, help="Labeling can take some time. "
                                                                       "You can pass your telegram chat token and id "
                                                                       "to get updates  on the labeling progress.")
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
        # initialize dict over all queries
        logger.info("Creating new Evaluation Dict")
        args_archive = dict(zip(args_wl.queries, [dict() for _ in range(len(args_wl.queries))]))
    string_hints = args.hints
    arg_hints = [int(string_hint) for string_hint in string_hints]

    args_chat = args.chat
    if args_chat is not None:
        args_token, args_id = args.chat
        args_chat = ChatInfo(args_token, args_id)

    if args.extension not in ["True", "False"]:
        raise ValueError(f"{args.extention} is not a valid option for -e.")
    else:
        args_use_extension = True if args.extension == "True" else False

    args_labeling = Labeling(args_db_string, args_archive, args_wl, arg_hints,
                             chat_info=args_chat, use_extension=args_use_extension)
    logger.info(f"\nRunning Labeling on:\n {args_labeling.dbc.version()}.\n")

    try:
        run(args_labeling, args_save_path)
        logger.info("Finished Label Generation")
    except KeyboardInterrupt:
        args_labeling.dbc.close_connection()

    # pr.disable()
    # ps = pstats.Stats(pr)
    # ps.dump_stats("fastgres/logs/profile2.pstat")
