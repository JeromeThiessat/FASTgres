import os
from configparser import ConfigParser

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))


# class StandardConfig:
#     # TODO: Fix from current config.ini handling to something more elegant
#
#     cfg = ConfigParser()
#     cfg.read(ROOT_DIR + "/config.ini")
#
#     dbs = cfg["DBConnections"]
#     PG_IMDB = dbs["imdb"]
#     PG_STACK_OVERFLOW = dbs["stack_overflow"]
#     PG_STACK_OVERFLOW_REDUCED_16 = dbs["stack_overflow_reduced_16"]
#     PG_STACK_OVERFLOW_REDUCED_13 = dbs["stack_overflow_reduced_13"]
#     PG_STACK_OVERFLOW_REDUCED_10 = dbs["stack_overflow_reduced_10"]
#     PG_TPC_H = dbs["tpc_h"]


class PathConfig:

    def __init__(self, path: str):
        self.path = path
        self.ROOT_DIR = os.path.dirname(self.path)
        self.LOG_DIR = self.ROOT_DIR + "/logs"
        # if not os.path.exists(self.LOG_DIR):
        #     os.mkdir(self.LOG_DIR)
        cfg = ConfigParser()
        cfg.read(self.ROOT_DIR + "/config.ini")

        dbs = cfg["DBConnections"]
        self.PG_IMDB = dbs["imdb"]
        self.PG_STACK_OVERFLOW = dbs["stack_overflow"]
        self.PG_STACK_OVERFLOW_REDUCED_16 = dbs["stack_overflow_reduced_16"]
        self.PG_STACK_OVERFLOW_REDUCED_13 = dbs["stack_overflow_reduced_13"]
        self.PG_STACK_OVERFLOW_REDUCED_10 = dbs["stack_overflow_reduced_10"]
        self.PG_TPC_H = dbs["tpc_h"]
