
import logging
from fastgres.definitions import PathConfig


class Logger:

    def __init__(self, path_config: PathConfig, log_name: str = "workload.log"):
        logging.basicConfig(filename=path_config.LOG_DIR + f'/{log_name}', filemode='w',
                            format='%(asctime)s %(name)s - %(levelname)s - %(message)s', level=logging.INFO)


def get_logger():
    logger = logging.getLogger("Logger")
    logger.setLevel(logging.INFO)
    return logger
