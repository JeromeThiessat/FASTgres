
import math
from time import time
from fastgres.baseline.database_connection import DatabaseConnection
from fastgres.models.context import Context
from fastgres.models.experience import Experience
from fastgres.models.train_model import Settings, Model
from fastgres.baseline.hint_sets import HintSet, set_hints
from fastgres.query_encoding.feature_extractor import EncodingInformation
from fastgres.query_encoding.query import Query
from fastgres.query_encoding.encoded_query import EncodedQuery
from fastgres.baseline.log_utils import get_logger
from fastgres.workloads.workload import Workload
from typing import Union


class Synchronizer:

    def __init__(self, context: Context, db_connection: DatabaseConnection, encoding_information: EncodingInformation,
                 experience: Experience, settings: Settings, autonomy: bool = True):
        self.context: Context = context
        self.experience = experience
        self.encoding_info = encoding_information
        self.model = None
        self.db_connection = db_connection
        self.autonomy = autonomy
        self.future_model = None
        self.future_experience = None
        self.cooldown = 0.0
        self.settings = settings

    def __str__(self):
        return_string = f"Synchronizer running on context:\n\n{str(self.context)}\n\n" \
                        f"Database connection: {self.db_connection.name}\n" \
                        f"Autonomy: {self.autonomy}"
        return return_string

    def build_model(self):
        if self.model is None:
            self.model = Model(self.settings, self.context)
        self.model.train(self.experience, self.context, self.encoding_info)

    def reduce_cooldown(self, reduction: float):
        self.cooldown -= reduction

    def pre_execution(self):
        if self.future_model is not None and self.cooldown <= 0:
            logger = get_logger()
            logger.info("Deploying new model on context: {}".format(self.context))
            self.model = self.future_model
            self.future_model = None
            self.cooldown = 0

    def post_execution(self, encoded_query: EncodedQuery, execution_time: Union[float, None]):
        logger = get_logger()
        global_reduction = 0.0
        self.experience.set_timeout(self.settings.percentile, self.settings.absolute)
        current_timeout = self.experience.timeout
        if execution_time is not None and execution_time < current_timeout:
            logger.info("Critical query observed")
            execution_time = self.experience.get_default_time(encoded_query.query.name)

        if self.cooldown <= 0 and self.future_model is None:
            if self.future_experience is None:
                logger.info(f"Model retraining initiated on context: {self.context}")
                self.future_experience = Experience(self.experience.archive)
                self.future_experience.add_experience(encoded_query.query)

                self.experience.add_experiences(self.future_experience.experience)
                self.build_model()
                self.cooldown += sum([self.future_experience.get_opt_time(query.name)
                                      for query in self.future_experience.experience])
            else:
                global_reduction += current_timeout

        global_reduction += execution_time
        return global_reduction

        # if result_time is None or result_time >= self.timeout:
        #     # the query is critical and should and possibly trigger retraining
        #     # same format as experience for easy handling
        #     print("Caught timeout")
        #     # at this point we already decide to return PG default as retraining does not influence this decision
        #     result_time = self.archive[query_name]["63"]
        #
        #     if self.cooldown <= 0 and self.new_model is None:
        #         self.critical[query_name]["featurization"] = deepcopy(query_featurization)
        #         self.critical[query_name]["label"] = self.archive[query_name]["opt"]
        #         self.critical[query_name]["time"] = self.archive[query_name][str(self.archive[query_name]["opt"])]
        #
        #         # we have capacity to train a new model, double check just in case
        #         # first we need to move any critical query to our experience
        #         # these queries decide on how long our cooldown will be
        #         # these four steps should always be taken when retraining
        #         labeling_time = self.move_critical_to_experience()
        #         self.train()
        #         self.update_timeout()
        #         self.cooldown += labeling_time
        #     else:
        #         # we still have a model that is being trained
        #         # -> we can additionally deduct the timeout for critical queries
        #         for c in context_models:
        #             model = context_models[c]
        #             if not isinstance(model, int):
        #                 model.cooldown -= self.timeout
        #         # self.cooldown -= self.timeout
        #
        # # At the end, we have to deduct our query runtime from the current cooldown
        # # self.cooldown -= result_time
        # for c in context_models:
        #     model = context_models[c]
        #     if not isinstance(model, int):
        #         model.cooldown -= result_time
        return

    def run_query(self, query: Query, hint_set: HintSet, workload: Workload, use_timeout: bool = True):
        query = workload.read_query(query.name)
        conn, cur = self.db_connection.establish_connection()
        if not use_timeout:
            # sets standard timeout for experience based handling, defaults to Postgres' default run times
            timeout = self.experience.timeout
            # catch faulty timeout measures due to floating point inaccuracies
            if timeout <= 0.0:
                logger = get_logger()
                logger.info("Adjusted timeout to 0.1 to avoid PostgreSQL issues with timeouts <= 0.0")
                timeout = 0.1
            # timeout is in milliseconds: round up and cast to int
            time_out = "SET statement_timeout = '{}ms'".format(int(math.ceil(timeout * 1000)))
            cur.execute(time_out)

        set_hints(hint_set, cur)
        try:
            start = time()
            cur.execute(query)
            stop = time()
            measured_time = stop-start
        except:
            measured_time = None
        cur.close()
        conn.close()

        return measured_time

