
from fastgres.baseline.database_connection import DatabaseConnection
from fastgres.baseline.hint_sets import HintSet
from fastgres.models.sync_model import Synchronizer
from fastgres.query_encoding.feature_extractor import EncodingInformation
from fastgres.workloads.workload import Workload
from fastgres.query_encoding.query import Query
from fastgres.query_encoding.encoded_query import EncodedQuery
from fastgres.models.context import Context
from fastgres.baseline.utility import load_pickle
from fastgres.definitions import PathConfig
from fastgres.baseline.log_utils import Logger


class Fastgres:

    def __init__(self, workload: Workload, db_connection: DatabaseConnection,
                 config_path: PathConfig,
                 model_path: str,
                 statistics_path: str,
                 eager_loading: bool = True):
        self.workload = workload
        self.db_c = db_connection
        self.enc_info = None
        self.models = None
        self.config = config_path
        self.model_path = model_path
        self.stats_path = statistics_path
        self.logger = Logger(config_path)
        if eager_loading:
            self._load_model()
            self._load_enc_info()

    def _load_model(self):
        if "stack" in self.workload.name.lower():
            path = self.model_path + "/stack/fastgres.pkl"
        elif "job" in self.workload.name.lower():
            path = self.model_path + "/job/fastgres.pkl"
        else:
            raise NotImplemented("Unsupported workload name")
        context_models = load_pickle(path)
        self.models = context_models

    def _load_enc_info(self):
        if "stack" in self.workload.name.lower():
            path = self.stats_path + "/stack/"
        elif "job" in self.workload.name.lower():
            path = self.stats_path + "/job/"
        else:
            raise NotImplemented("Unsupported workload name")
        self.enc_info = EncodingInformation(self.db_c, path, self.workload)
        # encoding_info = self._get_encoding_information()
        # Updated to eager loading by default
        # encoding_info.load_encoding_info()
        # self.enc_info = encoding_info

    # def _get_encoding_information(self):
    #     if "stack" in self.workload.name.lower():
    #         path = self.stats_path + "/stack/"
    #     elif "job" in self.workload.name.lower():
    #         path = self.stats_path + "/job/"
    #     else:
    #         raise NotImplemented("Unsupported workload name")
    #     return EncodingInformation(self.db_c, path, self.workload)

    def predict(self, query_name: str, reload: bool = False):
        if self.models is None or reload:
            self._load_model()
        if self.enc_info is None or reload:
            self._load_enc_info()

        query = Query(query_name, self.workload)
        context_set = query.context
        synchronizer: Synchronizer = self.models[context_set]

        context = Context(context_set)
        # set up context init with given frozenset
        # context.add_context(context_set)
        encoded_query = EncodedQuery(context, query, self.enc_info)
        # set up eager query encoding as default
        # encoded_query.encode_query()
        integer_hint_set = synchronizer.model.predict(encoded_query.encoded_query)
        return HintSet(integer_hint_set)
