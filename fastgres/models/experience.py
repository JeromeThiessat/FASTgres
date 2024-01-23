
import numpy as np

from fastgres.query_encoding.query import Query


class Experience:

    def __init__(self, archive: dict, use_default_time: bool = True):
        self.experience = set()
        self.archive = archive
        self.use_default = use_default_time
        self.time = None
        self.timeout = None

    def add_experience(self, query: Query):
        if query not in self.experience:
            self.experience.add(query)

    def add_experiences(self, queries: list[Query]):
        for query in queries:
            self.add_experience(query)

    def determine_time(self):
        self.determine_default_times() if self.use_default else self.determine_label_time()

    def determine_default_times(self):
        self.time = [self.archive[query.name]['63'] for query in self.experience]

    def determine_label_time(self):
        self.time = [self.archive[query.name][str(self.archive[query.name]['opt'])]
                     for query in self.experience]

    def set_timeout(self, percentile: int = 99, absolute: float = 1.0):
        if self.time is None:
            self.determine_default_times()
        new_timeout = np.percentile(self.time, percentile)
        self.timeout = max(absolute, new_timeout)

    def sort_by_context(self):
        context_query_dict = dict()
        for query in self.experience:
            try:
                context_query_dict[query.context].append(query)
            except KeyError:
                context_query_dict[query.context] = list()
                context_query_dict[query.context].append(query)
        return context_query_dict

    def get_default_time(self, query_name: str) -> float:
        return self.archive[query_name]["63"]

    def get_opt(self, query_name: str) -> int:
        return int(self.archive[query_name]['opt'])

    def get_opt_time(self, query_name: str) -> float:
        return self.archive[query_name][str(self.get_opt(query_name))]
