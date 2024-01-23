import numpy as np
from sklearn.ensemble import GradientBoostingClassifier

from fastgres.models.experience import Experience
from fastgres.models.context import Context
from fastgres.query_encoding.feature_extractor import EncodingInformation
from fastgres.query_encoding.query_encoder import Encoder


class Settings:

    def __init__(self, seed: int, percentile: int, absolute: float, estimators: int = 100, max_depth: int = 1000):
        self.estimators = estimators
        self.max_depth = max_depth
        self.seed = seed
        self.percentile = percentile
        self.absolute = absolute


class Model:

    def __init__(self, settings: Settings, context: Context):
        self.model: GradientBoostingClassifier = None
        self.settings = settings
        self.context = context
        self.has_single_label = True

    def build_model(self):
        self.model = GradientBoostingClassifier(n_estimators=self.settings.estimators,
                                                max_depth=self.settings.max_depth,
                                                random_state=self.settings.seed)

    def train(self, experience: Experience, context: Context, encoding_info: EncodingInformation,
              distributed: bool = False):
        if self.model is None:
            self.build_model()

        encoder = Encoder(context, encoding_info)
        encodings = list()
        labels = list()
        for query in sorted(experience.experience):
            encoded_query = encoder.encode_query(encoder.build_feature_dict_old(query))
            encodings.append(encoded_query)
            labels.append(experience.get_opt(query.name))

        uniques = np.unique(labels)
        if len(uniques) > 1 and not isinstance(self.model, int):
            self.model = self.model.fit(encodings, labels)
            self.has_single_label = False
        else:
            self.model = int(uniques[0])

    def predict(self, encoded_query: list[float]):
        if isinstance(self.model, int):
            return self.model
        return int(self.model.predict(np.reshape(encoded_query, (1, -1)))[0])
