
from fastgres.models.context import Context
from fastgres.query_encoding.query import Query
from fastgres.query_encoding.feature_extractor import EncodingInformation
from fastgres.query_encoding.query_encoder import Encoder


class EncodedQuery:

    def __init__(self, context: Context, query: Query, encoding_information: EncodingInformation,
                 eager_encode: bool = True):
        self.context = context
        self.query = query
        self.enc_info = encoding_information
        self.encoded_query = None
        if eager_encode:
            self.encode_query()

    def encode_query(self):
        encoder = Encoder(self.context, self.enc_info)
        self.encoded_query = encoder.encode_query(encoder.build_feature_dict_old(self.query))
