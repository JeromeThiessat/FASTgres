
import pandas as pd

from fastgres.baseline.database_connection import DatabaseConnection
from fastgres.baseline.log_utils import get_logger
from tqdm import tqdm


class FastgresLabelEncoder:
    def __init__(self):
        self.classes_ = None
        self.encoder = dict()

    def fit(self, y: list, sorty_by: list) -> None:
        self.classes_ = pd.Series(y).unique()
        self.encoder = get_sorted_dict(y, sorty_by)
        return

    def transform(self, values: list) -> list:
        return_list = list()
        for item in values:
            return_list.append(self.encoder[item])
        return return_list


def get_sorted_dict(values, sort_by):
    mixed = list(zip(values, sort_by))
    mixed.sort(key=lambda x: x[1])
    mixed_dict = {mixed[i][0]: i for i in range(len(values))}
    return mixed_dict


def build_label_encoders(db_connection: DatabaseConnection):
    unhandled = set()
    label_encoders = dict()
    conn, cursor = db_connection.establish_connection()
    cursor.execute("SELECT table_name "
                   "FROM information_schema.tables "
                   "WHERE table_schema = 'public'")
    logger = get_logger()
    for table in cursor.fetchall():
        t = table[0]
        cursor.execute("SELECT column_name, data_type "
                       "FROM information_schema.columns "
                       "WHERE table_name = '{}';".format(t))
        for column, d_type in tqdm(cursor.fetchall()):
            if d_type == 'character varying' or d_type == 'character':
                skip = False
                if "stack_overflow" in db_connection.connection_string:
                    skipped_string_columns = {
                        "account": ["display_name"],
                        "answer": ["title", "body"],
                        "question": ["title", "tagstring", "body"],
                        "site": ["site_name"],
                        "tag": ["name"],
                        "badge": ["name"],
                        "comment": ["body"]
                    }
                    # skipping all unneeded columns
                    for skipped_table in skipped_string_columns:
                        if t == skipped_table and column in skipped_string_columns[skipped_table]:
                            skip = True
                            break
                if skip:
                    continue

                cursor.execute("SELECT {}, COUNT({}) FROM {} GROUP BY {}".format(column, column, t, column))
                filter_list = list()
                for filter_value, cardinality in cursor.fetchall():
                    filter_list.append((filter_value, cardinality))
                logger.info("Fitting label encoder to table: {}, column: {}".format(t, column))
                label_encoder = FastgresLabelEncoder()
                label_encoder.fit(*list(zip(*filter_list)))
                try:
                    label_encoders[t][column] = label_encoder
                except KeyError:
                    label_encoders[t] = dict()
                    label_encoders[t][column] = label_encoder
            else:
                unhandled.add(d_type)
    cursor.close()
    conn.close()
    logger = get_logger()
    logger.info(f"Unhandled types for label encoding: {unhandled}")

    return label_encoders
