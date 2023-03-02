
import datetime
import hashlib
import numpy as np
import utility as u
from query import Query


def get_explain_analyze_result(db_string, query_path):
    conn, cursor = u.establish_connection(db_string)
    with open(query_path, 'r', encoding='utf-8') as f:
        q = f.read()
    cursor.execute("EXPLAIN ANALYZE " + q)
    res = cursor.fetchall()
    cursor.close()
    conn.close()
    return res


def encode_query(context: frozenset, feature_dict: dict, d_type_dict: dict):
    encoded_query = list()
    for table in context:
        if table not in feature_dict.keys():
            # extend by all column encodings to 0
            encoded_query.extend([0]*4*len(d_type_dict[table].keys()))
            continue
        column_dict = feature_dict[table]
        for column in d_type_dict[table].keys():
            if column in column_dict:
                entry = column_dict[column]
                encoded_query.extend(entry)
            else:
                encoded_query.extend([0] * 4)
    return encoded_query


def encode_operator(operator):
    try:
        if operator == "not_like":
            encoded_operator = u.operator_dictionary["like"]
        else:
            encoded_operator = u.operator_dictionary[operator]
    except KeyError:
        raise KeyError("Could not encode operator: {}. Operator dictionary needs to be adjusted"
                       .format(operator))
    return encoded_operator


def build_feature_dict(query: Query, db_string: str, mm_dict: dict,  label_encoders: dict, wildcard_dict: dict,
                       unhandled_operators:  set, unhandled_column_types: set, skipped_table_columns: dict):

    d_type_dict = u.build_db_type_dict(db_string)
    featurization_dict = u.tree()
    for table in query.attributes:
        for column in query.attributes[table]:
            feature_vector = [0.0] * 4
            for operator in query.attributes[table][column]:
                column_type = d_type_dict[table][column]
                filter_value = query.attributes[table][column][operator]
                feature_vector[:3] = encode_operator(operator)

                if isinstance(filter_value, dict):
                    # TODO: Handle if not gt offset date joins in future scenarios
                    continue

                if column_type == "integer":
                    # some weird join in stack
                    if isinstance(filter_value, str):
                        # print("Probably some undetected joins: {} ".format(filter_value))
                        continue
                    offset = 0.001
                    # encode min max
                    min_v, max_v = mm_dict[table][column]
                    feature_vector[3] = u.min_max_encode(min_v, max_v, filter_value, offset)
                elif column_type == "character varying":
                    offset = 1.0
                    if table in skipped_table_columns:
                        if column in skipped_table_columns[table]["columns"]:
                            max_enc = 2 ** 64  # md5 output is 64 bit standard
                            min_enc = 0  # should be min for hashes
                            offset = 1
                            if operator == "in":
                                merged_string_hash = list()
                                for string in filter_value:
                                    b_string = bytes(string, "utf-8")
                                    hash_v = int.from_bytes(hashlib.md5(b_string).digest()[:8], 'little')
                                    merged_string_hash.append(hash_v)
                                hash_value = int(round(sum(merged_string_hash) / len(filter_value), 0))
                            else:
                                b_string = bytes(filter_value, "utf-8")
                                hash_value = int.from_bytes(hashlib.md5(b_string).digest()[:8], 'little')
                            # TODO: Counter-check skipped table encodings for expressiveness
                            feature_vector[3] = (hash_value + offset - min_enc) / (max_enc - min_enc + offset)

                            # since we continue, we need to save here
                            featurization_dict[table][column] = feature_vector
                            continue

                    # single, ensemble, wildcard
                    if operator == "eq" or operator == "lt" or operator == "gt":
                        encoder = label_encoders[table][column]
                        min_v, max_v = 0, len(encoder.classes_)
                        adjusted_min = min_v - offset
                        try:
                            transformed = encoder.transform([filter_value])[0]
                        except KeyError:
                            print("Filter error, defaulting to 0 encoding: ", filter_value)
                            transformed = min_v
                        encoded_filter_value = (transformed - adjusted_min) / \
                                               (max_v - adjusted_min)
                        if not operator == "eq":
                            pass
                        feature_vector[3] = encoded_filter_value

                    elif operator == "in":
                        encoder = label_encoders[table][column]
                        min_v, max_v = 0, len(encoder.classes_)
                        adjusted_min = min_v - offset
                        try:
                            transformed = encoder.transform(filter_value)
                        except KeyError:
                            transformed = list()
                            for filter_value_i in filter_value:
                                try:
                                    transformed.append(encoder.transform([filter_value_i])[0])
                                except KeyError:
                                    transformed.append(-1)

                        encoded_filter_value = (np.array(transformed) - adjusted_min) / (max_v - adjusted_min)
                        encoded_filter_value = sum(encoded_filter_value) / len(encoded_filter_value)
                        feature_vector[3] = encoded_filter_value

                    elif operator == "like" or operator == "not_like":
                        if column not in wildcard_dict[table]:
                            encoded_filter_value = 1.0
                        elif filter_value in wildcard_dict[table][column]:
                            offset = 1.0
                            min_v, max_v = 0, wildcard_dict[table]['max']
                            adjusted_min = min_v - offset
                            encoded_filter_value = (wildcard_dict[table][column][filter_value] - adjusted_min) / \
                                                   (max_v - adjusted_min)
                        else:
                            # assume that cardinalities are as high as they can get
                            encoded_filter_value = 1.0
                        feature_vector[3] = encoded_filter_value

                    elif operator == "neq":
                        if filter_value == '':
                            encoded_filter_value = 1.0
                        else:
                            # default eq encoding
                            encoder = label_encoders[table][column]
                            min_v, max_v = 0, len(encoder.classes_)
                            adjusted_min = min_v - offset
                            try:
                                transformed = encoder.transform([filter_value])[0]
                            except KeyError:
                                transformed = min_v
                            encoded_filter_value = (transformed - adjusted_min) / \
                                                   (max_v - adjusted_min)
                        feature_vector[3] = encoded_filter_value

                    else:
                        unhandled_operators.add(operator)
                elif column_type == "timestamp without time zone":
                    # timestamps should always be caught by stc-dict
                    try:
                        # timestamps are ms-exact and the probability of having multiples is approaching 0
                        offset = datetime.timedelta(days=1)
                        # encode min max
                        min_v, max_v = mm_dict[table][column]
                        format_string = "%Y-%m-%d"
                        filter_value = datetime.datetime.strptime(filter_value, format_string)
                        feature_vector[3] = u.min_max_encode(min_v, max_v, filter_value, offset)
                    except:
                        # This is a join in our scenario and can be neglected
                        continue
                else:
                    unhandled_column_types.add(column_type)
                featurization_dict[table][column] = feature_vector
            pass
        pass

    if unhandled_column_types:
        # print('Unhandled column types are: {}'.format(unhandled_column_types))
        pass
    if unhandled_operators:
        # print('Unhandled operators are: {}\n'.format(unhandled_operators))
        pass
    if not unhandled_column_types and not unhandled_operators:
        pass

    return featurization_dict


def build_table_column_dict(queries: list[str], query_path: str) -> dict:
    # Preparing our context-encoding
    table_column_dict = u.tree()
    for query_name in queries:
        query = Query(query_name, query_path)
        tables = set(query.attributes.keys())
        for table in tables:
            columns = set(query.attributes[table])
            try:
                table_column_dict[table] = table_column_dict[table].union(columns)
            except:
                table_column_dict[table] = columns
    return table_column_dict

# TODO: useful when not encoding queries on the fly
# def main(query_path, mm_path, label_path, db_string, wildcard_path, output_path, dbi_path, skipped_path):
#     queries = u.get_queries(query_path)
#     # we need 3 types of information:
#     # integer
#     mm_dict = u.load_json(mm_path + "mm_dict.json")
#     # string
#     label_encoders = u.load_pickle(label_path + "label_encoders.pkl")
#     # wildcard
#     wildcard_dict = u.load_json(wildcard_path + "wildcard_dict.json")
#     # skipped columns: table -> columns | rows (count)
#     skipped_table_columns = u.load_json(skipped_path)
#
#     unhandled_operators = set()
#     unhandled_column_types = set()
#
#     feature_dict = u.tree()
#     d_type_dict = u.build_db_type_dict(db_string)
#
#     for query_name in queries[:20]:
#         query = Query(query_name, query_path)
#         t0 = time.time()
#         f_dict = build_feature_dict(query, db_string, mm_dict, label_encoders, wildcard_dict, unhandled_operators,
#                                     unhandled_column_types, skipped_table_columns)
#         t1 = time.time()
#         encoded_query = encode_query(frozenset(d_type_dict.keys()), f_dict, d_type_dict)
#         t2 = time.time()
#         print("Query: {} took: {} to build its features and: {} to encode".format(query_name, t1-t0, t2-t1))
#         feature_dict[query_name] = f_dict
#     # u.save_pickle(feature_dict_single, output_path)
#
#     return


# if __name__ == "__main__":
#     print("Using featurize v0.12")
#     parser = argparse.ArgumentParser(description="Generate featurization for input queries. This is highly specific to "
#                                                  "the stackoverflow workload and possibly breaks with any other set of "
#                                                  "queries")
#     parser.add_argument("queries", default=None, help="Directory in which .sql-queries are located")
#     parser.add_argument("-o", "--output", default=None, help="Output featurization dictionary save name")
#     parser.add_argument("-s", "--strategy", default="first-join", help="Featurization strategy that is used")
#     parser.add_argument("-l", "--label", default="", help="Label encoder path.")
#     parser.add_argument("-mm", "--minmax", default="", help="MinMax dictionary path.")
#     parser.add_argument("-w", "--wildcard", default="", help="Path to wildcard dictionary.")
#     parser.add_argument("-db", "--database", default=u.PG_IMDB, help="Database connection string in the form of "
#                                                                      "psycopg2.")
#     parser.add_argument("-di", "--dinfo", default="", help="Database information path, used for db type dictionary.")
#     parser.add_argument("-sc", "--skippedcolumns", default=None, help="Database columns to skip using preprocessing.")
#     args = parser.parse_args()
#
#     args_mm = args.minmax
#     args_label = args.label
#     args_db_string = args.database
#     if args_db_string == "stack":
#         args_db_string = u.PG_STACK_OVERFLOW
#     if args_db_string == "imdb":
#         args_db_string = u.PG_IMDB
#     args_wildcard = args.wildcard
#     args_output = args.output
#     args_type_path = args.dinfo
#     args_skip = args.skippedcolumns
#
#     main(args.queries, args_mm, args_label, args_db_string, args_wildcard, args_output, args_type_path, args_skip)
