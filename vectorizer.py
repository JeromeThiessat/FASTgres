
import os
import re
import numpy as np
import pandas as pd
import utility as u
from sklearn.preprocessing import LabelEncoder


# most of this code is unused for now
# we overwrite this function such that we can easily write encoders that do not auto-sort themselves
class LabelEncoder(LabelEncoder):

    def fit(self, y):
        # this line can cause memory issues
        # y = column_or_1d(y, warn=True)
        self.classes_ = pd.Series(y).unique()
        return self


def look_up_alias(table_alias, from_part):
    # lookup alias in select part
    table = None
    # print('alias search from part ', from_part)
    # print('table alias ', table_alias)
    for sel_dict in from_part:
        if isinstance(sel_dict, str):
            if sel_dict == table_alias:
                # case no alias was used
                table = sel_dict
            continue
        if sel_dict['name'] == table_alias:
            table = sel_dict['value']
            continue
            # print(table)
    if table is None:
        # the table has no alias and therefore could be used directly
        # print('No table found from alias, returning alias as table name')
        return table_alias
        # raise ValueError('No table found from alias')
    return table


def get_alias_to_table(from_part):
    if not isinstance(from_part, list):
        # instance of one elementary parts
        from_part = [from_part]
    alias_to_table = dict()
    for from_segment in from_part:
        if isinstance(from_segment, dict):
            # table has an alias
            alias = from_segment['name']
            table = from_segment['value']
        else:
            # no alias
            table = from_segment
            alias = table
        alias_to_table[alias] = table
    return alias_to_table


def replace_ilike_exp(path, queries):
    for i in range(len(queries)):
        query = queries[i]
        with open(path + query, 'r', encoding='utf-8') as f:
            q = f.read()
        pattern = r'[^\s^\(]+[\s]ILIKE[\s][^\s^\)]+'
        res = re.findall(pattern, q)
        # print(res)
        if res:
            res = res[0].split(' ')
            replacement = 'LOWER({}) LIKE LOWER({})'.format(res[0], res[2])
            # print(replacement)
            new_q = re.sub(pattern, replacement, q)
            # print(new_q)
            with open(path + query, 'w', encoding='utf-8') as f:
                f.write(new_q)
    return


def replace_interval_exp(path, queries):
    for i in range(len(queries)):
        query = queries[i]
        with open(path + query, 'r', encoding='utf-8') as f:
            q = f.read()
        pattern = r'[^\s]+[\s][^\s]+::interval'
        res = re.findall(pattern, q)
        # print(res)
        if res:
            res = res[0].split('::')
            # print(res)
            replacement = "interval {}".format(res[0])
            # print(replacement)
            new_q = re.sub(pattern, replacement, q)
            with open(path + query, 'w', encoding='utf-8') as f:
                f.write(new_q)
    return


def get_where_segments(where_part):
    where_keys = ['and', 'or', 'not']
    where_keys_dict = dict()
    where_part_keys = where_part.keys()
    for key in where_keys:
        if key in where_part_keys:
            where_keys_dict[key] = where_part[key]
    return where_keys_dict


def classify_eq_segment(left_element, right_element, alias_to_table, type_dict):
    predicates = set()
    alphabet = dict()

    filter_pred = right_element[0]
    alias, column = left_element.split('.')
    try:
        table = alias_to_table[alias]
    except:
        raise KeyError('Table alias not found in alias_to_table keys')
    d_type = type_dict[table][column]

    if isinstance(filter_pred, str):
        if not u.is_float(filter_pred):
            # join
            # print('Encountered join')
            return predicates, alphabet
        else:
            # float with no 'literal'
            raise NotImplementedError('Predicates not yet supported')
    elif isinstance(filter_pred, dict):
        if 'literal' in filter_pred.keys():
            literal = filter_pred['literal']
            # print('literal: ', literal)
            predicates.add(left_element)
            if u.is_float(literal) and d_type == 'integer':
                # print('is {} a float?'.format(literal))
                pass
            else:
                # print('is {} a string?'.format(literal))
                alphabet = u.add_or_create_dict_entry(alphabet, left_element, literal)
            return predicates, alphabet
        else:
            raise NotImplementedError('no literal in dict')
    elif isinstance(filter_pred, int):
        # something like eq on year
        predicates.add(left_element)
        alphabet = u.add_or_create_dict_entry(alphabet, left_element, filter_pred)
        return predicates, alphabet
    else:
        print(isinstance(filter_pred, int))
        print("Filter predicate: {}, {} caused an error.".format(left_element, right_element))
        raise NotImplementedError('Predicates not yet supported')


def classify_lte_gte_segment(left_element, right_element):
    predicates = set()
    alphabet = dict()
    filter_pred = right_element[0]
    # print('lgte filter: ', filter_pred)
    if u.is_float(filter_pred):
        # print('Is float: ', filter_pred)
        predicates.add(left_element)
        return predicates, alphabet
    elif isinstance(filter_pred, str):
        # join
        # print('Is {} in a join?'.format(filter_pred))
        return predicates, alphabet
    elif isinstance(filter_pred, dict):
        if 'add' in filter_pred.keys():
            fp_add = filter_pred['add']
            if not u.is_float(fp_add[0]):
                # join
                return predicates, alphabet
            else:
                raise NotImplementedError('lgte add operation not yet supported')
        elif 'cast' in filter_pred.keys():
            fp_cast = filter_pred['cast']
            if 'literal' in fp_cast[0].keys():
                # predominantly used for casting date filters
                right_literal = fp_cast[0]['literal']
                predicates.add(left_element)
                alphabet = u.add_or_create_dict_entry(alphabet, left_element, right_literal)
                # print('Cast literal result:\n', predicates, alphabet)
                return predicates, alphabet
            else:
                raise NotImplementedError('Other combination than cast + literal type not supported yet')
        else:
            raise NotImplementedError('lgte dict type not supported yet')
    else:
        raise NotImplementedError('Other lgte types not supported yet')


def classify_in_segment(left_element, right_element):
    predicates = set()
    alphabet = dict()
    filter_pred = right_element[0]
    # print(filter_pred)
    fp_keys = filter_pred.keys()
    if 'literal' in fp_keys:
        alphabet = u.add_or_create_dict_entry(alphabet, left_element, filter_pred['literal'])
        predicates.add(left_element)
        # print('in/like-result', predicates, alphabet)
        return predicates, alphabet
    elif 'lower' in fp_keys:
        lower_right = filter_pred['lower']
        if 'literal' in lower_right.keys():
            lower_right = lower_right['literal']
        else:
            raise NotImplementedError('case insensitive like has no literal value')
        lower_left = left_element['lower']
        if not u.is_float(lower_right):
            alphabet = u.add_or_create_dict_entry(alphabet, lower_left, lower_right)
            predicates.add(lower_left)
            return predicates, alphabet
        else:
            raise NotImplementedError('lower cast on float in like segment not implemented yet')
    else:
        raise NotImplementedError('in/like-segment not classifiable')


def classify_exist_segments(segment, type_dict):
    sub_statement = segment['exists']
    # print('Sub statement, maybe query? ', sub_statement)
    if u.is_query(sub_statement):
        sub_st_where = sub_statement['where']
        from_p = sub_statement['from']
        # print(isinstance(from_p, list))
        alias_to_table = get_alias_to_table(from_p)
        # print(alias_to_table)
        pred = set()
        alph = dict()
        for key in sub_st_where.keys():
            # and, or, not
            pred_t, alph_t = get_segment_info(sub_st_where[key], alias_to_table, type_dict)
            pred = pred.union(pred_t)
            alph = u.merge_dicts(alph, alph_t)
        # print('Subquery predicate and alphabet:\n', pred, alph)
        return pred, alph
    else:
        # statement is just a column
        raise NotImplementedError('Subquery type unknown')


def get_segment_info(segment, alias_to_table, type_dict):
    predicates = set()
    alphabet = dict()
    for predicate_line in segment:
        # print(predicate_line)
        for operator in predicate_line.keys():
            # handle nested queries
            if operator == 'exists':
                pred, alph = classify_exist_segments(predicate_line, type_dict)
                predicates = predicates.union(pred)
                alphabet = u.merge_dicts(alphabet, alph)
                continue
            elif operator == 'not':
                sub_predicate_line = predicate_line['not']
                # print('SPL ', sub_predicate_line)
                if isinstance(sub_predicate_line, dict):
                    for op in sub_predicate_line.keys():
                        if op == 'exists':
                            pred, alph = classify_exist_segments(sub_predicate_line, type_dict)
                            predicates = predicates.union(pred)
                            alphabet = u.merge_dicts(alphabet, alph)
                            continue
                        else:
                            raise NotImplementedError('Operator combination other than NOT EXISTS not supported yet')
                    continue
                else:
                    raise NotImplementedError('Not and non subdict not supported yet')
            elif operator not in u.operator_dictionary:
                # something like 'or' is not supported
                continue
                raise ValueError('Operator "{}" not classifiable'.format(operator))

            # Tricky part, which predicates to choose
            operator_line = predicate_line[operator]
            left_element = operator_line[0]
            right_element = operator_line[1:]

            if operator == 'eq' or operator == 'IS':
                pred, alph = classify_eq_segment(left_element, right_element, alias_to_table, type_dict)
            elif operator == 'gte' or operator == 'lte':
                pred, alph = classify_lte_gte_segment(left_element, right_element)
            elif operator == 'lt' or operator == 'gt':
                # careful here, double check for correctness
                pred, alph = classify_lte_gte_segment(left_element, right_element)
            elif operator == 'neq':
                # using eq method might just be enough
                # print('Using eq for neq')
                pred, alph = classify_eq_segment(left_element, right_element, alias_to_table, type_dict)
            elif operator == 'like':
                # warnings.warn('Defaulting to in-operator encoding')
                pred, alph = classify_in_segment(left_element, right_element)
            elif operator == 'in':
                pred, alph = classify_in_segment(left_element, right_element)
            elif operator == 'not':
                # print(left_element, right_element)
                raise NotImplementedError('Predicates not yet supported')
            else:
                raise NotImplementedError('Predicates not yet supported')
            predicates = predicates.union(pred)
            alphabet = u.merge_dicts(alphabet, alph)
    return predicates, alphabet


def look_ahead_pass(path, queries, db_string):

    predicates = set()
    alphabet = dict()
    cardinalities = dict()  # testing
    if not os.path.exists('db_type_dict.json'):
        d_type_dict = u.build_db_type_dict(db_string)
        u.save_json(d_type_dict, "db_type_dict.json")
    type_dict = u.load_json('db_type_dict.json')

    for query in queries:
        # predicates
        # alphabet
        print(query)
        parsed_query = u.parse_query(path, query)
        where = parsed_query['where']
        from_p = parsed_query['from']
        alias_to_table = get_alias_to_table(from_p)
        # print(alias_to_table)
        # where can be conditioned with and, or, not
        where_dict = get_where_segments(where)

        cardinality_dict = dict()  # testing

        # present keys
        for key in where_dict.keys():
            # and, or, not
            segment = where_dict[key]

            # get dict sorted by operators
            pred, alph = get_segment_info(segment, alias_to_table, type_dict)

            predicates = predicates.union(pred)
            alphabet = u.merge_dicts(alphabet, alph)
            cardinality_dict = u.merge_dicts(cardinality_dict, alph)  # testing

        # print('Cardinality dict: ', cardinality_dict)

        conn, cursor = u.establish_connection(db_string)
        with open(path + query, 'r', encoding='utf-8') as f:
            q = f.read()
        cursor.execute("EXPLAIN ANALYZE " + q)
        explain_result = cursor.fetchall()
        # testing
        for key in cardinality_dict.keys():
            filter_elements = cardinality_dict[key]
            for filter_element in filter_elements:
                filter_line = first_join_from_res(explain_result, filter_element)
                row_pattern = r'rows=\d+'
                rows = re.findall(row_pattern, filter_line)
                rows = int(rows[1].split('=')[1])
                cardinalities = u.add_or_create_dict_entry(cardinalities, key, tuple([filter_element, rows]))

        cursor.close()
        conn.close()

        # due to alias ambiguity, min max encoding should be done when encoding

    print('Final predicates: ', predicates)
    print('Final alphabet: ', alphabet)
    print('Final cardinalities: ', cardinalities)  # testing

    final_len = 0
    t_set = set()
    for key in alphabet.keys():
        final_len += len(alphabet[key])
        t_set = t_set.union(alphabet[key])
    print('Final alphabet size: ', final_len)
    print('Final test set size: ', len(t_set))
    return predicates, alphabet, cardinalities


def encode_single_segment(left_segment, right_segment, mm_dict, from_p, encoders, verbose=False):
    """
    delineated from
    https://github.com/lucaswo/cardest/blob/master/query_processing.py
    """
    if left_segment in encoders.keys():
        # print(left_segment, right_segment)
        if verbose:
            print('Left segment: ', left_segment)
            print('Right segment: ', right_segment[0])
        right_segment = right_segment[0]
        if isinstance(right_segment, dict):
            if 'cast' in right_segment.keys():
                right_segment = right_segment['cast'][0]
                value = [right_segment['literal']]
            elif 'literal' in right_segment.keys():
                value = [right_segment['literal']]
            else:
                raise NotImplementedError('Unknown Cast')
        else:
            raise NotImplementedError('Unknown right segment')

        try:
            value = encoders[left_segment].transform(value)[0]
        except:
            raise ValueError('Could not encode single segment query')
            # TODO: new handle for unseen queries, pick random around encoded mean
            print('Caught unseen element in single segment label encoder...')
            value = np.round(np.random.uniform(0.4, 0.6, 1)[0], 2)
        min_v = 0.0
        max_v = float(len(encoders[left_segment].classes_))
        step = 1.0
        if verbose:
            print('Transformed: ', value)
            print('Min Max: ', min_v, max_v)
    else:
        if u.is_float(right_segment[0]):
            value = right_segment[0]
        elif isinstance(right_segment[0], dict):
            value = right_segment[0]['literal']
        else:
            raise NotImplementedError('Unknown float')

        if isinstance(value, int):
            step = 1.0
        else:
            step = 1 / 1000
        alias, column = left_segment.split('.')
        table = look_up_alias(alias, from_p)
        min_v, max_v = mm_dict[table][column]
        if verbose:
            print('Min Max ', min_v, max_v)
            print('Value: ', value)
        value = min(value, float(max_v))
        value = max(value, float(min_v))
    return value, min_v, max_v, step


def encode_compound_segment(left_segment, right_segment, encoders, verbose=False):
    """
    delineated from
    https://github.com/lucaswo/cardest/blob/master/query_processing.py
    """
    if verbose:
        print('Left segment: ', left_segment)
        print('Right segment: ', right_segment[0]['literal'])
    value = right_segment[0]['literal']
    if not isinstance(value, list):
        value = [value]
    try:
        values = encoders[left_segment].transform(value)
    except:
        raise ValueError('Could not encode compound segment query')
        # TODO: new handle for unseen queries, pick random around encoded mean
        print('Caught unseen element in compound segment label encoder...')
        values = np.round(np.random.uniform(0.4, 0.6, len(value)), 2)

    value = sum(values)
    value /= len(values)
    if verbose:
        print('Value: {} = {} / {}'.format(value, sum(values), len(values)))

    min_v = 0.0
    max_v = float(len(encoders[left_segment].classes_))
    step = 1.0
    if verbose:
        print('Transformed: ', value)
        print('Min Max: ', min_v, max_v)
    return value, min_v, max_v, step


def encode_query(parsed_query, predicates, encoders, mm_dict, verbose=False):
    total_columns = len(predicates)
    encoded_query = np.zeros(total_columns*4)

    where = parsed_query['where']
    from_p = parsed_query['from']

    where_dict = get_where_segments(where)
    # present keys
    for key in where_dict.keys():
        # and, or, not, mostly just and
        segment = where_dict[key]
        for predicate_line in segment:
            # print('Predicate Line: ', predicate_line)
            for operator in predicate_line.keys():
                if operator == 'exists':
                    enc_t = encode_query(predicate_line['exists'], predicates, encoders, mm_dict)
                    for i in range(int(len(enc_t)/4)):
                        if sum(enc_t[i:i+3]) != 0:
                            print('Swapping {} for {}'.format(encoded_query[i:i+3], enc_t[i:i+3]))
                            encoded_query[i:i+3] = enc_t[i:i+3]
                    continue
                elif operator == 'not':
                    enc_t = encode_query(predicate_line['not']['exists'], predicates, encoders, mm_dict)
                    for i in range(int(len(enc_t)/4)):
                        if sum(enc_t[i:i+3]) != 0:
                            print('Swapping {} for {}'.format(encoded_query[i:i + 3], enc_t[i:i + 3]))
                            encoded_query[i:i+3] = enc_t[i:i+3]
                    continue
                elif operator not in u.operator_dictionary:
                    raise ValueError('Operator "{}" not classifiable'.format(operator))

                operator_line = predicate_line[operator]
                left_element = operator_line[0]
                right_element = operator_line[1:]

                if isinstance(left_element, dict):
                    if isinstance(right_element[0], dict):
                        left_element = left_element['lower']
                        right_element = [right_element[0]['lower']]
                    else:
                        raise ValueError('predicate is dictionary but filter value not')

                if left_element not in predicates:
                    continue

                single_segments = ['eq', 'IS', 'gte', 'lte', 'lt', 'gt', 'neq']
                compound_segments = ['like', 'in']
                # print('Encoding this line: ', predicate_line)
                if operator in single_segments:
                    value, min_v, max_v, step = encode_single_segment(left_element, right_element, mm_dict, from_p,
                                                                      encoders, verbose)
                    # print('Encoded single line: ')
                elif operator in compound_segments:
                    value, min_v, max_v, step = encode_compound_segment(left_element, right_element, encoders, verbose)
                    # print('Encoded compound line: ')
                elif operator == 'not':
                    # print(left_element, right_element)
                    raise NotImplementedError('Predicates not yet supported')
                else:
                    raise NotImplementedError('Predicates not yet supported')

                idx = sorted(predicates).index(left_element)
                encoded_query[idx * 4:idx * 4 + 3] = u.operator_dictionary[operator]
                encoded_query[idx * 4 + 3] = (value - min_v + step) / (max_v - min_v + step)
                # print(encoded_query)

    return encoded_query


def encode_queries(path, queries, predicates, encoders, verbose=True):

    mm_dict = u.load_json("mm_dict.json")
    encoded_queries = list()
    for query in queries:
        parsed_query = u.parse_query(path, query)
        encoded_query = encode_query(parsed_query, predicates, encoders, mm_dict, verbose)
        # print('Encoded Query {}: {}'.format(query, encoded_query))
        encoded_queries.append(encoded_query)

    return np.array(encoded_queries)


def sort_alphabet_by_card(alphabet, card_list):
    sorted_list = list(card_list)
    sorted_list.sort(key=lambda x: x[1])
    print('Alphabet: ', alphabet)
    print('Sorted List: ', sorted_list)
    return sorted_list


def load_or_build_baseline(path, queries, result_path, db_string, overwrite=False):
    if not os.path.exists(result_path+'predicates.json') or \
            not os.path.exists(result_path+'encoders.pkl') or overwrite:
        print('Predicates and Encoders not found. Building...')
        # set of predicates to encode, dictionary with strings for each column
        predicates, alphabet, cardinalities = look_ahead_pass(path, queries, db_string)
        print('...finished.')
        encoders = dict()
        # build label encoders for each column
        for key in alphabet.keys():
            sub_card = cardinalities[key]
            card_sorted_alphabet = sort_alphabet_by_card(alphabet[key], sub_card)  # testing
            card_sorted_alphabet = np.array(card_sorted_alphabet)
            card_sorted_alphabet = np.transpose(card_sorted_alphabet)[0]
            print('Cardinality sorted alphabet: ', tuple(card_sorted_alphabet))
            alphabet[key] = tuple(alphabet[key])
            enc = LabelEncoder()
            # encoders[key] = enc.fit(sorted(alphabet[key]))
            encoders[key] = enc.fit(tuple(card_sorted_alphabet))  # testing
        print('Saving Predicates, Encoders and Alphabet')
        u.save_json(tuple(predicates), result_path+'predicates.json')
        u.save_json(alphabet, result_path+'alphabet.json')
        u.save_pickle(encoders, result_path+'encoders.pkl')
        print('Saved')
    print('Loading Predicates and Encoders...')
    predicates = u.load_json(result_path+'predicates.json')
    encoders = u.load_pickle(result_path+'encoders.pkl')
    print('Loaded')
    return predicates, encoders


def get_compound_cardinality(result, right):
    # TODO: Still highly specific, adapt

    search_list = right['literal']
    search_string = '{'
    for i in range(len(search_list)):
        element = search_list[i]
        if element == 'null':
            search_string += '"{}"'.format(search_list[i])
        else:
            search_string += '{}'.format(search_list[i])
        if i != len(search_list)-1:
            search_string += ','
    search_string += '}'
    # print('Search string: ', search_string)

    filter_line = first_join_from_res(result, search_string)
    row_pattern = r'rows=\d+'
    rows = re.findall(row_pattern, filter_line)
    rows = int(rows[1].split('=')[1])
    # print('Found cardinality of {} for first filter'.format(rows))
    return rows


def first_join_from_res(result, search_string):
    join_keywords = ['Nested Loop', 'Nested Loop Semi Join', 'Hash Join', 'Merge Join']
    for i in range(len(result)):
        result_line = result[i][0]
        search_string = str(search_string)
        # print('Result Line: ', result_line)
        # print('Search string: ', str(search_string))
        # print('Condition found: ', 'Cond' in result_line)
        if search_string in result_line and ('Filter' in result_line or 'Cond' in result_line):
            # print('Found  Line: ', result_line)
            result_depth = result_line.count(' ')
            for j in range(i-1):
                previous_line = result[i-j-1][0]
                prev_depth = previous_line.count(' ')
                # print('Previous Line: ', previous_line)
                if prev_depth >= result_depth:
                    continue
                else:
                    for keyword in join_keywords:
                        if keyword in previous_line:
                            # print('Found first join location: ', previous_line)
                            return previous_line
                        else:
                            continue
        else:
            continue
    raise NotImplementedError('No join found')


def get_cardinality(result, right):
    # TODO: Still highly specific, adapt!

    if u.is_float(right):
        search_string = str(right)
    else:
        search_string = right['literal']

    filter_line = first_join_from_res(result, search_string)
    row_pattern = r'rows=\d+'
    rows = re.findall(row_pattern, filter_line)
    rows = int(rows[1].split('=')[1])
    print('Found cardinality of {} for first filter'.format(rows))
    return rows


def run_test():
    context_path = 'context_3/'
    query_path = 'queries/' + context_path
    result_path = 'results/' + context_path
    queries = u.get_queries(query_path)
    print(u.parse_query(query_path, queries[0]))

    predicates, encoders = load_or_build_baseline(query_path, queries, result_path)

    encoded_queries = encode_queries(query_path, queries, predicates, encoders)
    # build training featurized data
    columns = ['query']
    # fancy column names
    for i in range(len(predicates)):
        columns += ['op0_{}'.format(i), 'op1_{}'.format(i), 'op2_{}'.format(i), 'val_{}'.format(i)]
    # append query name for identification
    pd_queries = np.array([np.append(queries[i], encoded_queries[i]) for i in range(len(queries))])
    df = pd.DataFrame(pd_queries, columns=columns)

    feature_path = result_path+'featurization_2.csv'
    # no overwriting
    if not os.path.exists(feature_path):
        df.to_csv(feature_path, index=False)
    else:
        print('Feature path already exists, skipping saving...')

    print('Done')
    return
