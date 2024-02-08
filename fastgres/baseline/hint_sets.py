import enum
import warnings

import numpy as np


# def set_hints(hint_set, cursor):
#     for i in range(hint_set.hint_set_size):
#         name = hint_set.get_name(i)
#         value = hint_set.get(i)
#         # statement = f'set {name}={value};'
#         cursor.execute(f'set {name}={value};')
#     return hint_set


# def show_hint_status(cursor):
#     operators = HintSet.operators
#     for i in operators:
#         # statement = 'show {};'.format(i)
#         cursor.execute('show {};'.format(i))
#         res = cursor.fetchall()[0][0]
#         print('{} is set to "{}"'.format(i, res))
#     print('\n')
#     return


# def reset_hints(cursor):
#     reset_operators = HintSet.operators
#     for operator in reset_operators:
#         # statement = 'set ' + operator + '=true;'
#         cursor.execute(f"set {operator}=true;")
#     return


# class Hint(enum.Enum):
#     HASH_JOIN = 32
#     MERGE_JOIN = 16
#     NESTED_LOOP_JOIN = 8
#     INDEX_SCAN = 4
#     SEQ_SCAN = 2
#     INDEX_ONLY_SCAN = 1
#
#
# class PostgresOperator(enum.Enum):
#     HASH_JOIN = "enable_hashjoin"
#     MERGE_JOIN = "enable_mergejoin"
#     NESTED_LOOP_JOIN = "enable_nestloop"
#     INDEX_SCAN = "enable_indexscan"
#     SEQ_SCAN = "enable_seqscan"
#     INDEX_ONLY_SCAN = "enable_indexonlyscan"

pg_hints = [("ASYNC_APPEND", "enable_async_append"), ("BITMAP_SCAN", "enable_bitmapscan"),
            ("GATHER_MERGE", "enable_gathermerge"), ("HASH_AGG", "enable_hashagg"),
            ("INC_SORT", "enable_incremental_sort"), ("MATERIALIZATION", "enable_material"),
            ("MEMOIZE", "enable_memoize"), ("PARA_APPEND", "enable_parallel_append"),
            ("PARA_HASH", "enable_parallel_hash"), ("PART_PRUNING", "enable_partition_pruning"),
            ("PART_JOIN", "enable_partitionwise_join"), ("PART_AGG", "enable_partitionwise_aggregate"),
            ("PRESORT_AGG", "enable_presorted_aggregate"), ("SORT", "enable_sort"),
            ("TID_SCAN", "enable_tidscan "), ("HASH_JOIN", "enable_hashjoin"),
            ("MERGE_JOIN", "enable_mergejoin"), ("NESTED_LOOP_JOIN", "enable_nestloop"),
            ("INDEX_SCAN", "enable_indexscan"), ("SEQ_SCAN", "enable_seqscan"),
            ("INDEX_ONLY_SCAN", "enable_indexonlyscan")]
hints = [(list(reversed(pg_hints))[i][0], 2 ** i) for i in range(len(pg_hints))]

PostgresOperator = enum.Enum("PostgresOperator", pg_hints)
Hint = enum.Enum("Hint", hints)


class HintSet:
    default_operators = ['enable_hashjoin', 'enable_mergejoin', 'enable_nestloop', 'enable_indexscan', 'enable_seqscan',
                         'enable_indexonlyscan']

    all_operators = [operator.value for operator in PostgresOperator]

    def __init__(self, default: int = None, operators: list[str] = None):
        if operators is None:
            operators = self.default_operators
        self.operators = operators
        [self.__setattr__(PostgresOperator(op).name, True) for op in self.operators]
        # self.hash_join = True
        # self.merge_join = True
        # self.nested_join = True
        # self.index_scan = True
        # self.seq_scan = True
        # self.index_only_scan = True
        self.hint_set_size = len(self.operators)

        if default is not None:
            if not isinstance(default, int):
                raise ValueError('Input hint set is not of type int')
            self.set_hint_from_int(default)

        return

    def __str__(self):
        return f"Hint Set: {self.get_int()} : {self.get_binary()}"

    def set_hint_from_int(self, hint_int):
        binary_list = [int(i) for i in bin(hint_int)[2:].zfill(self.hint_set_size)]
        self.set_from_int_list(binary_list)
        return

    def set_hint_i(self, i: int, value: bool):
        if value not in [True, False]:
            raise ValueError('Trying to set hint set from non boolean')
        if i not in range(len(self.operators)):
            raise ValueError(f'Index {i} is out of bounds for {len(self.operators)} operators')
        self.__setattr__(PostgresOperator(self.operators[i]).name, value)
        # if value not in [True, False]:
        #     raise ValueError('Trying to set hint set from non boolean')
        # else:
        #     if i == 0:
        #         self.hash_join = value
        #     elif i == 1:
        #         self.merge_join = value
        #     elif i == 2:
        #         self.nested_join = value
        #     elif i == 3:
        #         self.index_scan = value
        #     elif i == 4:
        #         self.seq_scan = value
        #     elif i == 5:
        #         self.index_only_scan = value
        #     else:
        #         raise ValueError('Invalid Index')
        return

    def print_info(self):
        [print(f"{PostgresOperator[str(self.operators[i])]}: {self.get(i)}") for i in range(len(self.operators))]
        # print('hash join:', self.hash_join)
        # print('merge join:', self.merge_join)
        # print('nested loop join:', self.nested_join)
        # print('index scan:', self.index_scan)
        # print('sequential scan:', self.seq_scan)
        # print('index only scan:', self.index_only_scan, '\n')
        return

    def get(self, i):
        if i not in range(len(self.operators)):
            raise ValueError(f'Index {i} is out of bounds for {len(self.operators)} operators')
        return self.__getattribute__(PostgresOperator(self.operators[i]).name)
        # if i == 0:
        #     return self.hash_join
        # elif i == 1:
        #     return self.merge_join
        # elif i == 2:
        #     return self.nested_join
        # elif i == 3:
        #     return self.index_scan
        # elif i == 4:
        #     return self.seq_scan
        # elif i == 5:
        #     return self.index_only_scan
        # else:
        #     raise ValueError('Hint index out of bounds')

    def get_name(self, i):
        if i not in range(len(self.operators)):
            raise ValueError(f'Index {i} is out of bounds for {len(self.operators)} operators')
        return self.operators[i]
        # if i == 0:
        #     return 'enable_hashjoin'
        # elif i == 1:
        #     return 'enable_mergejoin'
        # elif i == 2:
        #     return 'enable_nestloop'
        # elif i == 3:
        #     return 'enable_indexscan'
        # elif i == 4:
        #     return 'enable_seqscan'
        # elif i == 5:
        #     return 'enable_indexonlyscan'
        # else:
        #     raise ValueError('Hint index out of bounds')

    def set_hints_boolean(self, boolean_list: list[bool]):
        if not isinstance(boolean_list, list):
            raise ValueError('No list provided for setting boolean hints')
        if len(boolean_list) != len(self.operators):
            raise ValueError(f'Boolean list length {len(boolean_list)} not supported '
                             f'for {len(self.operators)} operators.')

        hint_set_int = list(np.array(boolean_list).astype(int))
        self.set_from_int_list(hint_set_int)
        # for index in range(len(boolean_list)):
        #     index_element = boolean_list[index]
        #     if boolean_list[index] not in [True, False]:
        #         raise ValueError('Boolean hint list contains non boolean values')
        #     self.set_hint_i(index, index_element)
        return

    def set_from_int_list(self, int_list: list[int]):
        for i in range(len(int_list)):
            integer = int_list[i]
            if integer not in [0, 1]:
                raise ValueError('Setting Hint Set with values other than 0 or 1')
            self.set_hint_i(i, bool(integer))
        return

    def get_binary(self):
        binary = [int(self.get(i)) for i in range(self.hint_set_size)]
        return binary

    def get_int(self):
        bin_list = self.get_binary()
        return int("".join(str(i) for i in bin_list), 2)
