import math
import time

import psycopg2
import psycopg2 as pg

from fastgres.baseline.hint_sets import HintSet, set_hints
from fastgres.baseline.log_utils import get_logger


class DatabaseConnection:

    def __init__(self, psycopg_connection_string: str, name: str, use_pg_hint_plan: bool = False):
        self.connection_string = psycopg_connection_string
        self.name = name
        self._connection = None
        self._cursor = None
        self._extension_loaded = False
        self.load_extension = use_pg_hint_plan

    def __str__(self):
        return f"Database Connection: {self.name}"

    @property
    def connection(self):
        if self._connection is None:
            self._connection = self.establish_connection()
        return self._connection

    @property
    def cursor(self):
        if self._cursor is None:
            if self._connection is None:
                _ = self.connection
            self._cursor = self._connection.cursor()
        return self._cursor

    def close_connection(self):
        if self._cursor is not None:
            self._cursor.close()
            self._cursor = None
        if self._connection is not None:
            self._connection.close()
            self._connection = None
            self._extension_loaded = False

    def establish_connection(self):
        try:
            connection = pg.connect(self.connection_string)
            # https://www.psycopg.org/psycopg3/docs/basic/transactions.html#transactions
            connection.autocommit = True
        except ConnectionError:
            raise ConnectionError('Could not connect to database server')
        self._connection = connection
        self.enable_pg_hint_plan()
        return connection

    def version(self):
        cursor = self.cursor
        cursor.execute("SELECT version();")
        return cursor.fetchall()[0][0]

    def enable_pg_hint_plan(self):
        if not self._extension_loaded and self.load_extension:
            cursor = self.cursor
            cursor.execute("LOAD 'pg_hint_plan';")
            logger = get_logger()
            logger.info("Loaded pg_hint_plan")
        self._extension_loaded = True

    @staticmethod
    def _get_hint_statements(hint_set: HintSet):
        statement = ""
        for i in range(hint_set.hint_set_size):
            name = hint_set.get_name(i)
            value = hint_set.get(i)
            statement += f"set {name}={value};\n"
        return statement

    def _build_pre_statement(self, hint_set: HintSet, timeout: float):
        statement = ""
        statement += self._get_hint_statements(hint_set)
        if timeout is not None:
            if timeout <= 0.0:
                timeout = 0.01
            statement += f"SET LOCAL statement_timeout = '{round(timeout, 4)}s';\n"
        return statement

    def evaluate_hinted_query(self, query: str, hint_set: HintSet, timeout: float = None):
        cursor = self.cursor
        statement = self._build_pre_statement(hint_set, timeout)
        statement += query
        try:
            start = time.time()
            cursor.execute(statement)
            stop = time.time()
        except psycopg2.DatabaseError:
            return None
        return stop - start
