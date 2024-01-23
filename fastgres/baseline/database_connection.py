import math
import time

import psycopg2 as pg

from baseline.hint_sets import HintSet, set_hints


class DatabaseConnection:

    def __init__(self, psycopg_connection_string: str, name: str):
        self.connection_string = psycopg_connection_string
        self.name = name

    def __str__(self):
        return f"Database Connection: {self.name}"

    def establish_connection(self):
        try:
            connection = pg.connect(self.connection_string)
            # https://www.psycopg.org/psycopg3/docs/basic/transactions.html#transactions
            connection.autocommit = True
        except ConnectionError:
            raise ConnectionError('Could not connect to database server')
        cursor = connection.cursor()
        return connection, cursor

    def evaluate_hinted_query(self, query: str, hint_set: HintSet, timeout: float):

        conn, cur = self.establish_connection()
        if timeout is not None:
            if timeout <= 0.0:
                print('Adjusting timeout from {}'.format(timeout))
                timeout = 0.1
            time_out = "SET statement_timeout = '{}ms'".format(int(math.ceil(timeout * 1000)))
            cur.execute(time_out)

        set_hints(hint_set, cur)
        try:
            start = time.time()
            cur.execute(query)
            stop = time.time()
        except:
            return None

        cur.close()
        conn.close()
        return stop - start
