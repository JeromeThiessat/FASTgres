import os
import unittest

from fastgres.baseline.database_connection import DatabaseConnection
from fastgres.baseline.hint_sets import HintSet
from fastgres.baseline.public_api import Fastgres
from fastgres.workloads.workload import Workload
from fastgres.definitions import ROOT_DIR


class TestApiBehaviour(unittest.TestCase):

    def test_case_dbc_false(self):
        dbc_stack = DatabaseConnection(
            "dbname=stack_overflow_reduced_16 user=postgres password=postgres host=localhost port=5432",
            "stack_db")
        self.assertRaises(Exception, dbc_stack.establish_connection)

    def test_case_dbc_true(self):
        raised = False
        try:
            dbc_stack = DatabaseConnection(
                "dbname=stack_overflow user=postgres password=postgres host=localhost port=5432",
                "stack_db")
            dbc_stack.establish_connection()
        except:
            raised = True
        self.assertEqual(raised, False)

    def test_case_prediction(self):
        dbc_stack = DatabaseConnection(
            "dbname=stack_overflow user=postgres password=postgres host=localhost port=5432",
            "stack_db")
        workload_stack = Workload(ROOT_DIR + "/workloads/queries/stack/", "stack")
        fastgres_stack = Fastgres(workload_stack, dbc_stack)
        stack_queries = workload_stack.get_queries()
        predictions = [fastgres_stack.predict(q) for q in stack_queries[:10]]
        for h_set in predictions:
            self.assertIsInstance(h_set, HintSet)
        self.assertEqual(len(predictions), 10)
        int_predictions = [hint_set.get_int_name() for hint_set in predictions]
        for int_pred in int_predictions:
            self.assertIsInstance(int_pred, int)

    def test_case_run(self):
        exception = False
        try:
            workload_stack = Workload(ROOT_DIR + "/workloads/queries/stack/", "stack")
            dbc_stack = DatabaseConnection(
                "dbname=stack_overflow user=postgres password=postgres host=localhost port=5432",
                "stack_db")
            fastgres_stack = Fastgres(workload_stack, dbc_stack)
            stack_queries = workload_stack.get_queries()
            predictions = [fastgres_stack.predict(q) for q in stack_queries[:1]]
        except:
            exception = True
        self.assertEqual(exception, False)


if __name__ == '__main__':
    unittest.main()
