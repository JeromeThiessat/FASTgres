
import unittest
from models.context import Context


class TestContextValidity(unittest.TestCase):

    def test_case_length_1(self):
        c1 = frozenset({"1", "2", "chicken nuggets"})
        c2 = frozenset({"2", "3", "sandwiches"})

        new_context = Context()
        new_context.add_context(c1)
        new_context.add_context(c2)

        self.assertEqual(new_context.table_sets, 2)

    def test_case_length_2(self):
        c1 = frozenset({"1", "2", "chicken nuggets"})
        c2 = frozenset({"1", "2", "sandwiches"})

        new_context = Context()
        new_context.add_context(c1)
        new_context.add_context(c2)

        self.assertEqual(len(new_context.total_tables), 4)

    def test_case_length_3(self):
        c1 = frozenset({"1", "2", "chicken nuggets"})
        c2 = frozenset({"1", "2", "sandwiches"})

        new_context = Context()
        new_context.add_context(c1)
        new_context.add_context(c2)

        self.assertEqual(new_context.table_sets, 2)

    def test_case_equality_1(self):
        c1 = frozenset({"1", "2", "chicken nuggets"})
        c2 = frozenset({"2", "3", "sandwiches"})

        new_context = Context()
        new_context.add_context(c1)
        new_context.add_context(c2)

        other_context = Context()
        other_context.add_context(c1)
        other_context.add_context(c2)

        self.assertEqual(new_context, other_context)

    def test_case_equality_2(self):
        c1 = frozenset({"1", "2", "chicken nuggets"})
        c2 = frozenset({"2", "3", "sandwiches"})
        # c3 = frozenset({"3", "4", "hotdogs"})
        c4 = frozenset({"2", "3", "sand"})

        new_context = Context()
        new_context.add_context(c1)
        new_context.add_context(c2)

        other_context = Context()
        other_context.add_context(c1)
        other_context.add_context(c4)

        self.assertNotEqual(new_context, other_context)


if __name__ == '__main__':
    unittest.main()
