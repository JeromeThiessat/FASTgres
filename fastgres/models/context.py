
class Context:

    def __init__(self, init_context: frozenset = None, name: str = "default_context_name"):
        self.covered_contexts = set()
        self.is_merged = False
        self.table_sets = 0
        self.total_tables = frozenset()
        self.name = name
        if init_context:
            self.add_context(init_context)

    def __str__(self):
        return_string = f"Context {self.name} covering {self.table_sets} table sets\n"
        for _ in self.covered_contexts:
            return_string += str(_) + "\n"
        return_string += f"Encompassing {len(self.total_tables)} tables"
        return return_string

    def __eq__(self, other):
        if not isinstance(other, Context):
            raise ValueError("Compared value is not of class Context.")
        # Third comparison might be redundant
        return all([_ in self.covered_contexts for _ in other.covered_contexts])\
            and self.table_sets == other.table_sets \
            and self.total_tables == other.total_tables

    def add_context(self, context: frozenset):
        self.covered_contexts.add(context)
        self.total_tables = self.total_tables.union(context)
        self.table_sets += 1
        if self.table_sets != 1:
            self.is_merged = True

    def get_context_histogram(self):
        context_dictionary = dict()
        for sub_context in self.covered_contexts:
            context_size = len(sub_context)
            if context_size in context_dictionary:
                context_dictionary[context_size] += 1
            else:
                context_dictionary[context_size] = 1
        return context_dictionary
