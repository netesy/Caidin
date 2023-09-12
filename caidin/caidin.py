# caidin.py
from caidin.algorithms.content_based import ContentBased
from caidin.algorithms.collaborative_filtering import CollaborativeFiltering


class Caidin:
    def __init__(self):
        self.data = None
        self.method = None
        self.records = None
        self.filters = {
            "content_based": ContentBased,  # Default content-based recommendation method
            "collaborative_filtering": CollaborativeFiltering,  # Default collaborative filtering method
        }

        # Dictionary to hold registered recommendation methods
        self.method_registry = {}

    def register_method(self, method_name, method_class):
        self.method_registry[method_name] = method_class

    def load(self, data: list | dict | str) -> Caidin:
        self.data = data
        return self

    # def using(self, method_name):
    #     if method_name not in self.method_registry:
    #         raise ValueError(f"Unsupported recommendation method: {method_name}")
    #     self.method = self.method_registry[method_name](
    #         self.data, self.records, self.filters
    #     )
    #     return self

    def using(self, method_name: str = None) -> Caidin:
        if method_name:
            # Check if it's a user-defined method
            if method_name in self.default_methods:
                self.method = self.default_methods[method_name](
                    self.data, self.records, self.filters
                )
            else:
                raise ValueError(f"Unsupported recommendation method: {method_name}")
        else:
            raise ValueError(
                "Please specify a recommendation method using the 'using' method."
            )
        return self

    def train(self, records: list | dict | str) -> Caidin:
        self.records = records
        return self

    def where(self, category: str, value):
        self.filters[category] = value
        return self

    def whereNot(self, category: str, value):
        self.filters[category] = {"$ne": value}
        return self

    def get(self):
        if self.method is None:
            raise ValueError(
                "Please specify a recommendation method using the 'using' method."
            )
        return self.method.recommend()
