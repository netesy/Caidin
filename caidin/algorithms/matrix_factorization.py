import random
from caidin.models.recommendation_engine import RecommendationEngine


class MatrixFactorization(RecommendationEngine):
    def __init__(self):
        super().__init__()

    # def users(self):
    #     return self

    # def items(self):
    #     return self

    def load(self, data):
        for user, item_ratings in data.items():
            self.add_item(user, item_ratings)

        return self

    def train(self, num_factors=10, num_iterations=100, learning_rate=0.01):
        num_users = len(self.items)
        num_items = len(self.items[0].attributes)

        user_item_matrix = [[item.attributes for item in self.items]]

        self.user_factors = [
            [0.5 + 0.5 * random.random() for _ in range(num_factors)]
            for _ in range(num_users)
        ]
        self.item_factors = [
            [0.5 + 0.5 * random.random() for _ in range(num_factors)]
            for _ in range(num_items)
        ]

        for _ in range(num_iterations):
            for user_idx in range(num_users):
                for item_idx in range(num_items):
                    # TODO: fix the isssue that checks if a list > int
                    if user_item_matrix[user_idx][item_idx] > [0]:
                        predicted_rating = 0
                        for k in range(num_factors):
                            predicted_rating += (
                                self.user_factors[user_idx][k]
                                * self.item_factors[item_idx][k]
                            )

                        error = user_item_matrix[user_idx][item_idx] - predicted_rating

                        for k in range(num_factors):
                            self.user_factors[user_idx][k] += learning_rate * (
                                2 * error * self.item_factors[item_idx][k]
                            )
                            self.item_factors[item_idx][k] += learning_rate * (
                                2 * error * self.user_factors[user_idx][k]
                            )

    def get(self, user, num_recommendations=5):
        recommendations = self.recommend(user, num_recommendations)
        return recommendations
