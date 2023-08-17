from caidin.models.recommendation_engine import RecommendationEngine


class CollaborativeFiltering(RecommendationEngine):
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

    def train(self):
        # Calculate similarity matrix based on ratings
        self.similarity_matrix = {}
        for item1 in self.items:
            self.similarity_matrix[item1.item_id] = {}
            for item2 in self.items:
                similarity = self.calculate_similarity(item1, item2)
                self.similarity_matrix[item1.item_id][item2.item_id] = similarity

    def get(self, user, num_recommendations=5):
        recommendations = self.recommend(user, num_recommendations)
        return recommendations
