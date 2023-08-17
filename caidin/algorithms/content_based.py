from caidin.models.recommendation_engine import RecommendationEngine


class ContentBased(RecommendationEngine):
    def __init__(self):
        super().__init__()

    # def items(self):
    #     return self

    def load(self, data):
        for item, item_features in data.items():
            self.add_item(item, item_features)

        return self

    def train(self):
        # Calculate feature similarity matrix
        self.similarity_matrix = {}
        for item1 in self.items:
            self.similarity_matrix[item1.item_id] = {}
            for item2 in self.items:
                similarity = self.calculate_similarity(item1, item2)
                self.similarity_matrix[item1.item_id][item2.item_id] = similarity

    def get(self, item, num_recommendations=5):
        recommendations = self.recommend(item, num_recommendations)
        return recommendations
