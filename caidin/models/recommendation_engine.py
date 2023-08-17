from caidin.models.item import Item


class RecommendationEngine:
    def __init__(self):
        self.items = []
        self.filter_conditions = []

    def add_item(self, item_id, attributes):
        item = Item(item_id, attributes)
        self.items.append(item)

    def calculate_similarity(self, item1, item2):
        dot_product = sum(a * b for a, b in zip(item1.attributes, item2.attributes))
        norm_item1 = sum(a**2 for a in item1.attributes) ** 0.5
        norm_item2 = sum(b**2 for b in item2.attributes) ** 0.5
        if norm_item1 == 0 or norm_item2 == 0:
            return 0  # Avoid division by zero
        return dot_product / (norm_item1 * norm_item2)

    def where(self, condition):
        self.filter_conditions.append(condition)
        return self

    def get_filtered_items(self):
        filtered_items = self.items.copy()
        for condition in self.filter_conditions:
            filtered_items = [item for item in filtered_items if condition(item)]
        return filtered_items

    def recommend(self, item_id, num_recommendations=5):
        item_to_recommend = next(
            (item for item in self.items if item.item_id == item_id), None
        )
        if item_to_recommend is None:
            return []

        filtered_items = self.get_filtered_items()

        similarities = []
        for item in filtered_items:
            if item.item_id != item_id:
                similarity = self.calculate_similarity(item_to_recommend, item)
                similarities.append((item.item_id, similarity))

        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities
