import math
from collections import defaultdict
from caidin.algorithms.recommendation_engine import RecommendationEngine


class CollaborativeFiltering(RecommendationEngine):
    def __init__(self, data, records, filters):
        self.data = data
        self.records = records
        self.filters = filters
        self.user_field = None
        self.item_field = None
        self.rating_field = None
        self.user_item_matrix = None
        self.user_similarity_matrix = None

    def recommend(self):
        # Check if user_field, item_field, and rating_field are specified
        if not all(
            field in self.filters
            for field in ["user_field", "item_field", "rating_field"]
        ):
            raise ValueError(
                "Please specify 'user_field', 'item_field', and 'rating_field' using the 'where' method."
            )

        # Filter records based on user-defined criteria
        filtered_records = self.records
        for category, value in self.filters.items():
            if category in filtered_records and filtered_records[category] == value:
                continue
            else:
                return []  # No recommendations if filters don't match

        # Build user-item matrix and user similarity matrix
        self.build_user_item_matrix()

        if not self.user_field or not self.item_field or not self.rating_field:
            raise ValueError(
                "Please specify 'user_field', 'item_field', and 'rating_field' using the 'where' method."
            )

        self.build_user_similarity_matrix()

        # Get the index of the user that matches the given user_id
        user_idx = self.data[
            self.data[self.user_field] == self.filters["user_id"]
        ].index[0]

        # Calculate recommendations based on user similarities
        recommendations = self.calculate_recommendations(user_idx)

        return recommendations

    def where(self, user_field=None, item_field=None, rating_field=None, **kwargs):
        # Set user_field, item_field, and rating_field using the where method
        if user_field:
            self.user_field = user_field
        if item_field:
            self.item_field = item_field
        if rating_field:
            self.rating_field = rating_field
        return self

    def build_user_item_matrix(self):
        # Build user-item matrix from data
        user_item_matrix = {}

        for _, row in self.records.iterrows():
            user = row[self.user_field]
            item = row[self.item_field]
            rating = row[self.rating_field]

            if user not in user_item_matrix:
                user_item_matrix[user] = {}

            user_item_matrix[user][item] = rating

        self.user_item_matrix = user_item_matrix

    def build_user_similarity_matrix(self):
        # Build user similarity matrix using Pearson correlation coefficient
        user_similarity_matrix = {}

        for user1 in self.user_item_matrix.keys():
            user_similarity_matrix[user1] = {}
            for user2 in self.user_item_matrix.keys():
                if user1 == user2:
                    continue

                similarity = self.calculate_user_similarity(user1, user2)
                user_similarity_matrix[user1][user2] = similarity

        self.user_similarity_matrix = user_similarity_matrix

    def calculate_user_similarity(self, user1, user2):
        # Calculate similarity between two users using Pearson correlation coefficient
        shared_items = set(self.user_item_matrix[user1].keys()) & set(
            self.user_item_matrix[user2].keys()
        )

        if not shared_items:
            return 0  # Users have no shared items

        ratings_user1 = [self.user_item_matrix[user1][item] for item in shared_items]
        ratings_user2 = [self.user_item_matrix[user2][item] for item in shared_items]

        mean_user1 = sum(ratings_user1) / len(ratings_user1)
        mean_user2 = sum(ratings_user2) / len(ratings_user2)

        numerator = sum(
            (x - mean_user1) * (y - mean_user2)
            for x, y in zip(ratings_user1, ratings_user2)
        )
        denominator_user1 = sum((x - mean_user1) ** 2 for x in ratings_user1)
        denominator_user2 = sum((y - mean_user2) ** 2 for y in ratings_user2)

        if denominator_user1 == 0 or denominator_user2 == 0:
            return 0  # Users have no variance

        similarity = numerator / math.sqrt(denominator_user1 * denominator_user2)
        return similarity

    def calculate_recommendations(self, user_idx):
        # Calculate item recommendations for the user
        user = self.data.iloc[user_idx][self.user_field]
        user_ratings = self.user_item_matrix.get(user, {})

        recommendations = []

        for item in self.user_item_matrix.keys():
            if item not in user_ratings:
                weighted_sum = 0
                similarity_sum = 0

                for other_user in self.user_item_matrix.keys():
                    if other_user == user:
                        continue

                    similarity = self.user_similarity_matrix[user][other_user]
                    rating = self.user_item_matrix[other_user].get(item, 0)

                    weighted_sum += similarity * rating
                    similarity_sum += abs(similarity)

                if similarity_sum > 0:
                    predicted_rating = weighted_sum / similarity_sum
                    recommendations.append((item, predicted_rating))

        recommendations.sort(key=lambda x: x[1], reverse=True)
        recommended_items = [item[0] for item in recommendations]

        return recommended_items
