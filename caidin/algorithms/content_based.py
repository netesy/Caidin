import math
from collections import defaultdict
from caidin.algorithms.recommendation_engine import RecommendationEngine


class ContentBased(RecommendationEngine):
    def __init__(self, data, records, filters, content_field="content"):
        self.data = data
        self.records = records
        self.filters = filters
        self.content_field = content_field
        self.tfidf_matrix = None
        self.idf_dict = {}
        self.item_index = {}
        self.item_content = []

    def recommend(self):
        # Filter records based on user-defined criteria
        filtered_records = self.records
        for category, value in self.filters.items():
            if category in filtered_records and filtered_records[category] == value:
                continue
            else:
                return []  # No recommendations if filters don't match

        # Calculate content similarity using TF-IDF
        self.calculate_tfidf_matrix()

        # Get the index of the item that matches the given item_id
        item_idx = self.item_index.get(self.filters["item_id"])
        if item_idx is None:
            return []  # Item not found

        # Calculate cosine similarity between items
        sim_scores = self.calculate_cosine_similarity(item_idx)

        # Sort the items by similarity scores
        sim_scores = sorted(enumerate(sim_scores), key=lambda x: x[1], reverse=True)

        # Get the top N similar items (excluding the input item)
        top_similar_items = sim_scores[
            1:11
        ]  # Assuming you want the top 10 recommendations

        # Extract the item indices
        recommended_item_indices = [i[0] for i in top_similar_items]

        # Get the recommended items
        recommendations = [self.data[i] for i in recommended_item_indices]

        return recommendations

    def calculate_tfidf_matrix(self):
        # Calculate TF-IDF matrix for content-based recommendation
        num_items = len(self.data)
        self.tfidf_matrix = []

        for i, item in enumerate(self.data):
            content = item.get(self.content_field, "").lower()
            words = content.split()
            word_count = defaultdict(int)

            for word in words:
                word_count[word] += 1

            tfidf_vector = {}
            for word, count in word_count.items():
                tf = 0.5 + 0.5 * (count / max(word_count.values()))
                idf = self.idf_dict.get(word, 0)
                tfidf = tf * idf
                tfidf_vector[word] = tfidf

            self.tfidf_matrix.append(tfidf_vector)
            self.item_index[item["item_id"]] = i
            self.item_content.append(set(words))

            for word in set(words):
                self.idf_dict[word] = self.idf_dict.get(word, 0) + 1

        for word, idf in self.idf_dict.items():
            self.idf_dict[word] = math.log(num_items / (1 + idf))

        for i in range(num_items):
            for word, tfidf in self.tfidf_matrix[i].items():
                self.tfidf_matrix[i][word] = tfidf * self.idf_dict[word]

    def calculate_cosine_similarity(self, item_idx):
        # Calculate cosine similarity between items
        cosine_sim = []
        tfidf_item = self.tfidf_matrix[item_idx]
        norm_item = sum(tfidf**2 for tfidf in tfidf_item.values())

        for i in range(len(self.tfidf_matrix)):
            if i == item_idx:
                cosine_sim.append(0)
            else:
                tfidf_other = self.tfidf_matrix[i]
                dot_product = sum(
                    tfidf_item[word] * tfidf_other.get(word, 0)
                    for word in tfidf_item.keys()
                )
                norm_other = sum(tfidf**2 for tfidf in tfidf_other.values())
                similarity = (
                    dot_product / (math.sqrt(norm_item) * math.sqrt(norm_other))
                    if norm_item > 0 and norm_other > 0
                    else 0
                )
                cosine_sim.append(similarity)

        return cosine_sim
