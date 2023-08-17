import random
from caidin.models.item import Item
from caidin.algorithms.collaborative_filtering import CollaborativeFiltering
from caidin.algorithms.content_based import ContentBased
from caidin.algorithms.matrix_factorization import MatrixFactorization

# Create recommendation engines
cf_engine = CollaborativeFiltering()
cb_engine = ContentBased()
mf_engine = MatrixFactorization()

# Adding items to recommendation engines
cf_engine.load(
    {
        "user1": [1, 2, 0, 0, 1],
        "user2": [0, 3, 1, 0, 2],
        "user3": [2, 0, 3, 1, 0],
    }
)

cb_engine.load(
    {
        "item1": [1, 0, 1, 0, 1],
        "item2": [0, 1, 0, 1, 0],
        "item3": [1, 1, 0, 0, 1],
    }
)

mf_engine.load(
    {
        "user1": [1, 2, 0, 0, 1],
        "user2": [0, 3, 1, 0, 2],
        "user3": [2, 0, 3, 1, 0],
    }
)

# Training recommendation engines
cf_engine.train()
cb_engine.train()
mf_engine.train()

# Perform recommendations
cf_recommendations = cf_engine.get("user1", num_recommendations=3)
cb_recommendations = cb_engine.get("item1", num_recommendations=3)
mf_recommendations = mf_engine.get("user1", num_recommendations=3)

# Print recommendations
print("Collaborative Filtering recommendations for user1:")
for recommended_item, similarity in cf_recommendations:
    print(f"{recommended_item} (Similarity: {similarity:.2f})")

print("\nContent-Based recommendations for item1:")
for recommended_item, similarity in cb_recommendations:
    print(f"{recommended_item} (Similarity: {similarity:.2f})")

print("\nMatrix Factorization recommendations for user1:")
for recommended_item, similarity in mf_recommendations:
    print(f"{recommended_item} (Similarity: {similarity:.2f})")
