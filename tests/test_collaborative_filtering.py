import unittest
from caidin import Caidin
from caidin.algorithms.collaborative_filtering import CollaborativeFiltering

# Sample data for testing
data = [
    {
        "item_id": 1,
        "content": "This is the content of item 1",
    },
    {
        "item_id": 2,
        "content": "Content of item 2 is different",
    },
    # Add more data...
]


record_collaborative_filtering = {
    "user_id": ["User1", "User1", "User2"],
    "item_id": ["Item1", "Item2", "Item1"],
    "rating": [5, 4, 3],
    # ... other record data ...
}


class TestCollaborativeFiltering(unittest.TestCase):
    def test_collaborative_filtering_recommendation(self):
        caidin = Caidin()
        caidin.load(data).using(
            CollaborativeFiltering,
            user_field="user_id",
            item_field="item_id",
            rating_field="rating",
        ).train(record_collaborative_filtering).where(user_id="User1")
        recommendations = caidin.get()
        self.assertTrue(isinstance(recommendations, list))


if __name__ == "__main__":
    unittest.main()
