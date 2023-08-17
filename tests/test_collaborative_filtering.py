import random
import unittest
from caidin.models.item import Item
from caidin.algorithms.collaborative_filtering import CollaborativeFiltering


class TestCollaborativeFiltering(unittest.TestCase):
    def setUp(self):
        self.cf_engine = CollaborativeFiltering()

    def test_recommendation(self):
        self.cf_engine.load(
            {
                "user1": [1, 2, 0, 0, 1],
                "user2": [0, 3, 1, 0, 2],
                "user3": [2, 0, 3, 1, 0],
            }
        )
        self.cf_engine.train()

        recommendations = self.cf_engine.get("user1", num_recommendations=2)
        self.assertEqual(len(recommendations), 2)


if __name__ == "__main__":
    unittest.main()
