import random
import unittest
from caidin.models.item import Item
from caidin.algorithms.content_based import ContentBased


class TestContentBased(unittest.TestCase):
    def setUp(self):
        self.cb_engine = ContentBased()

    def test_recommendation(self):
        self.cb_engine.load(
            {
                "item1": [1, 0, 1, 0, 1],
                "item2": [0, 1, 0, 1, 0],
                "item3": [1, 1, 0, 0, 1],
            }
        )
        self.cb_engine.train()

        recommendations = self.cb_engine.get("item1", num_recommendations=2)
        self.assertEqual(len(recommendations), 2)


if __name__ == "__main__":
    unittest.main()
