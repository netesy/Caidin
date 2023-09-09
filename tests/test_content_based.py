import unittest
from caidin import Caidin
from caidin.algorithms.content_based import ContentBased


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

record_content_based = {
    "item_id": [1, 2],
    "content": ["Content 1", "Content 2"],
    # ... other record data ...
}


class TestContentBased(unittest.TestCase):
    def test_content_based_recommendation(self):
        caidin = Caidin()
        caidin.load(data).using(ContentBased, content_field="content").train(
            record_content_based
        ).where(item_id=1)
        recommendations = caidin.get()
        self.assertTrue(isinstance(recommendations, list))


if __name__ == "__main__":
    unittest.main()
