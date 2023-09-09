# Sample data
data = [
    {
        "item_id": 1,
        "content": "This is the content of item 1",
    },
    {
        "item_id": 2,
        "content": "Content of item 2 is different",
    },
    {
        "item_id": 3,
        "content": "Item 3 has unique content",
    },
    {
        "item_id": 4,
        "content": "Content for item 4 is here",
    },
    {
        "item_id": 5,
        "content": "Fifth item contains content",
    },
]

record = {
    "item_id": [1, 2, 3, 4, 5],
    # ... other record data ...
}

# Create an instance of Caidin
caidin = Caidin()

# Load your data and configure the recommendation method
caidin.load(data).using("content_based").train(record).where(item_id=1)

# Get recommendations
recommended_items = caidin.get()

# Show recommended items
for item in recommended_items:
    print(f"Item {item['item_id']}: {item['content']}")

or 
# Create an instance of Caidin
caidin = Caidin()
caidin.load(data).using(CollaborativeFiltering).train(record).where(user_id='User1').where(item_id='Item', rating='rating')

# Get recommendations
recommended_items = caidin.get()

# Show recommended items
for item in recommended_items:
    print(f"Item {item['item_id']}: {item['content']}")
# To use with Custom Recommmendation engines

# multi_base.py
from caidin.algorithms.recommendation_engine import RecommendationEngine


class MultiBased(RecommendationEngine):
    def recommend(self):
        # Custom recommendation logic here
        pass


# recommend.py
from caidin import Caidin

# import the custom recommendation engines
from multi_based import MultiBased

# Register custom recommendation methods
caidin = Caidin()
caidin.register_method("multi_based", MultiBased)

# Use the library with custom recommendation methods
caidin.load(data).using("multi_based").train(records).get()
