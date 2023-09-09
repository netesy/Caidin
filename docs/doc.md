
# Caidin Library Documentation

Caidin is a Python library for building recommendation engines, including content-based and collaborative filtering methods. This documentation provides an overview of the library's structure, key components, and how to use and extend it.

## Table of Contents

- [Caidin Library Documentation](#caidin-library-documentation)
  - [Table of Contents](#table-of-contents)
  - [1. Introduction](#1-introduction)
  - [2. Installation](#2-installation)
  - [3. Getting Started](#3-getting-started)
    - [Loading Data](#loading-data)
    - [Configuring Recommendation Methods](#configuring-recommendation-methods)
    - [Training the Recommendation Engine](#training-the-recommendation-engine)
    - [Filtering Data](#filtering-data)
    - [Getting Recommendations](#getting-recommendations)
  - [4. Content-Based Recommendation](#4-content-based-recommendation)
  - [5. Collaborative Filtering Recommendation](#5-collaborative-filtering-recommendation)
  - [6. Extending and Customizing](#6-extending-and-customizing)
  - [7. Default Recommendation Methods](#7-default-recommendation-methods)
  - [8. Testing](#8-testing)

## 1. Introduction

Caidin is a Python library designed to simplify the process of building recommendation engines for various applications. It provides two primary recommendation methods: content-based and collaborative filtering.

- **Content-Based Recommendation:** This method recommends items (e.g., articles, movies) based on their content attributes, such as text or features. It uses TF-IDF (Term Frequency-Inverse Document Frequency) to calculate item similarities.

- **Collaborative Filtering Recommendation:** This method recommends items based on the past interactions of users. It calculates user similarities using Pearson correlation coefficients and predicts user preferences.

## 2. Installation

To use Caidin, you need to install it first. You can install it via pip:

```bash
pip install caidin
```

## 3. Getting Started

### Loading Data

Before using the library, you need to load your data into Caidin. Your data should be structured as a list of dictionaries, where each dictionary represents an item with its attributes.

```python
from caidin import Caidin

data = [
    {
        'item_id': 1,
        'content': 'This is the content of item 1',
    },
    {
        'item_id': 2,
        'content': 'Content of item 2 is different',
    },
    # Add more data...
]

caidin = Caidin()
caidin.load(data)
```

### Configuring Recommendation Methods

You can configure the recommendation method you want to use, such as content-based or collaborative filtering:

```python
from caidin.content_based import ContentBased
from caidin.collaborative_filtering import CollaborativeFiltering

# Content-Based Recommendation
caidin.using(ContentBased, content_field='content')

# Collaborative Filtering Recommendation
caidin.using(CollaborativeFiltering, user_field='user_id', item_field='item_id', rating_field='rating')
```

### Training the Recommendation Engine

Train the recommendation engine by providing the user-item interaction records:

```python
record_content_based = {
    'item_id': [1, 2],
    'content': ['Content 1', 'Content 2'],
    # ... other record data ...
}

record_collaborative_filtering = {
    'user_id': ['User1', 'User1', 'User2'],
    'item_id': ['Item1', 'Item2', 'Item1'],
    'rating': [5, 4, 3],
    # ... other record data ...
}

caidin.train(record_content_based)  # For Content-Based Recommendation
caidin.train(record_collaborative_filtering)  # For Collaborative Filtering Recommendation
```

### Filtering Data

You can filter your data based on criteria using the `where()` method:

```python
caidin.where(item_id=1, user_id='User1')
```

### Getting Recommendations

Get recommendations for the configured method:

```python
recommendations = caidin.get()
for item in recommendations:
    print(f"Recommended Item: {item}")
```

## 4. Content-Based Recommendation

Content-Based Recommendation uses the content attributes of items to generate recommendations. The library calculates TF-IDF scores for each item's content and measures similarity between items based on these scores.

## 5. Collaborative Filtering Recommendation

Collaborative Filtering Recommendation suggests items to users based on their interactions and the interactions of similar users. It calculates user similarities using Pearson correlation coefficients and predicts item preferences.

## 6. Extending and Customizing

You can extend and customize Caidin by creating your recommendation methods or modifying existing ones. To create a new recommendation method, follow the structure of the existing classes (e.g., `ContentBased` or `CollaborativeFiltering`) and implement your logic.

## 7. Default Recommendation Methods

Caidin comes with default recommendation methods, including `ContentBased` and `CollaborativeFiltering`. You can use these methods without registering them explicitly.

## 8. Testing

Caidin includes unit tests to ensure the functionality of its recommendation methods. You can run the tests to verify that the library behaves as expected.

To run tests, execute the following command:

```bash
python -m unittest tests.test_caidin
```

---

This documentation provides an overview of the Caidin library, its usage, and how to extend and customize it. For more detailed information, refer to the source code and documentation of specific classes and methods within the library.
```