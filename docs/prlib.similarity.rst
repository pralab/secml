Similarity Functions and Matchers
=================================

A matcher is a pairwise metric that compute the similarity or dissimilarity between sets of patterns and return a corresponding score. If s(a, b) > s(a, c), objects a and b are considered “more similar” than objects a and c.

In this package, various strategies for similarity scores combination have been implemented.

Matcher Interface
-----------------

.. automodule:: prlib.similarity.c_similarity
    :members:
    :undoc-members:
    :show-inheritance:

Genuine Template Matching (CGenuineMatcher)
-------------------------------------------

.. automodule:: prlib.similarity.c_genuine_matching
    :members:
    :undoc-members:
    :show-inheritance:

Cohort-based Template Matching
------------------------------

.. automodule:: prlib.similarity.c_cohort_matching
    :members:
    :undoc-members:
    :show-inheritance:

