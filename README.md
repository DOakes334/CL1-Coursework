# Modelling Review Usefulness and Sentiment in Amazon Product Reviews
## ID: 11057361

## Overview

This repository contains the code and materials for a linguistic study examining the relationship between sentiment and perceived usefulness in online product reviews. Using a dataset of Amazon reviews annotated for sentiment, product type, and helpfulness, the project investigates whether sentiment polarity and sentiment extremity meaningfully contribute to review usefulness judgements.

Review usefulness is modelled as a graded construct using a dual-classifier approach rather than as a single binary label, allowing ambiguity in “neutral” reviews to be explicitly captured.

### Data

The dataset consists of 36,547 Amazon product reviews, each associated with:

• Review text

• Sentiment label (positive / negative, derived from star ratings)

• Helpfulness label (helpful / neutral / unhelpful)

• Product category (24 types)

The dataset is automatically downloaded from the course GitHub repository when the main script is run.

### Method

Text is represented using bag-of-words features with unigrams and bigrams, restricted to the 10,000 most frequent tokens.

Review usefulness is modelled via two logistic regression classifiers:

• Model A: helpful vs. not helpful

• Model B: not unhelpful vs. unhelpful

Predicted probabilities from both models are combined to form a continuous usefulness score.

A separate logistic regression model is trained for sentiment classification.

Correlation and aggregation analyses examine relationships between sentiment, usefulness, and product type.

All models are implemented from scratch using NumPy and trained via mini-batch stochastic gradient descent.

### Running the Code

Run CL1 Coursework.py reproduce all experiments and figures.

The script will:

• Download the dataset

• Train usefulness and sentiment models

• Evaluate models on test set

• Generate all reported statistics and plots

### Requirements

• Python 3.8+

• NumPy

• Matplotlib

No external machine learning libraries are required.

### Reproducibility

Random seeds are fixed for data splitting and model training.

All preprocessing, training, and evaluation steps are fully automated.

Results can be reproduced by running the main script without modification.
