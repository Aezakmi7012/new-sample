
# Introduction to Machine Learning

Machine learning is a subset of artificial intelligence that enables systems to learn
and improve from experience without being explicitly programmed.

## Supervised Learning

Supervised learning uses labeled training data to learn a mapping from inputs to outputs.
Common algorithms include linear regression, decision trees, and support vector machines.
The model is trained on a dataset where the correct answers are already known.

### Applications of Supervised Learning

- Email spam detection
- Image classification
- Medical diagnosis
- Credit scoring

## Unsupervised Learning

Unsupervised learning finds patterns in data without labeled responses.
Clustering and dimensionality reduction are the primary techniques used.
K-means clustering groups similar data points together based on feature similarity.

### Applications of Unsupervised Learning

- Customer segmentation
- Anomaly detection
- Topic modeling in NLP
- Recommendation systems

## Reinforcement Learning

Reinforcement learning trains agents to make decisions by rewarding desired behaviors.
An agent interacts with an environment and learns from rewards and penalties.
Deep Q-Networks (DQN) combine reinforcement learning with deep neural networks.

### Applications of Reinforcement Learning

- Game playing (AlphaGo, Atari games)
- Robotics and autonomous systems
- Resource management in data centers
- Financial trading strategies

## Neural Networks and Deep Learning

Deep learning uses multi-layered neural networks to learn hierarchical representations.
Convolutional Neural Networks (CNNs) excel at image recognition tasks.
Recurrent Neural Networks (RNNs) and Transformers handle sequential data like text.

### Transformer Architecture

Transformers use self-attention mechanisms to process sequences in parallel.
BERT and GPT are prominent transformer-based models for natural language processing.
The attention mechanism allows the model to focus on relevant parts of the input.

## Model Evaluation

Evaluating machine learning models requires careful selection of metrics.
Accuracy, precision, recall, and F1-score are common classification metrics.
Cross-validation helps estimate how well a model generalises to unseen data.
Overfitting occurs when a model performs well on training data but poorly on test data.

## Feature Engineering

Feature engineering transforms raw data into meaningful inputs for ML models.
Normalisation and standardisation scale features to comparable ranges.
One-hot encoding converts categorical variables into numerical representations.
Principal Component Analysis (PCA) reduces dimensionality while preserving variance.