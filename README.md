# Sentiment Analysis of IMDB Movie Reviews (NLP-Project)

## Project Overview
This project focuses on classifying movie reviews from IMDB as either positive or negative using two approaches:
- **Classic Machine Learning:** Naive Bayes.
- **Deep Learning:** LSTM (Long Short-Term Memory).

The primary goal is to explore the strengths and weaknesses of each approach for sentiment analysis.

---

## Dataset Description
- **Source:** IMDB Movie Reviews dataset from TensorFlow Datasets.
- **Size:** 50,000 reviews labeled as positive or negative.
- **Balance:** 50% positive and 50% negative reviews.
- **Characteristics:** Reviews vary in length, requiring preprocessing for compatibility with models.

---

## Use Cases
1. **Content Moderation:** 
   - Social media platforms can use similar models to filter or prioritize content, improving user experience.
2. **Market Research:** 
   - Companies can analyze public sentiment about their brand to develop better marketing strategies and products.

---

## Models Implemented

### 1. Naive Bayes Classifier
- **Preprocessing:**
  - Converts text into numerical features using `TfidfVectorizer` (TF-IDF).
  - Focuses on the top 5,000 important words.
  - Removes standalone numbers to improve feature relevance.
- **Model Details:**
  - Multinomial Naive Bayes, optimized with hyperparameter tuning (e.g., `alpha`).

### 2. LSTM (Long Short-Term Memory)
- **Preprocessing:**
  - Tokenizes text into numeric sequences using the top 5,000 frequent words.
  - Pads or truncates sequences to a fixed length of 200 words.
- **Model Architecture:**
  - **Embedding Layer:** Captures word relationships.
  - **2 Stacked LSTM Layers:** Handles sequential dependencies.
  - **Dense Layer:** Uses sigmoid activation for binary classification.
- **Improvements:**
  - Added **Dropout Layers** to mitigate overfitting.
  - Implemented **EarlyStopping** to stop training when validation loss stops improving.

---

## Challenges and Solutions

### 1. Varying Text Lengths
- **Challenge:** Reviews have different lengths, complicating input processing for LSTM.
- **Solution:** Applied padding and truncation to standardize review lengths to 200 words.

### 2. Overfitting in LSTM
- **Challenge:** The model showed signs of overfitting, with training loss decreasing but validation loss increasing.
- **Solution:**
  - Increased dropout rate to deactivate random neurons during training.
  - Implemented early stopping to halt training when validation loss stopped improving.

---

## Results and Comparison

| Metric      | Naive Bayes | LSTM   |
|-------------|-------------|--------|
| Accuracy    | 83.9%       | 86.4%  |
| Precision   | 85.3%       | 87.2%  |
| Recall      | 82.4%       | 85.1%  |
| F1 Score    | 83.8%       | 86.1%  |

### Observations
- **Naive Bayes:**
  - Efficient and fast.
  - Struggles with contextual understanding, leading to higher false negatives.
- **LSTM:**
  - Better at capturing context and sequential dependencies.
  - Requires more data and computational resources.

---

## Visualizations
1. **Confusion Matrices:**
   - Highlights differences in false positives and false negatives for both models.
2. **Training History Plots:**
   - Tracks accuracy and loss during training to monitor overfitting.

---

## Libraries Used
- **Deep Learning:** TensorFlow, TensorFlow Datasets.
- **Machine Learning:** Scikit-learn.
- **Visualization:** Matplotlib, Seaborn.
- **Data Manipulation:** NumPy, Pandas.

---
