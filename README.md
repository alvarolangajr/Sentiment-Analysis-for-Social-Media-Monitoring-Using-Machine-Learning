# Sentiment-Analysis-for-Social-Media-Monitoring-Using-Machine-Learning
## Project Overview

This project investigates how machine learning can be used to automatically classify social media posts according to sentiment.

Businesses receive large amounts of customer feedback through social media, making it difficult to manually analyse every post. The purpose of this project is to develop a machine-learning application that classifies tweets as either positive or negative.

The research question for the project is:

> How accurately can machine learning classify tweets as positive or negative using text data from social media?

The project was developed using Python and the Scikit-learn machine-learning library.

---

## Dataset

The project uses the **Sentiment140** dataset, which contains approximately 1.6 million tweets.

Dataset source:

https://www.kaggle.com/datasets/kazanova/sentiment140

The original dataset contains six fields:

- Target
- Tweet ID
- Date
- Query
- User
- Text

The project uses the tweet text as the main input and the sentiment target as the classification label.

For this project, the negative (`0`) and positive (`1`) sentiment classes are selected and converted into a binary classification problem:

- `0` = Negative
- `1` = Positive

The neutral class is not used in this implementation.

The dataset is not included in this repository because of its large file size. It should be downloaded from the source above and placed in the `data` directory.

---

## System Pipeline

The machine-learning pipeline consists of the following stages:

```text
Sentiment140 Dataset
        |
        v
Data Loading
        |
        v
Data Inspection
        |
        v
Text Cleaning
        |
        v
Train/Test Split
        |
        v
TF-IDF Feature Extraction
        |
        v
Machine-Learning Models
   /        |        \
  /         |         \
Logistic   Naive      SVM
Regression Bayes
   \         |         /
    \        |        /
     v       v       v
      Evaluation
          |
          v
Confusion Matrices
          |
          v
Classification Error Analysis
```

---

## Text Preprocessing

Social media text can contain URLs, usernames, symbols and inconsistent capitalisation.

The preprocessing stage performs the following operations:

1. Converts text to lowercase.
2. Removes URLs.
3. Removes user mentions.
4. Removes non-alphanumeric characters.
5. Removes unnecessary whitespace.
6. Removes English stop words during TF-IDF feature extraction.

Empty records remaining after cleaning are removed.

---

## TF-IDF Feature Extraction

The cleaned tweet text is converted into numerical features using **TF-IDF (Term Frequency-Inverse Document Frequency)**.

TF-IDF allows the machine-learning algorithms to work with textual data by representing words according to their importance within the dataset.

The implementation uses:

- `max_features = 50,000`
- `min_df = 2`
- English stop-word removal

The training data produced a TF-IDF matrix containing 50,000 features.

---

## Machine-Learning Models

Three classification algorithms are implemented and compared.

### Logistic Regression

Logistic Regression is used as a binary classification model for predicting whether a tweet belongs to the negative or positive class.

### Naive Bayes

Multinomial Naive Bayes provides a computationally efficient probabilistic approach commonly used for text classification.

### Support Vector Machine

A linear Support Vector Machine is used to classify the high-dimensional TF-IDF representation of the tweets.

Comparing multiple models allows their performance to be evaluated using the same dataset and feature representation.

---

## Train/Test Split

The dataset is divided into:

- **80% training data**
- **20% testing data**

A stratified split is used to preserve the class distribution between the training and testing sets.

The final implementation contains:

- 1,277,122 training observations
- 319,281 testing observations

---

## Evaluation

The models are evaluated using:

- Accuracy
- Precision
- Recall
- F1-score
- Confusion matrices

Classification errors are also examined to identify examples where the predicted sentiment differs from the actual sentiment.

---

## Results

The models produced the following results on the test set:

| Model | Accuracy | Precision | Recall | F1-score |
|---|---:|---:|---:|---:|
| Logistic Regression | 0.7773 | 0.7662 | 0.7982 | 0.7819 |
| Naive Bayes | 0.7616 | 0.7628 | 0.7592 | 0.7610 |
| SVM | 0.7722 | 0.7608 | 0.7940 | 0.7771 |

The results show that all three models were able to learn sentiment patterns from the TF-IDF representation.

Logistic Regression produced an accuracy of approximately 77.73% and an F1-score of approximately 78.19%. The SVM produced an accuracy of approximately 77.22%, while Naive Bayes produced an accuracy of approximately 76.16%.

The results are used as part of the analysis in the accompanying academic report.

---

## Confusion Matrices

The repository contains confusion matrices for the three machine-learning models.

The confusion matrices show the number of:

- Correctly classified negative tweets
- Incorrectly classified negative tweets
- Correctly classified positive tweets
- Incorrectly classified positive tweets

These results provide additional information beyond overall accuracy.

---

## Classification Error Analysis

The implementation saves incorrectly classified tweets to:

```text
classification_errors.csv
```

These examples can be inspected to investigate potential causes of classification errors, including:

- Sarcasm
- Informal social-media language
- Abbreviations
- Spelling mistakes
- Context-dependent meaning

---

## Project Structure

```text
sentiment-analysis/
│
├── Sentiment Analysis.py
├── README.md
├── requirements.txt
├── model_results.csv
├── classification_errors.csv
│
├── data/
│   └── training.1600000.processed.noemoticon.csv
│
└── figures/
    ├── logistic_regression_confusion_matrix.png
    ├── naive_bayes_confusion_matrix.png
    └── svm_confusion_matrix.png
```

The Sentiment140 dataset itself should not be uploaded to the repository. Download it from the Kaggle source and place it in the `data` directory.

---

## Requirements

The project uses Python and the following libraries:

- pandas
- numpy
- scikit-learn
- matplotlib
- seaborn

Install the required libraries using:

```bash
pip install -r requirements.txt
```

---

## Running the Project

### 1. Download the dataset

Download the Sentiment140 dataset from:

https://www.kaggle.com/datasets/kazanova/sentiment140

Place:

```text
training.1600000.processed.noemoticon.csv
```

inside:

```text
data/
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Run the Python program

```bash
python "Sentiment Analysis.py"
```

The program loads and preprocesses the dataset, creates TF-IDF features, trains the three machine-learning models, evaluates their performance, generates confusion matrices and saves the classification results.

---

## Limitations

The project has several limitations.

First, Sentiment140 is a historical dataset and may not fully represent language used on modern social-media platforms.

Second, the sentiment labels were generated using an automated approach rather than manually labelled by human annotators, which can introduce label noise.

Third, the implementation performs binary classification using positive and negative classes. Real customer opinions can also be neutral, mixed or expressed with different levels of emotion.

Finally, traditional TF-IDF-based models can have difficulty interpreting sarcasm, context and other forms of informal social-media language.

---

## Business Application

The system demonstrates how automated sentiment classification could support social-media monitoring by processing large volumes of customer feedback.

In a real business environment, the model could be used as a decision-support tool to identify messages that may require further attention. However, important customer feedback should still be reviewed by employees because machine-learning predictions can contain errors.

Future improvements could include retraining the model with newer social-media data and investigating more advanced natural-language-processing approaches for better contextual understanding.

---

## References

Go, A., Bhayani, R. and Huang, L. (2009) *Sentiment140 dataset with 1.6 million tweets*. Kaggle. Available at: https://www.kaggle.com/datasets/kazanova/sentiment140

Scikit-learn developers (2026) *TfidfVectorizer*. Scikit-learn documentation.

Scikit-learn developers (2026) *LogisticRegression*. Scikit-learn documentation.

Scikit-learn developers (2026) *MultinomialNB*. Scikit-learn documentation.

Scikit-learn developers (2026) *LinearSVC*. Scikit-learn documentation.

Scikit-learn developers (2026) *Classification metrics*. Scikit-learn documentation.
