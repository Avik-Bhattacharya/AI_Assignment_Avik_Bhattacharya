# Customer Support Ticket Classification System

An end-to-end Machine Learning and Natural Language Processing (NLP) project that classifies customer support ticket descriptions into five operational categories.

---

## Table of Contents
1. [Project Overview](#project-overview)
2. [Problem Statement](#problem-statement)
3. [Technologies Used](#technologies-used)
4. [Dataset Description](#dataset-description)
5. [Data Preprocessing](#data-preprocessing)
6. [Exploratory Data Analysis](#exploratory-data-analysis)
7. [Model Development](#model-development)
8. [Model Evaluation](#model-evaluation)
9. [Project Structure](#project-structure)
10. [Installation and Setup](#installation-and-setup)
11. [How to Run](#how-to-run)
12. [Sample Prediction](#sample-prediction)
13. [Conclusion](#conclusion)
14. [Project Limitations](#project-limitations)
15. [Future Improvements](#future-improvements)
16. [AI Assistance Disclosure](#ai-assistance-disclosure)

---

## Project Overview

The **Customer Support Ticket Classification System** is an NLP project designed to automatically categorize incoming support tickets based on their text descriptions. 

Using text processing techniques and machine learning classifiers in Python, the project trains a model to predict one of five operational categories: **Account Update**, **Application Error**, **Login Issue**, **Performance**, or **Report**. The project includes a data preprocessing pipeline, model training and evaluation routines, a terminal-based prediction script, and a local Streamlit web application.

---

## Problem Statement

Customer support teams receive a large volume of tickets every day. Manually reviewing and assigning tickets to the right department takes time and can lead to routing mistakes and response delays.

Automating this classification process using machine learning helps:
- Direct support requests to the right team quickly.
- Reduce manual triage effort for support agents.
- Improve response times and service consistency.

---

## Technologies Used

| Technology | Purpose |
| :--- | :--- |
| **Python** | Core programming language for data processing, modeling, and applications |
| **Pandas** | Loading, cleaning, and inspecting tabular ticket data |
| **NumPy** | Numerical operations and array handling |
| **Scikit-learn** | Text feature extraction (TF-IDF), model training, pipelines, and evaluation metrics |
| **TF-IDF Vectorizer** | Converting text descriptions into numerical n-gram features |
| **LinearSVC** | Linear Support Vector Classifier used for category prediction |
| **Streamlit** | Local interactive web interface for testing ticket predictions |
| **Matplotlib & Seaborn** | Creating exploratory data analysis plots and the confusion matrix chart |
| **Joblib** | Saving and loading the trained model pipeline |

---

## Dataset Description

The dataset contains **250 customer support ticket records** categorized into five distinct support areas:

- **Account Update**: Requests to update user profiles, billing details, contact information, or seat allocations.
- **Application Error**: Software bugs, unhandled exceptions, HTTP errors (e.g., 500, 404), and broken user interface elements.
- **Login Issue**: Authentication errors, password reset issues, single sign-on (SSO) failures, and multi-factor authentication (MFA) problems.
- **Performance**: Reports of slow loading, system lag, timeouts, and high resource utilization.
- **Report**: Requests regarding scheduled reports, analytics exports (CSV/PDF), and metric discrepancies.

### Dataset Schema

| Column Name | Data Type | Description |
| :--- | :--- | :--- |
| `ticket_id` | String | Unique identifier for each ticket (e.g., `T001` to `T250`) |
| `ticket_description` | String | Text describing the customer problem |
| `category` | String | Target category label (5 categories) |
| `priority` | String | Priority level (`Low`, `Medium`, `High`) |
| `status` | String | Ticket status (`Open`, `In Progress`, `Closed`) |

> **Note on Dataset**: The dataset was synthetically created for this technical assignment to simulate realistic support ticket scenarios without using private customer information. Two ambiguous records were manually audited and corrected in `data/tickets.csv`: `T025` (*Application Error*) and `T132` (*Account Update*).

---

## Data Preprocessing

The preprocessing pipeline is implemented in `src/preprocessing.py`. It prepares raw customer ticket text for machine learning through the following steps:

1. **Handling Missing Values**:
   - Validates that every record has a valid `ticket_description` and `category`. Any record missing these essential fields is removed because models require both input text and target labels to learn.
   - Any missing values in optional fields (`priority`, `status`) are filled with `"Unknown"`.
2. **Deduplication**:
   - Checks for and removes exact duplicate records.
   - Verifies ticket ID uniqueness to avoid duplicate identifiers.
3. **Text Cleaning (`clean_text`)**:
   - Converts all text to lowercase so words with different capitalizations (e.g., `"Login"` and `"login"`) are treated identically.
   - Removes punctuation and special characters using a regular expression (`[^a-z0-9\s]`), keeping only alphanumeric characters and spaces.
   - Replaces multiple consecutive spaces with a single space and trims whitespace.
   - Retains negation words such as `"not"`, `"cannot"`, and `"unable"` to preserve critical negative context in problem descriptions.
4. **Usage in Training**:
   - The cleaned text provides standardized, noise-free input features for TF-IDF feature extraction during model training directly in memory.

---

## Exploratory Data Analysis

Exploratory data analysis was conducted to understand distributions across the dataset. The resulting charts are saved in the `screenshots/` directory:

1. **Category Distribution (`screenshots/category_distribution.png`)**:
   - Shows the count of tickets across all five categories: Application Error (51, 20.4%), Account Update (51, 20.4%), Report (50, 20.0%), Performance (50, 20.0%), and Login Issue (48, 19.2%).
2. **Priority Distribution (`screenshots/priority_distribution.png`)**:
   - Displays the breakdown across priority levels: Low with 93 tickets (37.2%), High with 91 tickets (36.4%), and Medium with 66 tickets (26.4%).
3. **Status Distribution (`screenshots/status_distribution.png`)**:
   - Shows the distribution across operational statuses: Open with 93 tickets (37.2%), Closed with 80 tickets (32.0%), and In Progress with 77 tickets (30.8%).

---

## Model Development

Model development and evaluation are implemented in `src/train.py`.

### 1. Feature Extraction: TF-IDF
Text descriptions are converted to numerical vectors using Scikit-learn's `TfidfVectorizer`:
- **N-gram Range**: `(1, 2)` (unigrams and bigrams), allowing the model to capture single words (e.g., `"timeout"`) as well as key word pairs (e.g., `"cannot login"`, `"reset password"`).
- **Stop Words**: Standard English stop words are removed to eliminate generic filler words.
- **Lowercasing**: Text is converted to lowercase during tokenization.
- **Vocabulary Size**: 2,700 unique n-gram features learned from the training set.

### 2. Model Comparison and Selection
Three classification algorithms were compared using 5-fold stratified cross-validation on the 250 records:

| Model | Mean CV Accuracy | Mean CV Weighted F1-Score |
| :--- | :---: | :---: |
| **Linear Support Vector Machine (LinearSVC)** | **0.8000** | **0.7967** |
| **Logistic Regression** | 0.7960 | 0.7934 |
| **Multinomial Naive Bayes** | 0.7920 | 0.7857 |

All three models showed comparable performance on this dataset. **LinearSVC** was selected as the final classifier because it achieved a slightly higher mean weighted F1-score (0.7967) and is well-suited for high-dimensional, sparse TF-IDF feature representations.

### 3. Training Setup
- **Train-Test Split**: An 80:20 stratified split divides the dataset into 200 training tickets and 50 held-out test tickets.
- **Stratification**: Preserves equal class proportions (10 test tickets per category) across both sets.
- **Random State**: `random_state=42` ensures reproducible data splitting and model initialization.
- **Pipeline Structure**: TF-IDF vectorization and LinearSVC are combined in a Scikit-learn `Pipeline`. The vectorizer is fit only on the training set to prevent data leakage.

---

## Model Evaluation

The trained LinearSVC model was evaluated on the **50 held-out test records** that were not used during training.

### Held-Out Test Set Performance

| Metric | Score |
| :--- | :---: |
| **Accuracy** | **84.00%** |
| **Weighted Precision** | **85.09%** |
| **Weighted Recall** | **84.00%** |
| **Weighted F1-Score** | **84.09%** |
| **Training Records** | 200 |
| **Testing Records** | 50 |

### Classification Report by Category

| Category | Precision | Recall | F1-Score | Support |
| :--- | :---: | :---: | :---: | :---: |
| **Account Update** | 1.00 | 1.00 | 1.00 | 10 |
| **Application Error** | 0.75 | 0.60 | 0.67 | 10 |
| **Login Issue** | 0.89 | 0.80 | 0.84 | 10 |
| **Performance** | 0.62 | 0.80 | 0.70 | 10 |
| **Report** | 1.00 | 1.00 | 1.00 | 10 |
| **Weighted Average** | **0.85** | **0.84** | **0.84** | **50** |

### Confusion Matrix
The confusion matrix chart is saved at `screenshots/confusion_matrix.png`:
- **Account Update** and **Report** achieved perfect classification (10/10 correct).
- Minor misclassifications occurred between **Application Error** and **Performance**, where tickets describing application freezes or timeouts share similar vocabulary.

### Manual Testing on Unseen Samples
Five additional support ticket descriptions not used in model training were tested to evaluate qualitative behavior on new text:

| # | Test Input Description | Expected Category | Predicted Category | Match |
| :---: | :--- | :---: | :---: | :---: |
| 1 | Two-factor authentication SMS code is never delivered to my registered mobile phone. | Login Issue | Login Issue | Correct |
| 2 | Encountering HTTP 500 Internal Server Error when trying to submit an invoice for approval. | Application Error | Application Error | Correct |
| 3 | Please update the primary billing contact email address to accounting@company.com. | Account Update | Account Update | Correct |
| 4 | Exporting the Q3 inventory report to CSV results in a corrupted file with misaligned columns. | Report | Report | Correct |
| 5 | The main analytics dashboard takes over 45 seconds to load during peak morning hours. | Performance | Performance | Correct |

- **Result**: All **5 out of 5 (100%)** unseen descriptions were classified into their expected categories.
- *Note*: This manual test provides qualitative demonstration of prediction behavior and is distinct from the statistical held-out test accuracy of 84.00%.

---

## Project Structure

The project follows the company-provided final folder structure:

```text
AI_Assignment_Avik_Bhattacharya/
│
├── data/
│   └── tickets.csv
│
├── model/
│   └── trained_model.joblib
│
├── screenshots/
│   ├── category_distribution.png
│   ├── confusion_matrix.png
│   ├── priority_distribution.png
│   └── status_distribution.png
│
├── src/
│   ├── preprocessing.py
│   ├── train.py
│   ├── predict.py
│   └── app.py
│
├── .gitignore
├── README.md
├── requirements.txt
└── report.pdf
```

---

## Installation and Setup

### 1. Prerequisites
- Windows 10/11
- Python 3.11 or higher
- PowerShell

### 2. Navigate to the Project Directory

Open PowerShell and navigate to the directory where you have downloaded or cloned the project.

```powershell
cd path\to\AI_Assignment_Avik_Bhattacharya
```

### 3. Create and Activate Virtual Environment
```powershell
# Create virtual environment
python -m venv venv

# If PowerShell script execution is restricted, run:
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope Process

# Activate virtual environment
.\venv\Scripts\Activate.ps1
```

### 4. Install Dependencies
```powershell
pip install -r requirements.txt
```

---

## How to Run

### 1. Preprocess the Dataset
To inspect raw tickets, clean text descriptions, and verify data quality:
```powershell
python src/preprocessing.py
```

### 2. Train and Evaluate the Model
To train the LinearSVC model on the dataset, evaluate metrics on the test partition, save the confusion matrix chart, and export the trained model artifact:
```powershell
python src/train.py
```
- Outputs training summary and evaluation metrics in the terminal.
- Saves the confusion matrix heatmap to `screenshots/confusion_matrix.png`.
- Saves the trained model pipeline to `model/trained_model.joblib`.

### 3. Run Predictions in the Terminal (CLI)
To test ticket predictions interactively from the command line:
```powershell
python src/predict.py
```
- Type or paste a support ticket description when prompted.
- The script prints the entered text and the predicted category.
- Type `exit` or `quit` to end the session.

### 4. Run the Streamlit Web Application
To launch the interactive web interface:
```powershell
streamlit run src/app.py
```
- Opens locally in your default web browser (usually `http://localhost:8501`).
- **Features**:
  - A text box to enter custom ticket descriptions.
  - A dropdown menu to load sample pre-filled tickets for testing.
  - A "Predict Category" button that runs inference using the saved model.
  - A prominent prediction card displaying the predicted category.
  - An "Inspection Details" expandable section showing input verification and classifier details.
  - A sidebar listing the five supported categories and model information.

---

## Sample Prediction

To test the model, enter the following sample ticket description:

- **Input**:
  ```text
  I cannot access my account because the password reset link is not working.
  ```

- **Predicted Category**:
  ```text
  Login Issue
  ```

- **Programmatic Python Example**:
  ```python
  from src.predict import predict_ticket

  ticket = "I cannot access my account because the password reset link is not working."
  category = predict_ticket(ticket)
  print(f"Predicted Category: {category}")
  # Output: Predicted Category: Login Issue
  ```

---

## Conclusion

This project demonstrates a functional machine learning pipeline for automated customer support ticket triage. Combining TF-IDF n-gram vectorization with a Linear Support Vector Classifier achieves an **84.00% accuracy** and an **84.09% weighted F1-score** on unseen test data. The system offers accessible testing through both a command-line interface (`src/predict.py`) and a user-friendly Streamlit web application (`src/app.py`).

---

## Project Limitations

1. **Dataset Size**: The current dataset contains 250 records (~50 per category), which is sufficient for learning basic linear decision boundaries but lacks the breadth needed for rare, highly idiosyncratic customer phrasings.
2. **Lexical Overlap Across Classes**: Tickets describing complex software failures (e.g., an application freezing before crashing) share vocabulary with performance issues, occasionally leading to confusion between *Application Error* and *Performance*.
3. **Single-Category Assignment**: Real-world support requests occasionally touch on multiple topics (e.g., updating an account to resolve a login failure). The current model operates under a single-label assumption.
4. **Synthetic Data**: The dataset was synthetically constructed for this assignment and may not capture conversational noise, typos, or grammatical irregularities of live production environments.

---

## Future Improvements

1. **Dataset Scale & Diversity**: Expand the corpus to several thousand real-world customer tickets to capture long-tail domain jargon and conversational variations.
2. **Multi-Label Classification**: Transition to multi-label classification to allow support requests to be tagged with primary and secondary departments.
3. **Confidence Thresholding**: Utilize classifier decision margins to route high-confidence tickets automatically while flagging uncertain predictions for human supervisor review.
4. **Transformer Models**: Experiment with lightweight, domain-adapted transformer architectures (e.g., DistilBERT) to test whether contextual embeddings better resolve semantic overlap between application bugs and performance bottlenecks.

---

## AI Assistance Disclosure

AI tools were used as supplementary aids during this project for generating synthetic sample ticket descriptions, assisting with selected coding and debugging tasks, improving documentation and formatting, and reviewing the project for completeness.

The project implementation, model evaluation, and final outputs were reviewed and tested as part of the submission process. AI-generated suggestions were checked and adapted where necessary.
