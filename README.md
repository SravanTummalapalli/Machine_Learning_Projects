# Machine Learning Projects

A collection of 17 end-to-end Machine Learning projects covering recommendation systems, classification, regression, NLP, time series forecasting, and deep learning.

---

## Table of Contents

1. [Article Recommendation System](#1-article-recommendation-system)
2. [Breast Cancer Survival Prediction](#2-breast-cancer-survival-prediction)
3. [Classification with Neural Networks](#3-classification-with-neural-networks)
4. [Covid-19 Death Prediction](#4-covid-19-death-prediction)
5. [Cryptocurrency Price Prediction](#5-cryptocurrency-price-prediction)
6. [Flipkart Reviews Sentiment Analysis](#6-flipkart-reviews-sentiment-analysis)
7. [Future Sales Prediction](#7-future-sales-prediction)
8. [Instagram Reach Analysis](#8-instagram-reach-analysis)
9. [Netflix Movie Recommendation System](#9-netflix-movie-recommendation-system)
10. [Netflix Stock Price Prediction](#10-netflix-stock-price-prediction)
11. [Online Payments Fraud Detection](#11-online-payments-fraud-detection)
12. [Perfume Sales Analysis](#12-perfume-sales-analysis)
13. [Stock Price Prediction with LSTM Neural Network](#13-stock-price-prediction-with-lstm-neural-network)
14. [Stress Detection](#14-stress-detection)
15. [Time Series Analysis and Prediction](#15-time-series-analysis-and-prediction)
16. [Ukraine Russia War Twitter Sentiment Analysis](#16-ukraine-russia-war-twitter-sentiment-analysis)
17. [Waiter Tips Prediction](#17-waiter-tips-prediction)

---

## 1. Article Recommendation System

**Type:** Content-Based Filtering

**Overview:**
Recommends articles to readers based on the content similarity of articles they are currently reading — similar to how Medium.com suggests related content.

**Dataset:** `articles.csv` — article titles and full text content.

**Approach:**
- Convert article text to TF-IDF feature vectors (filtering English stop words)
- Compute pairwise cosine similarity between all articles
- Return the top 5 most similar articles for any given article

**Libraries:** `numpy`, `pandas`, `scikit-learn`

**Key Result:** An article on "Clustering Algorithm" correctly returns clustering-related recommendations, confirming strong content-based matching.

---

## 2. Breast Cancer Survival Prediction

**Type:** Binary Classification

**Overview:**
Predicts whether a breast cancer patient will survive after surgery using clinical and protein expression data from over 400 patients.

**Dataset:** `breast_cancer.csv`

| Column | Description |
|--------|-------------|
| `Age`, `Gender` | Patient demographics |
| `Protein1–4` | Protein expression levels |
| `Tumour_Stage` | Stage I, II, or III |
| `Histology` | Tumour tissue type |
| `Patient_Status` | **Target** — Alive or Dead |

**Approach:**
- EDA with pie charts for tumour stage and histology distributions
- Preprocessing: drop nulls, `LabelEncoder` for categoricals, `SimpleImputer` for missing values
- Train a **Support Vector Classifier (SVC)**
- Evaluate with `classification_report`

**Libraries:** `pandas`, `numpy`, `scikit-learn`, `plotly`

**Key Finding:** Most patients are in Stage II; Infiltrating Ductal Carcinoma is the most common histology type.

---

## 3. Classification with Neural Networks

**Type:** Deep Learning — Image Classification

**Overview:**
Classifies 70,000 grayscale clothing images from the **Fashion MNIST** dataset into 10 categories using a fully connected neural network.

**Dataset:** `keras.datasets.fashion_mnist` (auto-downloaded) — 60,000 train + 10,000 test images, each 28×28 pixels.

**Architecture:**
```
Flatten(28×28) → Dense(300, ReLU) → Dense(100, ReLU) → Dense(10, Softmax)
```

**Training:** Loss: Sparse Categorical Crossentropy | Optimizer: SGD | Epochs: 30

**Libraries:** `tensorflow`, `keras`, `numpy`, `matplotlib`

---

## 4. Covid-19 Death Prediction

**Type:** Time Series Forecasting

**Overview:**
Uses historical daily Covid-19 case and death data from India (Jan 2020 – Jan 2022) to forecast deaths for the next **30 days** using automated time series modelling.

**Dataset:** `covid19death.csv`

| Column | Description |
|--------|-------------|
| `Date_YMD` | Date (time index) |
| `Daily Confirmed` | Daily confirmed cases |
| `Daily Deceased` | Daily deaths (forecast target) |

**Approach:**
- Visualize case spikes and death rate (pie chart)
- Calculate overall death rate: `(Total Deceased / Total Confirmed) × 100`
- Forecast next 30 days using **AutoTS** with ensemble methods

**Libraries:** `pandas`, `numpy`, `plotly`, `autots`

**Key Finding:** A major wave is visible between April–May 2021 in the data.

---

## 5. Cryptocurrency Price Prediction

**Type:** Time Series Forecasting

**Overview:**
Predicts Bitcoin (BTC-USD) prices for the next **30 days** using 2 years of live historical data fetched from Yahoo Finance.

**Data Source:** Live via `yfinance` API — Ticker: `BTC-USD`, last 730 days.

**Approach:**
- Fetch and prepare live BTC price data
- Visualize with an interactive candlestick chart
- Analyze feature correlations with `Close` price
- Forecast using **AutoTS** (automated model selection + ensemble)

**Libraries:** `pandas`, `numpy`, `yfinance`, `plotly`, `autots`

**Note:** Predictions are probabilistic estimates and not financial advice.

---

## 6. Flipkart Reviews Sentiment Analysis

**Type:** NLP — Sentiment Analysis

**Overview:**
Analyses customer reviews on **Flipkart** (India's largest e-commerce platform) and classifies them as Positive, Negative, or Neutral using VADER sentiment scoring.

**Dataset:** `flipkart_reviews.csv` — Review text and star ratings. No null values.

**Approach:**
- Clean text: lowercase, remove URLs, HTML, punctuation, numbers, stopwords, apply stemming
- Visualize rating distribution (pie chart) and most common words (word cloud)
- Score each review with VADER: Positive / Negative / Neutral scores
- Aggregate to determine overall sentiment

**Libraries:** `pandas`, `numpy`, `matplotlib`, `plotly`, `nltk`, `wordcloud`

**Key Finding:** ~60% of reviewers gave 5-star ratings; overall sentiment is predominantly **Neutral**.

---

## 7. Future Sales Prediction

**Type:** Regression

**Overview:**
Predicts product sales based on advertising spend across TV, Radio, and Newspaper channels to help businesses optimize their marketing budget.

**Dataset:** `advertising.csv`

| Column | Description |
|--------|-------------|
| `TV` | TV advertising spend (USD) |
| `Radio` | Radio advertising spend (USD) |
| `Newspaper` | Newspaper advertising spend (USD) |
| `Sales` | Units sold — **target** |

**Approach:**
- Scatter plots with OLS trendlines for each channel vs. sales
- Correlation analysis: `TV` has the strongest correlation with `Sales`
- Train a **Linear Regression** model

**Libraries:** `pandas`, `numpy`, `plotly`, `scikit-learn`

**Key Finding:** TV advertising has the strongest positive impact on sales.

---

## 8. Instagram Reach Analysis

**Type:** EDA + Regression

**Overview:**
Analyses real Instagram post data to understand what drives impressions and builds a model to predict future post reach.

**Dataset:** `Instagram data.csv` — per-post impression metrics from Home, Hashtags, Explore, and Other sources, plus engagement metrics (likes, comments, saves, shares).

**Approach:**
- Distribution plots for each impression source
- Pie chart of total impressions by source
- Correlation between engagement metrics and total impressions
- Train a **PassiveAggressiveRegressor** to predict total impressions

**Libraries:** `pandas`, `numpy`, `matplotlib`, `seaborn`, `plotly`, `wordcloud`, `scikit-learn`

**Key Findings:**
- Home feed is the primary source of impressions
- Hashtags are best for reaching new audiences
- Explore section has high variability across posts

---

## 9. Netflix Movie Recommendation System

**Type:** Content-Based Filtering

**Overview:**
Recommends movies based on **cast and crew similarity** — if you liked a film with a particular actor or director, the system finds others featuring the same people.

**Dataset:** `tmdb_5000_credits.csv` — ~4,800 movies with cast and crew in JSON format.

**Approach:**
- Parse top 3 actors and top 3 crew members per movie from JSON fields
- Combine into a single `cast_crew` feature string
- Vectorize with **CountVectorizer**
- Compute cosine similarity and recommend top-N similar movies

**Libraries:** `pandas`, `numpy`, `json`, `scikit-learn`, `matplotlib`

---

## 10. Netflix Stock Price Prediction

**Type:** Deep Learning — Time Series

**Overview:**
Predicts Netflix (NFLX) stock closing prices using an **LSTM neural network** trained on ~13 years of historical data fetched live from Yahoo Finance.

**Data Source:** Live via `yfinance` — Ticker: `NFLX`, last 5,000 days.

**Architecture:**
```
LSTM(128, return_sequences=True) → LSTM(64) → Dense(25) → Dense(1)
```

**Training:** Features: Open, High, Low, Volume | Target: Close | Optimizer: Adam | Loss: MSE | Epochs: 30

**Libraries:** `pandas`, `numpy`, `yfinance`, `plotly`, `keras`, `tensorflow`

---

## 11. Online Payments Fraud Detection

**Type:** Binary Classification

**Overview:**
Detects fraudulent online payment transactions from a large financial dataset using a Decision Tree classifier.

**Dataset:** `credit card.csv`

| Column | Description |
|--------|-------------|
| `type` | Transaction type (CASH_OUT, PAYMENT, etc.) |
| `amount` | Transaction amount |
| `oldbalanceOrg` / `newbalanceOrig` | Account balance before/after |
| `isFraud` | **Target** — Fraud or No Fraud |

**Approach:**
- Visualize transaction type distribution
- Compute correlations with `isFraud`
- Encode `type` to numeric, map `isFraud` to labels
- Train a **Decision Tree Classifier** on 4 key features

**Libraries:** `pandas`, `numpy`, `plotly`, `scikit-learn`

---

## 12. Perfume Sales Analysis

**Type:** Exploratory Data Analysis (EDA)

**Overview:**
Explores pricing, discounts, ratings, and category distributions across a perfume product catalogue to uncover insights about market structure and consumer behaviour.

**Dataset:** `perfume_dataset.csv` — product listings with price, discount, rating, rating count, and gender category.

**Analysis Performed:**
- Product count by category (For Men / For Women / For Men & Women)
- Rating distribution with mean/median reference lines
- Price and discount analysis across categories
- Rating count patterns by category and price range

**Libraries:** `pandas`, `numpy`, `matplotlib`, `seaborn`

**Key Findings:**
- Men's products dominate the catalogue (~45%)
- Women's products are significantly underrepresented (~19%)
- Ratings cluster towards the higher end (positivity bias)

---

## 13. Stock Price Prediction with LSTM Neural Network

**Type:** Deep Learning — Time Series

**Overview:**
Predicts **Apple Inc. (AAPL)** stock closing prices using an LSTM network trained on ~13 years of live historical data. LSTM's memory capability makes it ideal for capturing long-term price trends.

**Data Source:** Live via `yfinance` — Ticker: `AAPL`, last 5,000 days.

**Architecture:**
```
LSTM(128, return_sequences=True) → LSTM(64) → Dense(25) → Dense(1)
```

**Training:** Features: Open, High, Low, Volume | Target: Close | Optimizer: Adam | Loss: MSE | Epochs: 30 | Batch size: 1

**Libraries:** `pandas`, `numpy`, `yfinance`, `plotly`, `scikit-learn`, `keras`, `tensorflow`

---

## 14. Stress Detection

**Type:** NLP — Binary Classification

**Overview:**
Classifies Reddit posts as **Stressed** or **Not Stressed** based on text content, helping organizations identify individuals who may need mental health support.

**Dataset:** `stress.csv` — Reddit posts from mental health subreddits. Labels: 0 = No Stress, 1 = Stress.

**Approach:**
- Clean text: lowercase, remove URLs, HTML, punctuation, numbers, stopwords; apply Snowball stemming
- Visualize most common words via word cloud
- Vectorize with **CountVectorizer**
- Train a **Bernoulli Naive Bayes** classifier (well-suited for binary text classification)
- Accept live user input and predict stress in real time

**Libraries:** `pandas`, `numpy`, `nltk`, `matplotlib`, `wordcloud`, `scikit-learn`

---

## 15. Time Series Analysis and Prediction

**Type:** Time Series — SARIMA Forecasting

**Overview:**
Analyses and forecasts **Furniture sales** from the Sample Superstore dataset using classical time series techniques including decomposition and SARIMA modelling.

**Dataset:** `Sample - Superstore.csv` — retail order data; filtered to Furniture category and resampled to monthly averages.

**Approach:**
- Filter to Furniture, set `Order Date` as index, resample to monthly mean sales
- Visualize seasonal patterns (sales dip at year start, peak at year end)
- Decompose series into **Trend**, **Seasonality**, and **Residual**
- ADF test for stationarity
- Grid search over SARIMA `(p,d,q)(P,D,Q,s)` parameters by AIC
- Generate forecasts with confidence intervals

**Libraries:** `pandas`, `numpy`, `matplotlib`, `statsmodels`

---

## 16. Ukraine Russia War Twitter Sentiment Analysis

**Type:** NLP — Sentiment Analysis

**Overview:**
Analyses public Twitter sentiment about the **Ukraine–Russia War**, focusing on English-language tweets to determine the overall tone of global discourse.

**Dataset:** `filename.csv` — tweets with username, tweet text, and language. No null values. Majority of tweets are in English.

**Approach:**
- Filter to `username`, `tweet`, `language` columns
- Clean text: lowercase, remove URLs, HTML, punctuation, numbers, stopwords
- Generate word cloud of most frequent terms
- Score each tweet with **VADER** (Positive / Negative / Neutral)
- Aggregate scores for overall sentiment conclusion

**Libraries:** `pandas`, `numpy`, `matplotlib`, `nltk`, `wordcloud`

---

## 17. Waiter Tips Prediction

**Type:** Regression

**Overview:**
Predicts the tip amount given to restaurant waiters based on bill size, party size, day of week, meal time, gender, and smoking status.

**Dataset:** `tips.csv`

| Column | Description |
|--------|-------------|
| `total_bill` | Total bill in USD |
| `tip` | Tip amount — **target** |
| `sex` | Gender of the payer |
| `smoker` | Smoker or non-smoker |
| `day` | Day of the week |
| `time` | Lunch or Dinner |
| `size` | Party size |

**Approach:**
- Scatter plots of tips vs. bill, party size, day, gender, and smoking status
- Encode categorical features to numeric
- Train a **Linear Regression** model

**Libraries:** `pandas`, `numpy`, `plotly`, `scikit-learn`

**Key Findings:**
- Saturday generates the most tips
- Men tip more than women on average
- Total bill is the strongest predictor of tip amount

---

## Requirements

Install all common dependencies with:

```bash
pip install numpy pandas scikit-learn matplotlib seaborn plotly nltk wordcloud \
            tensorflow keras yfinance autots statsmodels
```

For NLTK resources, run once in Python:
```python
import nltk
nltk.download("stopwords")
nltk.download("vader_lexicon")
```

---

## Project Structure

```
Machine_Learning_Projects/
├── Article Recommendation System/
├── Breast Cancer Survival Prediction/
├── Classification with Neural Networks/
├── Covid-19 Death Prediction/
├── Cryptocurrency Price Prediction/
├── Flipkart Reviews Sentiment Analysis/
├── Future Sales Prediction/
├── Instagram_Reach_Analysis/
├── Netflix_Movie_Recommendation/
├── Netflix Stock Price Prediction/
├── Online_Payments_fraud_detection/
├── Perfume sales analysis/
├── Stock Price Prediction with LSTM Neural Network/
├── Stress Detection/
├── Time Series Analysis and Prediction/
├── Ukraine Russia War Twitter Sentiment Analysis/
└── Waiter Tips Prediction/
```
