# Beer-Scrapping-and-Analysis
An end-to-end data science project focused on scraping, processing, and analyzing market data from BeerAdvocate (American IPAs). The project transitions from raw web automation to predictive machine learning models and implements a custom frequentist statistical ranking system.

## Project Overview
* **Data Collection:** Automated dynamic web scraping of ~4,950 beer products using **Playwright** and **BeautifulSoup**.
* **Data Engineering:** Extracted numerical metrics from unstructured text (ABV, ratings, dates) and engineered new behavioral features like the **"Hype Factor"**.
* **Machine Learning:** Trained and optimized 7 regression models (including **LightGBM, XGBoost, and Random Forest**) using `GridSearchCV` to predict beer popularity/scores.
* **Statistical Ranking:** Implemented a **Student's t-distribution lower confidence bound** algorithm to solve the "low-volume rating bias" and create a fair product leaderboard.

## Tech Stack
* **Web Scraping:** Playwright, BeautifulSoup4
* **Data Processing:** Pandas, NumPy
* **Visualization:** Seaborn, Matplotlib
* **Machine Learning:** Scikit-Learn, XGBoost, LightGBM
* **Statistics:** SciPy (Stats module)

## Key Highlights & Methodologies

### 1. Robust Web Scraping
To bypass modern bot-detection systems, the scraper utilizes Playwright in headless mode with `AutomationControlled` flags disabled, combined with randomized user behavior simulation. 

### 2. Smart Feature Engineering
* **Hype Factor:** Calculated as `Wants / (Gots + 1)` to evaluate consumer desire versus actual product availability.
* **Days on Site:** Transformed static dates into dynamic continuous variables representing product maturity.

### 3. Machine Learning & Hyperparameter Tuning
Evaluated multiple algorithms to predict the average score (`Avg`). Advanced ensemble methods (XGBoost/LightGBM) were tuned via 5-fold Cross-Validation (`GridSearchCV`).

### 4. Advanced Statistical Ranking (The Solution to Low-Volume Bias)
A common issue in e-commerce and review sites is that a product with a single 5.0 rating ranks higher than a product with a 4.8 rating based on 10,000 reviews. 
To fix this, I implemented a **95% Lower Confidence Bound score** using the Student's t-distribution. This heavily penalizes products with high variance ($\sigma$) or very low rating counts ($n$), ensuring a mathematically reliable top-10 recommendation list.
