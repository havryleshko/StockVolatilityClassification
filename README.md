# Stock Volatility Classifier

**Goal:** I want to classify the volatility class using 15Y stock data to help predict whether the stock is likely to be volatile in the future based on past volatility

**What does it do:**
> Data cleaning and EDA of 5 tech stocks
> Time-series feature engineering, such as:
  - Daily returns
  - Rolling volatility (20-day window)
  - Moving averages (-5, -10, -20)
  - Lagged features (previous day volatility and close price)
> Balanced Random Forest Classifier used, hyperparameter tuning
> TimeSeriesSplit
> class_weight='balanced' class imbalance fix

**Clone repository:**
https://github.com/havryleshko/StockVolatilityClassification

**Activating VE:**
python3 -m venv venv
source venv/bin/activate

Usage

Prepare data

Place 15 Years Stock Data of NVDA AAPL MSFT GOOGL and AMZN.csv in a data/ folder.

Run the script

python main.py \
  --data_path data/15_years_stock_data.csv \
  --output_dir models/ \
  --quantile_threshold 0.55

Inspect outputs

Classification report & confusion matrices in console

Best model saved

**ML Pipeline**

1. *Data Loading & Cleaning*

Read CSV, parse dates, drop missing values.

2. *Exploratory Data Analysis*

Summary statistics.

Correlation heatmap.

3. *Feature Engineering*

Compute daily returns, rolling volatility.

Label volatility class by thresholding at a quantile.

Calculate moving averages & lagged features.

4. *Train/Test Split*

80/20 chronological split.

Scale features with StandardScaler.

5. *Modeling*

Train a RandomForestClassifier with class_weight='balanced'.

Evaluate with test set (classification report, confusion matrix).

6. *Validation & Tuning*

Time-series cross-validation (TimeSeriesSplit).

Hyperparameter search with GridSearchCV.

7. *Final Evaluation*

Report performance of tuned model on hold‑out set.

**Results**

*Baseline RF:* accuracy & F1 scores printed in console

Tuned RF

*Optimal hyperparameters:* shown in script output

Final classification report & confusion matrix




