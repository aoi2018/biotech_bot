## pip install dependencies
# pip install pandas
# pip install yfinance
# pip install dotenv
# pip install transformers
# pip install torch
# pip install joblib
# pip install xgboost

# import libraries (pip install dependencies)
import os
import requests
import pandas as pd
from pandas.tseries.offsets import BDay
import numpy as np
import yfinance as yf
from datetime import datetime
from dotenv import load_dotenv
from transformers import pipeline
import torch
import joblib

# new day catalyst import

load_dotenv()
api_key = os.getenv("api_key")
last_trading_day = (pd.Timestamp.now().normalize() - BDay(1)).strftime('%Y-%m-%d')
url = f"https://api.biopharmcatalyst.com/api/user/v1/historical-catalysts"

params = {
    "key": api_key,
    "start_date": last_trading_day,
    "end_date": last_trading_day
}

response = requests.get(url, params = params)
data = response.json()

data = pd.DataFrame(data).dropna().reset_index(drop=True)

if data.empty:
    print("No catalysts for today")
    exit(0)

data = data.drop_duplicates(
    subset=['company_ticker', 'date'], keep = 'last').reset_index(drop=True)

# calculate polarity
finbert = pipeline(task = "text-classification",
                   model = "ProsusAI/finbert",
                   tokenizer = "ProsusAI/finbert",
                   )
output = finbert(data['catalyst'].astype(str).to_list(), batch_size = 32, truncation = True, top_k = None)

data['polarity'] = [{s['label']:s['score'] for s in opt}['positive']-
{s['label']:s['score'] for s in opt}['negative']
for opt in output]

# merge with historical 2025 company features
company_features = pd.read_csv("datasets/company_ds.csv")

data = pd.merge(
    data,
    company_features,
    left_on='company_ticker',
    right_on='ticker',
    how = 'left',
).drop(columns=['ticker'])

# updated market data
market_data = []    
for ticker in data['company_ticker'].unique():
    hist = yf.Ticker(ticker).history(start = pd.to_datetime(last_trading_day) - pd.DateOffset(days=150), end = pd.to_datetime(last_trading_day)).reset_index()
    hist['Ticker'] = ticker
    hist['Date'] = pd.to_datetime(hist['Date'], utc = True).dt.tz_localize(None).dt.normalize()
    keptColumns = ['Date', 'Ticker', 'Open', 'Close', 'High', 'Low', 'Volume']
    hist = hist.reindex(columns = keptColumns)
    market_data.append(hist)
market_data_df = pd.concat(market_data, ignore_index=True)

nbi = yf.Ticker('^NBI').history(start = pd.to_datetime(last_trading_day) - pd.DateOffset(days=150), end = pd.to_datetime(last_trading_day)).reset_index()
nbi = nbi[['Date', 'Close']].rename(columns= {'Close':'NBI'})
nbi['Date'] = pd.to_datetime(nbi['Date'], utc = True).dt.tz_localize(None).dt.normalize()
market_data_df = pd.merge(market_data_df, nbi, on= 'Date', how='left')

# add 30d/60d trends
def trend(market_data_df, ticker, eventDate):
    df = (market_data_df[market_data_df['Ticker'] == ticker].sort_values('Date').reset_index(drop=True))

    df = df[df['Date'] <= pd.to_datetime(eventDate).normalize()]

    if df.empty:
        return {
            'Stock_Trend_30d': np.nan,
            'Stock_Trend_60d': np.nan,
            'NBI_Trend_30d': np.nan,
            'NBI_Trend_60d': np.nan,
            'isDelisted': 1
    }

     
    # extract stock values at specified timepoints
    n=len(df)
    s0 = df.iloc[-1]['Close'] if n>=1 else np.nan
    s30 = df.iloc[-31]['Close'] if n>=31 else np.nan
    s60 = df.iloc[-61]['Close'] if n>=61 else np.nan
    n0 = df.iloc[-1].get('NBI', np.nan) if n>=1 else np.nan
    n30 = df.iloc[-31].get('NBI', np.nan) if n>=31 else np.nan
    n60 = df.iloc[-61].get('NBI', np.nan) if n>=61 else np.nan

    # calculate trends
    return {
        'Stock_Trend_30d': (s0-s30) / s30 if pd.notna(s30) and s30!=0 else np.nan,
        'Stock_Trend_60d': (s0-s60) / s60 if pd.notna(s60) and s60!=0 else np.nan,
        'NBI_Trend_30d': (n0-n30) / n30 if pd.notna(n30) and n30!=0 else np.nan,
        'NBI_Trend_60d': (n0-n60) / n60 if pd.notna(n60) and n60!=0 else np.nan,
        'isDelisted': pd.isna(s30) or pd.isna(s60)
    }

trends_res = [trend(market_data_df, ticker, date)
                          for ticker, date in zip(data['company_ticker'], 
                                                  data['date'])]
trends_df = pd.DataFrame([res if isinstance(res, dict) else {} for res in trends_res], index = data.index)

data = pd.concat([data, trends_df], axis = 1)

# upload models
models = joblib.load('xgb_models_1_to_20.joblib')

# labels
features = ['stage', 
            'polarity',
            'drug_count',
            'Total Revenue', 
            'Operating Revenue', 
            'Cash And Cash Equivalents', 
            'Capital Expenditure', 
            'Stock_Trend_30d', 
            'Stock_Trend_60d',
            'NBI_Trend_30d', 
            'NBI_Trend_60d',
            'isDelisted']

for c in features:
    if data[c].dtype == 'object':
        data[c] = data[c].astype('category')

# new data
X_new = data[features].copy()
X_new['stage'] = X_new['stage'].astype("category")
X_new['isDelisted'] = X_new['isDelisted'].astype(int)

# run regressor predictions
preds_df = pd.DataFrame({f"NCAR{i}": model.predict(X_new) for i, model in enumerate(models.values(), start=1)})
data = pd.concat([data, preds_df], axis =1)
