# import libraries (pip install dependencies)
import os
import requests
import pandas as pd
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

url = f"https://api.biopharmcatalyst.com/api/user/v1/historical-catalysts?key={api_key}"

response = requests.get(url)
data = response.json()

data = pd.DataFrame(data).dropna().reset_index(drop=True)

today = pd.Timestamp.now().normalize()
data['date'] = pd.to_datetime(data['date'])
data = data[data['date'].dt.normalize() == today]
if data.empty:
    print("No catalysts for today")
    exit(0)

data = data.drop_duplicates(
    subset=['company_ticker', 'date'], keep = 'last').reset_index(drop=True)

model_load = joblib.load("xgb_model.joblib")

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
    hist = yf.Ticker(ticker).history(start = today - pd.DateOffset(days=150), end = today + pd.DateOffset(days=1)).reset_index()
    hist['Ticker'] = ticker
    hist['Date'] = pd.to_datetime(hist['Date'], utc = True).dt.tz_localize(None).dt.normalize()
    keptColumns = ['Date', 'Ticker', 'Open', 'Close', 'High', 'Low', 'Volume']
    hist = hist.reindex(columns = keptColumns)
    market_data.append(hist)
market_data_df = pd.concat(market_data, ignore_index=True)

nbi = yf.Ticker('^NBI').history(start = today - pd.DateOffset(days=150), end = today + pd.DateOffset(days=1)).reset_index()
nbi = nbi[['Date', 'Close']].rename(columns= {'Close':'NBI'})
nbi['Date'] = pd.to_datetime(nbi['Date'], utc = True).dt.tz_localize(None).dt.normalize()
market_data_df = pd.merge(market_data_df, nbi, on= 'Date', how='left')

# add 30d/60d trends
def trend(market_data_df, ticker, eventDate):
    df = (market_data_df[market_data_df['Ticker'] == ticker].sort_values('Date').reset_index(drop=True))

    if df.empty:
        raise ValueError(
            f"No trend data found for this stock: {ticker}"
        )

    df = df[df['Date'] <= pd.to_datetime(eventDate).normalize()]
     
    # extract stock values at specified timepoints
    s0, s30, s60 = df.iloc[-1]['Close'], df.iloc[-31]['Close'], df.iloc[-61]['Close']
    n0, n30, n60 = df.iloc[-1].get('NBI', np.nan), df.iloc[-31].get('NBI', np.nan), df.iloc[-61].get('NBI', np.nan)

    # calculate trends
    return {
        'Stock_Trend_30d': (s0-s30) / s30,
        'Stock_Trend_60d': (s0-s60) / s60,
        'NBI_Trend_30d': (n0-n30) / n30,
        'NBI_Trend_60d': (n0-n60) / n60,
    }

trends_res = [trend(market_data_df, ticker, date)
                          for ticker, date in zip(data['company_ticker'], 
                                                  data['date'])]
trends_df = pd.DataFrame([res if isinstance(res, dict) else {} for res in trends_res], index = data.index)

data = pd.concat([data, trends_df], axis = 1)

# labels
features = ['stage', 
            'polarity',
            'drug_count',
            'Total Revenue', 
            'Cost Of Revenue', 
            'Operating Revenue', 
            'Cash And Cash Equivalents', 
            'Capital Expenditure', 
            'Stock_Trend_30d', 
            'Stock_Trend_60d',
            'NBI_Trend_30d', 
            'NBI_Trend_60d']

for c in features:
    if data[c].dtype == 'object':
        data[c] = data[c].astype('category')

# new data
X_new = data[features].copy()

# run regressor predictions
pred = model_load.predict(X_new)