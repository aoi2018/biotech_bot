## run from the terminal
# brew install libomp

## pip install dependencies
# pip install pandas
# pip install yfinance
# pip install dotenv
# pip install transformers
# pip install torch
# pip install joblib
# pip install xgboost
# pip install Flask
# pip install flask_bootstrap
# pip install matplotlib
# pip install ollama
# ollama pull llama3.2


# import libraries (pip install dependencies)
from flask import Flask, request, jsonify, render_template
from flask_bootstrap import Bootstrap
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
import io
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import base64
import ollama
import json


app=Flask(__name__)
Bootstrap(app)

clicked_ticker = None

@app.route('/endpoint', methods=['POST'])
def endpoint():
    global clicked_ticker
    t = request.get_json() or {}
    clicked_ticker  = t.get('clicked_ticker', '')
    return clicked_ticker

# new day catalyst import

load_dotenv()
api_key = os.getenv("api_key")
# last_trading_day = (pd.Timestamp.now().normalize() - BDay(1)).strftime('%Y-%m-%d')
last_trading_day = (pd.offsets.BDay().rollback(pd.Timestamp.now().normalize() - pd.Timedelta(days = 1))).strftime('%Y-%m-%d')

print(f"Last trading day: {last_trading_day}")

url = f"https://api.biopharmcatalyst.com/api/user/v1/historical-catalysts"

params = {
    "key": api_key
}

response = requests.get(url, params = params)
res = response.json()

df = pd.DataFrame(res)

data = df[df["date"] == last_trading_day].copy()

if data.empty:
    print("No catalysts for last trading day")
    exit(0)

data = data.drop_duplicates(
    subset=['company_ticker', 'date'], keep = 'last').reset_index(drop=True)

print(f"Found {len(data)} catalysts for {last_trading_day}")

# calculate polarity
finbert = pipeline(task = "text-classification",
                   model = "ProsusAI/finbert",
                   tokenizer = "ProsusAI/finbert",
                   )
output = finbert(data['catalyst'].astype(str).to_list(), batch_size = 32, truncation = True, top_k = None)

data['polarity'] = [{s['label']:s['score'] for s in opt}['positive']-
{s['label']:s['score'] for s in opt}['negative']
for opt in output]

### extract company features
def getFinancialInfo(tickers):
    financial_data = []
    for ticker in tickers:
        company = yf.Ticker(ticker)
        row = {"ticker": ticker}
        for statement in [company.financials, company.balance_sheet, company.cashflow]:
            for col in statement.columns:
                if col.year == 2024:
                    row.update(statement[col])
        financial_data.append(row)
    financial_data = pd.DataFrame(financial_data).reindex(columns = 
            ['ticker', 
            'Total Revenue',  
            'Operating Revenue',
            'Cash And Cash Equivalents',
            'Total Debt',
            'Capital Expenditure'])
    return financial_data

company_features = getFinancialInfo(data['company_ticker']).dropna()

data['date'] = pd.to_datetime(data['date'])

drug_count = data[data['date'].dt.year.isin([2025])].groupby('company_ticker')['drug_name'].nunique()

# map back to the df company table
company_features['drug_count'] = data['company_ticker'].map(drug_count).fillna(0).astype(int)

# merge with historical 2025 company features
# company_features = pd.read_csv("datasets/company_ds.csv")

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
            'NBI_Trend_60d': np.nan
            # 'isDelisted': 1
    }

     
    # extract stock values at specified timepoints
    n=len(df)
    closing_price = df.iloc[0]['Close'] if n>=1 else np.nan
    s0 = df.iloc[-1]['Close'] if n>=1 else np.nan
    s30 = df.iloc[-31]['Close'] if n>=31 else np.nan
    s60 = df.iloc[-61]['Close'] if n>=61 else np.nan
    n0 = df.iloc[-1].get('NBI', np.nan) if n>=1 else np.nan
    n30 = df.iloc[-31].get('NBI', np.nan) if n>=31 else np.nan
    n60 = df.iloc[-61].get('NBI', np.nan) if n>=61 else np.nan

    # calculate trends
    return {
        'Closing Price': closing_price,
        'Stock_Trend_30d': (s0-s30) / s30 if pd.notna(s30) and s30!=0 else np.nan,
        'Stock_Trend_60d': (s0-s60) / s60 if pd.notna(s60) and s60!=0 else np.nan,
        'NBI_Trend_30d': (n0-n30) / n30 if pd.notna(n30) and n30!=0 else np.nan,
        'NBI_Trend_60d': (n0-n60) / n60 if pd.notna(n60) and n60!=0 else np.nan
        # 'isDelisted': pd.isna(s30) or pd.isna(s60)
    }

trends_res = [trend(market_data_df, ticker, date)
                          for ticker, date in zip(data['company_ticker'], 
                                                  data['date'])]
trends_df = pd.DataFrame([res if isinstance(res, dict) else {} for res in trends_res], index = data.index)

data = pd.concat([data, trends_df], axis = 1)

# upload models
def eval_rank_ic(*args, **kwargs):
    pass
models = joblib.load('xgb_models_1_to_20.joblib')

# labels
features = ['polarity',
            # 'drug_count',
            'Total Revenue', 
            'Operating Revenue', 
            'Cash And Cash Equivalents', 
            'Capital Expenditure', 
            'Stock_Trend_30d', 
            'Stock_Trend_60d',
            # 'NBI_Trend_30d', 
            'NBI_Trend_60d'
            # 'isDelisted'
            ]

for c in features:
    if data[c].dtype == 'object':
        data[c] = data[c].astype('category')

# new data
X_new = data[features].copy()
# X_new['stage'] = X_new['stage'].astype("category")
# X_new['isDelisted'] = X_new['isDelisted'].astype(int)

# run regressor predictions
preds_df = pd.DataFrame({f"NCAR{i}": model.predict(X_new) for i, model in enumerate(models.values(), start=1)})
data = pd.concat([data, preds_df], axis =1)

# quintile 1 is the highest

if (len(data) >=5):
    data["NCAR20_quintiles"] = pd.qcut(data["NCAR20"].rank(method = "first"), q=5, labels = [5,4,3,2,1])
else: 
    print("Fewer than 5 catalysts for the last trading day")

data_rows = data.to_dict(orient = "records")

print(data.columns)
# print(data_rows)

# local LLM function
def genAIanalysis(clicked_ticker, data):
    clicked_ticker_data = data[data['company_ticker'] == clicked_ticker].to_dict(orient = 'records')[0]
    company_name = clicked_ticker_data.get('company_name')

    prompt =  f"""
        You are an expert biotech investment analyst. Analyze this data for {company_name}:
        Do not show actual figure estimates (dollar amounts etc.). I have separate view for that.
        1. Financial data: 
        
        {json.dumps({
            'Total Revenue': clicked_ticker_data.get('Total Revenue'), 
            'Operating Revenue': clicked_ticker_data.get('Operating Revenue'),
            'Cash And Cash Equivalents':clicked_ticker_data.get('Cash And Cash Equivalents'),
            'Total Debt': clicked_ticker_data.get('Total Debt'), 
            'Capital Expenditure': clicked_ticker_data.get('Capital Expenditure'), 
            # 'drug_count': clicked_ticker_data.get('drug_count'),
            'Stock_Trend_30d': clicked_ticker_data.get('Stock_Trend_30d'),
            'Stock_Trend_60d': clicked_ticker_data.get('Stock_Trend_60d'), 
            # 'NBI_Trend_30d':clicked_ticker_data.get('NBI_Trend_30d'), 
            'NBI_Trend_60d': clicked_ticker_data.get('NBI_Trend_60d'),
            'Forecasted NCAR20': clicked_ticker_data.get('NCAR20')
        })}
        2. Catalyst data:
          {json.dumps({
            'Catalyst': clicked_ticker_data.get('catalyst'),
            'Polarity': clicked_ticker_data.get('polarity'),
            'Drug name': clicked_ticker_data.get('drug_name'), 
            'Indication': clicked_ticker_data.get('indication'), 
            'Label': clicked_ticker_data.get('label'), 
            'Stage': clicked_ticker_data.get('stage')
          })}
        3. Concise and specific trading action advice (Sell, Hold, Buy). Summarize ways to minimize risk.
        """
    try:
        response = ollama.chat(
            model = 'llama3.2',
            messages = [
                {
                    'role': 'user',
                    'content': prompt,
                },
            ]
        )

        return response['message']['content']
    except Exception as e:
        return f"Could not connect to LLM instance"

# generate NCAR chart
def generate_chart(clicked_ticker, df):
    if not clicked_ticker:
        return None
    df_ticker = df[df['company_ticker'] == clicked_ticker]
    if df_ticker.empty:
        return None

    ncar_columns = [f"NCAR{i}"for i in range(1,21)]
    y = df_ticker[ncar_columns].iloc[0].values
    x = list(range(1,21))

    fig, ax = plt.subplots(figsize = (15,10))
    ax.plot(x,y)
    ax.set_xticks(x)
    ax.set_title("Forecasted Daily NCAR Trend", fontsize = 24)
    ax.set_xlabel("Days After Catalyst", fontsize = 20)
    ax.set_ylabel("Normalized Cumulative Abnormal Return (NCAR)", fontsize = 20)
    plt.ylim(-0.3, 0.3)
    buf = io.BytesIO()
    plt.savefig(buf, format='png',bbox_inches = 'tight')
    buf.seek(0)
    chart = base64.b64encode(buf.getvalue()).decode('utf8')
    plt.close(fig)
    return chart
    return None

# generate indicator table info
def display_indicators(clicked_ticker):
    if not clicked_ticker:
        return []
    for item in data_rows:
        if item.get("company_ticker") == clicked_ticker:
            date = item.get('date', 'N/A')
            ticker = item.get('company_ticker', 'N/A')
            company_name = item.get('company_name', 'N/A')
            drug_name = item.get('drug_name', 'N/A')
            nct_number = item.get('nct_number', 'N/A')
            press_link = item.get('press_link', 'N/A')
            polarity = item.get('polarity', 'N/A')
            stage = item.get('stage', 'N/A')
            ncar20 = item.get('NCAR20')
            ncar20_quintile = item.get('NCAR20_quintiles', 'N/A')
            closing_price = item.get('Closing Price', 'N/A')
            stop_loss = closing_price * (1-0.28)
            take_profit = closing_price * (1+0.4)
            lines = [{
                    "Catalyst Date": date, 
                    "Ticker": ticker, 
                    "Company Name": company_name,
                    "Drug Name": drug_name,
                    "NCT Number": nct_number,
                    "Press Link": press_link,
                    "Sentiment Polarity": np.round(polarity,4),
                    "Development Stage": stage,
                    "NCAR20": round(ncar20,4) if ncar20 is not None else 'N/A', 
                    "NCAR20 Quintile": ncar20_quintile,
                    "Closing Price On Catalyst Date": round(closing_price, 2),
                    "Stop Loss Price": round(stop_loss, 2) if ncar20_quintile == 1 else 'N/A',
                    "Take Profit Price": round(take_profit, 2) if ncar20_quintile == 1 else 'N/A',
            }]
            return lines
    return []

# generate catalyst table info
def display_catalysts():
    table_rows = []
    for item in data_rows:
        row = {
            "Catalyst": item.get('catalyst', 'N/A'),
            "Date": item.get('date', 'N/A'),
            "Ticker": item.get('company_ticker', 'N/A'),
            "NCAR20_quintile": item.get('NCAR20_quintiles', 'Too few catalysts'),
        }
        table_rows.append(row)
    return(table_rows)

@app.route("/analytics.html")    
def analytics():
    chart = generate_chart(clicked_ticker, data) if clicked_ticker else None
    tableAnalytics = display_indicators(clicked_ticker)
    # message_text = genAIanalysis(clicked_ticker, data) if clicked_ticker else None

    return render_template('analytics.html', 
    chart = chart, 
    tableAnalytics = tableAnalytics, 
    # text = message_text
    )

@app.route("/recommendations.html")    
def recommendations():
    message_text = genAIanalysis(clicked_ticker, data) if clicked_ticker else None
    return render_template('recommendations.html', text = message_text)

@app.route("/")
def index():
    tableData = display_catalysts()
    return render_template('index.html', tableData = tableData)

if __name__ == '__main__':
    app.run(debug=True, use_reloader = False)