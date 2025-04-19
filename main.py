import os
import time
import datetime
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from xgboost import XGBClassifier
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import requests
import telegram
import warnings
warnings.filterwarnings("ignore")

# --- Telegram Bot Setup ---
BOT_TOKEN = "7427829197:AAGoKH7szS7c30jnSSYIT4YBF141z1UZ2e0"
CHAT_ID = "6329579481"
bot = telegram.Bot(token=BOT_TOKEN)

# --- NewsData.io API Setup ---
NEWS_API_KEY = "pub_817045ce645df79e5a0cd83309850d38e110b"

# --- Market Hours ---
MARKET_OPEN = datetime.time(9, 15)
MARKET_CLOSE = datetime.time(15, 30)

# --- Indicator Functions ---
def compute_rsi(data, window=14):
    delta = data.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def compute_macd(data, short=12, long=26, signal=9):
    macd = data.ewm(span=short, adjust=False).mean() - data.ewm(span=long, adjust=False).mean()
    macd_signal = macd.ewm(span=signal, adjust=False).mean()
    return macd, macd_signal

def compute_ema(data, window=20):
    return data.ewm(span=window, adjust=False).mean()

def compute_bollinger_bands(data, window=20):
    sma = data.rolling(window).mean()
    std = data.rolling(window).std()
    upper = sma + 2 * std
    lower = sma - 2 * std
    return upper, lower

def compute_supertrend(df, period=10, multiplier=3):
    hl2 = (df['High'] + df['Low']) / 2
    atr = (df['High'] - df['Low']).rolling(period).mean()
    upperband = hl2 + multiplier * atr
    lowerband = hl2 - multiplier * atr
    trend = [True] * len(df)
    for i in range(1, len(df)):
        if df['Close'][i] > upperband[i-1]: trend[i] = True
        elif df['Close'][i] < lowerband[i-1]: trend[i] = False
        else: trend[i] = trend[i-1]
    return pd.Series(trend, index=df.index)

# --- News Sentiment Function ---
def get_news_sentiment(api_key, query="NIFTY 50", country="in"):
    url = f"https://newsdata.io/api/1/news?apikey={api_key}&q={query}&country={country}&language=en&category=business"
    try:
        res = requests.get(url)
        data = res.json()
        headlines = [art['title'] for art in data.get('results', [])[:5]]
    except Exception:
        return 0.0
    sid = SentimentIntensityAnalyzer()
    scores = [sid.polarity_scores(h)['compound'] for h in headlines]
    return round(np.mean(scores), 3) if scores else 0.0

print("Starting live NIFTY 50 bot with News Sentiment...")

while True:
    now = datetime.datetime.now()
    # check market hours
    if now.weekday() >= 5 or not (MARKET_OPEN <= now.time() <= MARKET_CLOSE):
        print(f"Market closed ({now.strftime('%Y-%m-%d %H:%M:%S')}), sleeping...")
        time.sleep(60)
        continue

    # fetch data
    df = yf.download("^NSEI", period="7d", interval="5m", progress=False)
    if df.empty:
        print("Data fetch failed, retrying...")
        time.sleep(60)
        continue

    # compute indicators
    df['rsi'] = compute_rsi(df['Close'])
    df['macd'], df['macd_signal'] = compute_macd(df['Close'])
    df['ema'] = compute_ema(df['Close'])
    df['upper_bb'], df['lower_bb'] = compute_bollinger_bands(df['Close'])
    df['supertrend'] = compute_supertrend(df)

    # fetch news sentiment
    sentiment = get_news_sentiment(NEWS_API_KEY)
    df['news_sentiment'] = sentiment

    df.dropna(inplace=True)
    df['target'] = (df['Close'].shift(-1) > df['Close']).astype(int)

    features = ['rsi','macd','macd_signal','ema','upper_bb','lower_bb','news_sentiment']
    X = df[features]
    y = df['target']

    # train-test split
    split = int(len(df) * 0.8)
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    model = XGBClassifier(use_label_encoder=False, eval_metric='logloss')
    model.fit(X_train, y_train)

    # predict latest
    last_X = X_test.tail(1)
    pred = model.predict(last_X)[0]
    signal = "Buy" if pred==1 else "Sell"
    price = df['Close'].iloc[-1]
    ts = now.strftime('%Y-%m-%d %H:%M')

    # send alerts
    msg = f"[{ts}] {signal} @ ₹{price:.2f}\nSentiment: {sentiment}"
    bot.send_message(chat_id=CHAT_ID, text=msg)

    # logCSV
    os.makedirs('data', exist_ok=True)
    log_path = os.path.join('data','signal_log.csv')
    header = not os.path.exists(log_path)
    pd.DataFrame([[ts,signal,price,sentiment]], columns=['Time','Signal','Price','Sentiment'])\
      .to_csv(log_path, mode='a', index=False, header=header)

    # plot
    os.makedirs('charts', exist_ok=True)
    plt.figure(figsize=(10,5))
    df['Close'].plot(label='Price')
    df['ema'].plot(label='EMA')
    df['upper_bb'].plot(label='Upper BB', linestyle='--')
    df['lower_bb'].plot(label='Lower BB', linestyle='--')
    plt.title('NIFTY 50 live with news sentiment')
    plt.legend(); plt.grid(); plt.tight_layout()
    plt.savefig('charts/latest_chart.png'); plt.close()

    print(msg)
    time.sleep(300)
