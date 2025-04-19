# automated_model_update.py
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
import datetime
import os
import telegram
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

# --- CONFIG ---
MODEL_PATH = "model/latest_model.pkl"
LOG_PATH = "logs/model_log.csv"
IMG_PATH = "charts/latest_chart.png"
CSV_PATH = "data/latest_data.csv"
BOT_TOKEN = "7427829197:AAGoKH7szS7c30jnSSYIT4YBF141z1UZ2e0"
TELEGRAM_USER_ID = "6329579481"

# --- TELEGRAM BOT ---
bot = telegram.Bot(token=BOT_TOKEN)

def send_telegram(msg):
    bot.send_message(chat_id=TELEGRAM_USER_ID, text=msg)

def compute_indicators(df):
    df['rsi'] = compute_rsi(df['Close'])
    df['ema'] = df['Close'].ewm(span=20, adjust=False).mean()
    df['macd'], df['macd_signal'] = compute_macd(df['Close'])
    df['bb_upper'], df['bb_lower'] = compute_bollinger(df['Close'])
    return df

def compute_rsi(data, window=14):
    delta = data.diff()
    gain = delta.where(delta > 0, 0).rolling(window).mean()
    loss = -delta.where(delta < 0, 0).rolling(window).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def compute_macd(data, short=12, long=26, signal=9):
    macd = data.ewm(span=short, adjust=False).mean() - data.ewm(span=long, adjust=False).mean()
    signal_line = macd.ewm(span=signal, adjust=False).mean()
    return macd, signal_line

def compute_bollinger(data, window=20):
    sma = data.rolling(window).mean()
    std = data.rolling(window).std()
    return sma + 2*std, sma - 2*std

def fetch_data():
    df = yf.download("^NSEI", period="60d", interval="5m", auto_adjust=True)
    df.dropna(inplace=True)
    return df

def prepare_data(df):
    df = compute_indicators(df)
    df.dropna(inplace=True)
    df['target'] = (df['Close'].shift(-1) > df['Close']).astype(int)
    features = ['rsi', 'ema', 'macd', 'macd_signal', 'bb_upper', 'bb_lower']
    return df, df[features], df['target']

def retrain_model(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
    model = RandomForestClassifier()
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    accuracy = accuracy_score(y_test, preds)
    return model, accuracy, preds, X_test, y_test

def plot_predictions(df, preds):
    plt.figure(figsize=(12,6))
    df = df.tail(len(preds)).copy()
    df['pred'] = preds
    df['Close'].plot(label='Price', color='blue')
    buy_signals = df[df['pred'] == 1]
    sell_signals = df[df['pred'] == 0]
    plt.scatter(buy_signals.index, buy_signals['Close'], label='Buy', marker='^', color='green')
    plt.scatter(sell_signals.index, sell_signals['Close'], label='Sell', marker='v', color='red')
    plt.legend()
    plt.title("Live NIFTY50 Predictions")
    plt.savefig(IMG_PATH)
    plt.close()

def log_model_info(accuracy):
    now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_entry = pd.DataFrame([[now, accuracy]], columns=["timestamp", "accuracy"])
    if os.path.exists(LOG_PATH):
        log_entry.to_csv(LOG_PATH, mode='a', header=False, index=False)
    else:
        log_entry.to_csv(LOG_PATH, index=False)

def save_model(model):
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    joblib.dump(model, MODEL_PATH)

def main():
    send_telegram("📈 Starting Auto-Model Update for NIFTY50...")
    df = fetch_data()
    df.to_csv(CSV_PATH)
    df, X, y = prepare_data(df)
    model, accuracy, preds, X_test, y_test = retrain_model(X, y)
    save_model(model)
    plot_predictions(df, preds)
    log_model_info(accuracy)
    summary = f"✅ Model retrained\nAccuracy: {accuracy*100:.2f}%\nData Points: {len(df)}"
    send_telegram(summary)

if __name__ == "__main__":
    main()
