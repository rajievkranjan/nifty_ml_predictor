import yfinance as yf
import pandas as pd
import numpy as np
import xgboost as xgb
import time
import datetime
import telegram
from plyer import notification
import matplotlib.pyplot as plt

# --- Your Telegram Bot Setup ---
BOT_TOKEN = "7427829197:AAGoKH7szS7c30jnSSYIT4YBF141z1UZ2e0"
TELEGRAM_USER_ID = "6329579481"
bot = telegram.Bot(token=BOT_TOKEN)

# --- Function to send Telegram messages ---
def send_telegram_message(message):
    bot.send_message(chat_id=TELEGRAM_USER_ID, text=message)

# --- Add more Indicators ---
def compute_bollinger_bands(data, window=20, num_std_dev=2):
    sma = data.rolling(window=window).mean()
    rolling_std = data.rolling(window=window).std()
    upper_band = sma + (rolling_std * num_std_dev)
    lower_band = sma - (rolling_std * num_std_dev)
    return upper_band, lower_band

def compute_vwap(data):
    vwap = (data['Close'] * data['Volume']).cumsum() / data['Volume'].cumsum()
    return vwap

def compute_supertrend(data, atr_period=14, multiplier=3):
    hl = data['High'] - data['Low']
    hl_avg = hl.rolling(window=atr_period).mean()
    atr = hl_avg  # Simplified version of ATR calculation (can be improved)
    upper_band = data['Close'] + (multiplier * atr)
    lower_band = data['Close'] - (multiplier * atr)
    supertrend = pd.DataFrame(index=data.index)
    supertrend['supertrend'] = np.where(data['Close'] > upper_band, upper_band, lower_band)
    return supertrend['supertrend']

# --- Fetch Data from Yahoo Finance ---
def fetch_data():
    print("Fetching NIFTY data...")
    df = yf.download("^NSEI", start="2023-01-01", end="2024-01-01", auto_adjust=True)
    print("Data fetched!")
    return df

# --- Main Trading Logic ---
def trade_logic():
    while True:
        now = datetime.datetime.now()
        # Only run between market hours (9:15 AM - 3:30 PM)
        if now.weekday() >= 5 or now.hour < 9 or (now.hour == 9 and now.minute < 15) or now.hour > 15:
            print("Market closed. Sleeping until next check.")
            time.sleep(60)
            continue
        
        df = fetch_data()

        # --- Adding Indicators ---
        print("Adding technical indicators...")
        df['rsi'] = compute_rsi(df['Close'])
        df['macd'], df['macd_signal'] = compute_macd(df['Close'])
        df['ema'] = compute_ema(df['Close'])
        df['upper_band'], df['lower_band'] = compute_bollinger_bands(df['Close'])
        df['vwap'] = compute_vwap(df)
        df['supertrend'] = compute_supertrend(df)

        # --- Drop NaNs ---
        df.dropna(inplace=True)

        # --- Prepare Data for Prediction ---
        X = df[['rsi', 'macd', 'ema', 'vwap', 'supertrend']]
        y = (df['Close'].shift(-1) > df['Close']).astype(int)

        # --- Train the Model ---
        model = xgb.XGBClassifier()
        model.fit(X, y)

        # --- Make Predictions ---
        preds = model.predict(X)
        df['prediction'] = preds

        # --- Log the Signals ---
        for idx, row in df.iterrows():
            if row['prediction'] == 1:
                send_telegram_message(f"Buy signal at {row.name} with price {row['Close']}")
                notification.notify(
                    title="Buy Signal",
                    message=f"Price: {row['Close']}",
                    timeout=10
                )
                with open('signal_log.csv', 'a') as f:
                    f.write(f"{row.name},{row['Close']},Buy\n")
            elif row['prediction'] == 0:
                send_telegram_message(f"Sell signal at {row.name} with price {row['Close']}")
                notification.notify(
                    title="Sell Signal",
                    message=f"Price: {row['Close']}",
                    timeout=10
                )
                with open('signal_log.csv', 'a') as f:
                    f.write(f"{row.name},{row['Close']},Sell\n")

        # --- Plot Predictions ---
        plt.figure(figsize=(12, 6))
        plt.plot(df['Close'], label='Close Price')
        plt.plot(df['prediction'], label='Prediction', alpha=0.7)
        plt.legend()
        plt.title("Live NIFTY 50 Predictions")
        plt.savefig("prediction_plot.png")
        plt.close()

        print("Sleeping for 5 minutes before next check.")
        time.sleep(300)  # Sleep for 5 minutes before next check

# --- RSI Function ---
def compute_rsi(data, window=14):
    delta = data.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

# --- MACD Function ---
def compute_macd(data, short=12, long=26, signal=9):
    macd = data.ewm(span=short, adjust=False).mean() - data.ewm(span=long, adjust=False).mean()
    macd_signal = macd.ewm(span=signal, adjust=False).mean()
    return macd, macd_signal

# --- EMA Function ---
def compute_ema(data, window=20):
    return data.ewm(span=window, adjust=False).mean()

# --- Start the Trading Bot ---
if __name__ == "__main__":
    trade_logic()
