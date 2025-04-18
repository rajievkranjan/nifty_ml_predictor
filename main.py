import yfinance as yf
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import time
import requests
from plyer import notification

# Telegram Bot credentials
BOT_TOKEN = "7427829197:AAGoKH7szS7c30jnSSYIT4YBF141z1UZ2e0"
USER_ID = "6329579481"

# Function to send Telegram alert
def send_telegram(message):
    url = f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage"
    data = {"chat_id": USER_ID, "text": message}
    requests.post(url, data=data)

# Compute technical indicators
def compute_rsi(data, window=14):
    delta = data.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def compute_macd(data, short=12, long=26, signal=9):
    macd = data.ewm(span=short, adjust=False).mean() - data.ewm(span=long, adjust=False).mean()
    macd_signal = macd.ewm(span=signal, adjust=False).mean()
    return macd, macd_signal

def compute_ema(data, window=20):
    return data.ewm(span=window, adjust=False).mean()

# Loop for live prediction
print("📈 Live NIFTY 50 Predictor Running...\n")
while True:
    try:
        # Fetch latest 1-minute NIFTY data (auto_adjust explicitly set to False)
        df = yf.download("^NSEI", period="1d", interval="1m", progress=False, auto_adjust=False)

        if df.empty or len(df) < 50:
            print("⚠️ Market might be closed or data not available.")
            time.sleep(60)
            continue

        # Add indicators
        df['rsi'] = compute_rsi(df['Close'])
        df['macd'], df['macd_signal'] = compute_macd(df['Close'])
        df['ema'] = compute_ema(df['Close'])
        df.dropna(inplace=True)

        # Create target column
        df['target'] = (df['Close'].shift(-1) > df['Close']).astype(int)

        # Use last 120 rows for training, 1 row for prediction
        df = df.tail(120)
        X = df[['rsi', 'macd', 'ema']]
        y = df['target']
        X_train, y_train = X[:-1], y[:-1]
        X_live = X.tail(1)

        # Train model
        model = RandomForestClassifier()
        model.fit(X_train, y_train)

        # Predict
        prediction = model.predict(X_live)[0]

        # Fix: Get float value from Series
        current_price = float(df['Close'].iloc[-1])
        signal = "🔼 BUY" if prediction == 1 else "🔻 SELL"
        message = f"NIFTY 50 Signal: {signal}\nCurrent Price: ₹{current_price:.2f}"

        # Show alerts
        print(f"[{pd.Timestamp.now().strftime('%H:%M:%S')}] {message}")
        notification.notify(title="NIFTY 50 Signal", message=message, timeout=5)
        send_telegram(message)

        time.sleep(60)

    except Exception as e:
        print(f"Error: {e}")
        time.sleep(60)
