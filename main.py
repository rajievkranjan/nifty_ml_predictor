import yfinance as yf
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Download NIFTY data
print("Fetching NIFTY data...")
df = yf.download("^NSEI", start="2023-01-01", end="2024-01-01")
print("Data fetched!")

# --- Custom Indicator Calculation ---
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

# Adding the indicators to the dataframe
print("Adding technical indicators...")
df['rsi'] = compute_rsi(df['Close'])
df['macd'], df['macd_signal'] = compute_macd(df['Close'])
df['ema'] = compute_ema(df['Close'])

# Drop missing values (because indicators will create NaN at the start)
df.dropna(inplace=True)
print(f"Data after dropping NaNs: {df.shape[0]} rows")

# Create the target column: 1 if price goes up next day, 0 otherwise
df['target'] = (df['Close'].shift(-1) > df['Close']).astype(int)

# Features and labels
X = df[['rsi', 'macd', 'ema']]
y = df['target']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

# Train the model
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Predict and evaluate
preds = model.predict(X_test)
accuracy = accuracy_score(y_test, preds)

print(f"\nModel Accuracy: {accuracy * 100:.2f}%")

# Show sample predictions
df_test = df.iloc[len(X_train):].copy()
df_test['prediction'] = preds
print(df_test[['Close', 'rsi', 'macd', 'ema', 'target', 'prediction']].tail())
