import websocket
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import threading
import time
import telegram

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, GRU, Dense
from xgboost import XGBRegressor
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX

# 🚀 Telegram Bot Config
TELEGRAM_BOT_TOKEN = "7108929247:AAFW56Lkn8dyXISXH7lOJNIPPxzMlDGb0oU"
CHAT_ID = "1009817856"
bot = telegram.Bot(token=TELEGRAM_BOT_TOKEN)

# 📊 Live Price Data Store
price_data = []

# 🛠️ WebSocket Connection - Get Live Market Data
def on_message(ws, message):
    data = json.loads(message)
    if "tick" in data:
        price = data["tick"]["quote"]
        price_data.append(price)
        if len(price_data) > 500:
            price_data.pop(0)

def on_open(ws):
    print("✅ WebSocket Connected!")
    ws.send(json.dumps({"ticks": "R_100", "subscribe": 1}))

ws = websocket.WebSocketApp("wss://ws.deriv.com/websockets/v3",
                             on_message=on_message,
                             on_open=on_open)

threading.Thread(target=ws.run_forever, daemon=True).start()

# 🎯 Create AI Models
def create_lstm():
    model = Sequential([
        LSTM(64, return_sequences=True, input_shape=(20, 1)),
        LSTM(64),
        Dense(32, activation="relu"),
        Dense(1)
    ])
    model.compile(optimizer="adam", loss="mse")
    return model

def create_gru():
    model = Sequential([
        GRU(64, return_sequences=True, input_shape=(20, 1)),
        GRU(64),
        Dense(32, activation="relu"),
        Dense(1)
    ])
    model.compile(optimizer="adam", loss="mse")
    return model

# 📈 Train AI Model
def train_model(price_data, model_type="LSTM"):
    if len(price_data) < 100:
        return None  
    x_train, y_train = [], []
    for i in range(80):
        x_train.append(price_data[i:i+20])
        y_train.append(price_data[i+20])
    x_train = np.array(x_train).reshape(-1, 20, 1)
    y_train = np.array(y_train)

    if model_type == "LSTM":
        model = create_lstm()
    else:
        model = create_gru()

    model.fit(x_train, y_train, epochs=10, batch_size=16, verbose=0)
    return model

# 🔮 AI Predictions
def predict_lstm(model, price_data):
    return model.predict(np.array(price_data[-20:]).reshape(1, 20, 1))[0][0]

def predict_gru(model, price_data):
    return model.predict(np.array(price_data[-20:]).reshape(1, 20, 1))[0][0]

def predict_xgboost(price_data):
    model = XGBRegressor(n_estimators=50, learning_rate=0.05)
    x_train = np.array(price_data[:-1]).reshape(-1, 1)
    y_train = np.array(price_data[1:])
    model.fit(x_train, y_train)
    return model.predict(np.array(price_data[-1]).reshape(1, -1))[0]

def predict_arima(price_data):
    model = ARIMA(price_data, order=(3,1,3))
    model_fit = model.fit()
    return model_fit.forecast(steps=1)[0]

def predict_sarima(price_data):
    model = SARIMAX(price_data, order=(3,1,3), seasonal_order=(1,1,1,12))
    model_fit = model.fit(disp=False)
    return model_fit.forecast(steps=1)[0]

# 🖼️ Generate Chart
def generate_chart():
    plt.figure(figsize=(12,6))
    plt.plot(price_data, label="Live Price", color="blue")

    if len(price_data) >= 100:
        lstm_model = train_model(price_data, "LSTM")
        gru_model = train_model(price_data, "GRU")

        if lstm_model and gru_model:
            next_lstm = predict_lstm(lstm_model, price_data)
            next_gru = predict_gru(gru_model, price_data)
            next_xgb = predict_xgboost(price_data)
            next_arima = predict_arima(price_data)
            next_sarima = predict_sarima(price_data)

            plt.scatter(len(price_data), next_lstm, color="red", label="LSTM Prediction")
            plt.scatter(len(price_data)+1, next_gru, color="green", label="GRU Prediction")
            plt.scatter(len(price_data)+2, next_xgb, color="purple", label="XGBoost Prediction")
            plt.scatter(len(price_data)+3, next_arima, color="orange", label="ARIMA Prediction")
            plt.scatter(len(price_data)+4, next_sarima, color="pink", label="SARIMA Prediction")

            plt.legend()
            plt.title("Live Price + AI Predictions")

            img_path = "prediction_chart.png"
            plt.savefig(img_path)
            return img_path
    return None

# 📤 Send Predictions to Telegram
def send_telegram():
    img_path = generate_chart()
    if img_path:
        bot.send_photo(chat_id=CHAT_ID, photo=open(img_path, "rb"))
        bot.send_message(chat_id=CHAT_ID, text="📊 AI Predictions Sent!")

# 🔄 Auto-Update Every 10s
def main_loop():
    while True:
        time.sleep(10)
        send_telegram()

threading.Thread(target=main_loop, daemon=True).start()
