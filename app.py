import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from transformers import pipeline
import feedparser
import urllib.parse
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from datetime import datetime, timedelta
import matplotlib.pyplot as plt

st.set_page_config(page_title="SentimentStock", page_icon="📈", layout="wide")
st.title("📈 SentimentStock — Indian Stock Predictor")
st.markdown("**AI-powered stock movement prediction using News Sentiment + Technical Analysis**")
st.markdown("---")

@st.cache_resource
def load_finbert():
    return pipeline("text-classification", model="ProsusAI/finbert")

@st.cache_data
def get_stock_data(symbol, days=180):
    ticker = yf.Ticker(f"{symbol}.NS")
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days)
    df = ticker.history(start=start_date, end=end_date)
    df = df[["Open","High","Low","Close","Volume"]]
    df.dropna(inplace=True)
    df["Target"] = (df["Close"].shift(-1) > df["Close"]).astype(int)
    return df

def add_indicators(df):
    df["MA_5"] = df["Close"].rolling(5).mean()
    df["MA_20"] = df["Close"].rolling(20).mean()
    df["Returns"] = df["Close"].pct_change()
    df["Returns_5"] = df["Close"].pct_change(5)
    df["Volatility"] = df["Returns"].rolling(5).std()
    delta = df["Close"].diff()
    gain = delta.where(delta > 0, 0).rolling(14).mean()
    loss = -delta.where(delta < 0, 0).rolling(14).mean()
    df["RSI"] = 100 - (100 / (1 + gain/loss))
    df["Volume_Change"] = df["Volume"].pct_change()
    df["High_Low_Ratio"] = (df["Close"] - df["Low"]) / (df["High"] - df["Low"])
    df.dropna(inplace=True)
    return df

def get_sentiment(stock_name, finbert):
    query = urllib.parse.quote(stock_name + " stock NSE India")
    url = f"https://news.google.com/rss/search?q={query}&hl=en-IN&gl=IN&ceid=IN:en"
    feed = feedparser.parse(url)
    headlines = [e.title for e in feed.entries[:15]] or ["Stable market conditions expected"]
    scores = []
    for h in headlines:
        r = finbert(h[:512])[0]
        s = r["score"] if r["label"]=="positive" else (-r["score"] if r["label"]=="negative" else 0)
        scores.append(s)
    return headlines, np.mean(scores)

st.sidebar.header("⚙️ Settings")
stock_options = {
    "Reliance Industries":"RELIANCE",
    "TCS":"TCS",
    "Infosys":"INFY",
    "HDFC Bank":"HDFCBANK",
    "Wipro":"WIPRO",
    "Adani Ports":"ADANIPORTS"
}
selected = st.sidebar.selectbox("Select NSE Stock", list(stock_options.keys()))
symbol = stock_options[selected]

if st.sidebar.button("🔮 Predict Now", type="primary"):
    with st.spinner("Loading FinBERT AI model..."):
        finbert = load_finbert()
    with st.spinner("Fetching stock data..."):
        df = get_stock_data(symbol)
        df = add_indicators(df)
    with st.spinner("Analyzing news sentiment..."):
        headlines, avg_sent = get_sentiment(selected, finbert)

    df["Sentiment"] = avg_sent
    features = ["MA_5","MA_20","Returns","Returns_5","Volatility","RSI","Volume_Change","High_Low_Ratio","Sentiment"]
    X = df[features].copy()
    X.replace([np.inf, -np.inf], np.nan, inplace=True)
    X.fillna(X.median(), inplace=True)
    y = df["Target"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, shuffle=False)
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    latest = X.iloc[-1:]
    pred = model.predict(latest)[0]
    prob = model.predict_proba(latest)[0]

    col1, col2, col3 = st.columns(3)
    col1.metric("Current Price", f"₹{df['Close'].iloc[-1]:.2f}")
    col2.metric("RSI", f"{df['RSI'].iloc[-1]:.1f}")
    col3.metric("News Sentiment", f"{avg_sent:.4f}")

    st.markdown("---")
    if pred == 1:
        st.success(f"## ✅ PREDICTION: {selected} will go UP tomorrow")
        st.success(f"### 🎯 Confidence: {prob[1]*100:.1f}%")
    else:
        st.error(f"## 🔻 PREDICTION: {selected} will go DOWN tomorrow")
        st.error(f"### 🎯 Confidence: {prob[0]*100:.1f}%")

    st.markdown("---")
    st.subheader("📰 Recent News Headlines")
    for h in headlines[:5]:
        st.write(f"• {h}")

    st.subheader("📊 Stock Price Chart")
    fig, ax = plt.subplots(figsize=(12,4))
    ax.plot(df.index, df["Close"], color="blue", linewidth=1.5)
    ax.set_title(f"{selected} - Last 6 Months")
    ax.set_ylabel("Price (INR)")
    st.pyplot(fig)

    st.markdown("---")
    st.warning("⚠️ Disclaimer: Educational purposes only. Not financial advice.")
