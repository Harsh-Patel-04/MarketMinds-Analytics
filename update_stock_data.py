import yfinance as yf
import pandas as pd
import time

EXCEL_FILE = "StockScope.xlsx"

STOCK_SYMBOLS = [
    "RELIANCE.NS", "TCS.NS", "INFY.NS", "HDFCBANK.NS", "ICICIBANK.NS",
    "SBIN.NS", "BAJFINANCE.NS", "KOTAKBANK.NS", "HINDUNILVR.NS", "ITC.NS",
    "LT.NS", "AXISBANK.NS", "ASIANPAINT.NS", "HCLTECH.NS", "MARUTI.NS",
    "WIPRO.NS", "ONGC.NS", "ULTRACEMCO.NS", "TITAN.NS", "ADANIENT.NS",
    "NTPC.NS", "POWERGRID.NS", "SUNPHARMA.NS", "BHARTIARTL.NS", "TECHM.NS",
    "TATAMOTORS.NS", "DRREDDY.NS", "HEROMOTOCO.NS", "BPCL.NS", "COALINDIA.NS",
    "DIVISLAB.NS", "INDUSINDBK.NS", "NESTLEIND.NS", "ADANIGREEN.NS", "JSWSTEEL.NS",
    "HINDALCO.NS", "TATASTEEL.NS", "BAJAJ-AUTO.NS", "GRASIM.NS", "UPL.NS",
    "ADANIPORTS.NS", "EICHERMOT.NS", "M&M.NS", "APOLLOHOSP.NS", "BRITANNIA.NS",
    "SHREECEM.NS", "DABUR.NS", "SBICARD.NS", "PIDILITIND.NS", "BAJAJFINSV.NS"
]

def update_stock_data():
    while True:
        try:
            stock_data = []
            for symbol in STOCK_SYMBOLS:
                ticker = yf.Ticker(symbol)
                hist = ticker.history(period="2d", interval="1m")

                if not hist.empty:
                    prev_close = hist.iloc[-2]["Close"]
                    latest = hist.iloc[-1]

                    stock_data.append({
                        "Symbol": symbol,
                        "Datetime": latest.name,
                        "Open": latest["Open"],
                        "High": latest["High"],
                        "Low": latest["Low"],
                        "Close": latest["Close"],
                        "PrevClose": prev_close,
                        "Volume": latest["Volume"],
                    })

            df = pd.DataFrame(stock_data)
            df['Datetime'] = pd.to_datetime(df['Datetime'], errors='coerce').dt.tz_localize(None)
            df.to_excel(EXCEL_FILE, index=False)
            print(f"Excel sheet updated at {time.ctime()}!")
            time.sleep(60)
        except Exception as e:
            print(f"Error: {e}")
            time.sleep(60)

if __name__ == "__main__":
    update_stock_data()