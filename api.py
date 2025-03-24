from flask import Flask, jsonify
from flask_cors import CORS
import pandas as pd
import yfinance as yf

app = Flask(__name__)
CORS(app)

EXCEL_FILE = "StockScope.xlsx"

@app.route("/get-stock-data", methods=["GET"])
def get_stock_data():
    try:
        df = pd.read_excel(EXCEL_FILE)
        df['Datetime'] = pd.to_datetime(df['Datetime'], errors='coerce').astype(str)
        return jsonify(df.to_dict(orient="records"))
    except Exception as e:
        return jsonify({"error": str(e)})

@app.route("/get-company-insights/<symbol>", methods=["GET"])
def get_company_insights(symbol):
    try:
        stock = yf.Ticker(symbol)
        info = stock.info
        balance_sheet = stock.balance_sheet
        balance_sheet_dict = balance_sheet.to_dict() if balance_sheet is not None else {}

        insights = {
            "name": info.get("longName", "N/A"),
            "sector": info.get("sector", "N/A"),
            "industry": info.get("industry", "N/A"),
            "marketCap": info.get("marketCap", "N/A"),
            "peRatio": info.get("trailingPE", "N/A"),
            "eps": info.get("trailingEps", "N/A"),
            "dividendYield": info.get("dividendYield", "N/A"),
            "profitMargins": info.get("profitMargins", "N/A"),
            "revenueGrowth": info.get("revenueGrowth", "N/A"),
            "shareholders": info.get("heldPercentInsiders", "N/A"),
            "balanceSheet": balance_sheet_dict,
        }
        return jsonify(insights)
    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == "__main__":
    app.run(debug=True)