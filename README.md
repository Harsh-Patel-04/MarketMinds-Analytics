# MarketMinds-Analytics

## Overview

This project provides a comprehensive stock market analysis and price prediction system. It processes historical stock data, applies machine learning models for forecasting, and serves predictions via a web API.

## Features

### 1. Stock Data Handling

Loads historical stock price data from CSV files.

Updates stock data dynamically using automated scripts.

### 2. Machine Learning & Forecasting

Implements LSTM, GRU, and Prophet models for stock price prediction.

Uses pre-trained models for forecasting future trends.

### 3. Web API for Stock Data

Serves stock data and predictions through API endpoints.

Built using Flask for real-time data access.

### 4. Stock Dashboard

Generates stock insights using stock_dashboard.py.

Web-based visualization using index.html.

### 5. Automated Data Processing

Includes scripts to automate stock data updates.

Performs data cleaning and transformation for analysis.

## Benefits

- Real-time Stock Analysis – Provides updated market insights.
- Predictive Analytics – Helps in forecasting future trends.
- Web Integration – Can be accessed via a web interface.
- Data-Driven Decision Making – Useful for investors and traders.

## Installation

- Clone the repository: git clone <repository_url>

- Navigate to the project folder: cd project_folder

- Install dependencies: pip install -r requirements.txt

## Usage

- Run the API Server: python api.py

- Fetch Live Stock Data: python update_stock_data.py

- Run Dashboard: streamit run dashboard.py

## License

This project is licensed under the MIT License.

