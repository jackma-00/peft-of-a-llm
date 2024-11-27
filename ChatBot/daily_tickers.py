from polygon import RESTClient
import os
import json

# Your Polygon API Key
try:
    with open("api_key.txt", "r") as file:
        api_key = file.read()
except FileNotFoundError:
    api_key = os.getenv("Polygon_API_Key")

client = RESTClient(api_key=api_key)

# List of tickers to track
tickers = ["SPY", "DIA", "QQQ", "IWM", "VXX"]


def get_tickers() -> str:
    # Collect data in a structured format
    data = []

    for ticker in tickers:
        quote = client.get_previous_close_agg(ticker=ticker)

        # Access attributes directly from the first result
        formatted_quote = {
            "ticker": quote[0].ticker,
            "open": quote[0].open,
            "high": quote[0].high,
            "low": quote[0].low,
            "close": quote[0].close,
            "volume": quote[0].volume,
            "timestamp": quote[0].timestamp,
            "vwap": quote[0].vwap,
        }
        data.append(formatted_quote)

    # Convert the data to a JSON string for LLM analysis
    return json.dumps(data, indent=4)
