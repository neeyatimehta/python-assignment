import requests
import pandas as pd

from datetime import datetime, timedelta
import streamlit as st


headers = {
    'x-cg-demo-api-key': 'CG-guNSEPbUbcVBmnbpsvn1Qhbg'
}

def fetch_coins():
    url = "https://api.coingecko.com/api/v3/coins/list"
    try:
        response = requests.get(url,headers=headers)
        response.raise_for_status()
        data = response.json()
        if not data:
            st.error("No data received from the API.")
            return {}
        return {coin['name']: coin['id'] for coin in data}
    except requests.exceptions.HTTPError as http_err:
        st.error(f"HTTP error occurred: {http_err}")
    except Exception as err:
        st.error(f"An error occurred: {err}")
    return {}


def fetch_historical_data(coin_id, days=365):
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days)
    url = f"https://api.coingecko.com/api/v3/coins/{coin_id}/market_chart/range?vs_currency=usd&from={start_date.timestamp()}&to={end_date.timestamp()}"
    response = requests.get(url, headers=headers)
    data = response.json()
    
    
    if 'prices' in data and data['prices']:
        prices = pd.DataFrame(data['prices'], columns=['timestamp', 'price'])
        prices['date'] = pd.to_datetime(prices['timestamp'], unit='ms')
        prices.drop('timestamp', axis=1, inplace=True)  
        return prices
    else:
        st.error("Failed to fetch price data. Please try again.")
        return pd.DataFrame(columns=['date', 'price'])  



