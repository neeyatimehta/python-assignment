import streamlit as st
import requests
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import numpy as np
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.utils import to_categorical
from PIL import Image


headers = {
    'x-cg-demo-api-key': 'CG-guNSEPbUbcVBmnbpsvn1Qhbg'
}

# Load and preprocess data
def load_data():
    (train_images, train_labels), _ = mnist.load_data()
    train_images = train_images.reshape((60000, 28, 28, 1)).astype('float32') / 255  # Normalize
    train_labels = to_categorical(train_labels)
    return train_images[:1000], train_labels[:1000]  # Use a subset for quick training

# Define the model
def create_model():
    model = Sequential([
        Flatten(input_shape=(28, 28, 1)),
        Dense(128, activation='relu'),
        Dense(10, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

# Train the model
def train_model(model, train_images, train_labels):
    model.fit(train_images, train_labels, epochs=5, batch_size=64)

# Preprocess uploaded image for prediction
def preprocess_image(img, target_size):
    if img.mode != "RGB":
        img = img.convert("RGB")
    img = img.resize(target_size).convert('L')  # Convert to grayscale
    img_array = np.array(img)
    img_array = img_array.reshape((1, 28, 28, 1)).astype('float32') / 255  # Normalize
    return img_array

# Predict digit from image
def predict_digit(model, img):
    prediction = model.predict(img)
    return np.argmax(prediction), np.max(prediction)




# Function to fetch the list of all cryptocurrencies
def fetch_coins():
    url = "https://api.coingecko.com/api/v3/coins/list"
    try:
        response = requests.get(url,headers=headers)
        response.raise_for_status()  # This will raise an exception for HTTP errors
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


# Function to fetch historical price data for a given cryptocurrency
def fetch_historical_data(coin_id, days=365):
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days)
    url = f"https://api.coingecko.com/api/v3/coins/{coin_id}/market_chart/range?vs_currency=usd&from={start_date.timestamp()}&to={end_date.timestamp()}"
    response = requests.get(url, headers=headers)
    data = response.json()
    
    # Check if 'prices' key exists in the response
    if 'prices' in data and data['prices']:
        prices = pd.DataFrame(data['prices'], columns=['timestamp', 'price'])
        prices['date'] = pd.to_datetime(prices['timestamp'], unit='ms')
        prices.drop('timestamp', axis=1, inplace=True)  # Clean up the DataFrame
        return prices
    else:
        st.error("Failed to fetch price data. Please try again.")
        return pd.DataFrame(columns=['date', 'price'])  # Return an empty DataFrame to avoid further errors



def stock_detail():
    st.title('Stock Detail')
    coins = fetch_coins()
    coin_list = list(coins.keys())
    selected_coin = st.selectbox('Select a cryptocurrency', options=coin_list)

    if selected_coin:
        coin_id = coins[selected_coin]
        prices = fetch_historical_data(coin_id)
        
        # Plotting
        fig, ax = plt.subplots()
        ax.plot(prices['date'], prices['price'])
        ax.set_xlabel('Date')
        ax.set_ylabel('Price in USD')
        ax.set_title(f"{selected_coin} Price Over the Last Year")
        st.pyplot(fig)

        # Display max and min prices
        max_price = prices['price'].max()
        min_price = prices['price'].min()
        max_date = prices[prices['price'] == max_price]['date'].dt.strftime('%Y-%m-%d').values[0]
        min_date = prices[prices['price'] == min_price]['date'].dt.strftime('%Y-%m-%d').values[0]

        st.write(f"Maximum Price: ${max_price} on {max_date}")
        st.write(f"Minimum Price: ${min_price} on {min_date}")

def coin_cmp():
    st.title('Coin Comparission')
    coins = fetch_coins()
    coin_list = list(coins.keys())
    
    selected_coin1 = st.selectbox('Select the first cryptocurrency', options=coin_list, index=coin_list.index('Bitcoin') if 'Bitcoin' in coin_list else 0)
    selected_coin2 = st.selectbox('Select the second cryptocurrency', options=coin_list, index=coin_list.index('Ethereum') if 'Ethereum' in coin_list else 1)
    
    time_frames = {
        '1 week': 7,
        '1 month': 30,
        '1 year': 365,
        '5 years': 1825
    }
    selected_time_frame = st.selectbox('Select a time frame', options=list(time_frames.keys()))

    if st.button('Compare'):
        days = time_frames[selected_time_frame]
        
        coin1_data = fetch_historical_data(coins[selected_coin1], days)
        coin2_data = fetch_historical_data(coins[selected_coin2], days)
        
        # Plotting
        plt.figure(figsize=(10, 6))
        plt.plot(coin1_data['date'], coin1_data['price'], label=selected_coin1)
        plt.plot(coin2_data['date'], coin2_data['price'], label=selected_coin2)
        plt.xlabel('Date')
        plt.ylabel('Price in USD')
        plt.title(f"{selected_coin1} vs {selected_coin2} - Last {selected_time_frame}")
        plt.legend()
        st.pyplot(plt)

def img_classifier():
    st.title("Digit Image Classifier")

    # Load and prepare the data
    train_images, train_labels = load_data()

    # Create and train the model
    model = create_model()
    train_model(model, train_images, train_labels)

    uploaded_file = st.file_uploader("Choose an image of a digit to classify", type=["png", "jpg", "jpeg"])
    
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption='Uploaded Image', use_column_width=True)
        
        preprocessed_image = preprocess_image(image, (28, 28))
        label, confidence = predict_digit(model, preprocessed_image)
        
        st.write(f"Predicted Digit: {label} with confidence {confidence:.2f}")

# Use Streamlit's sidebar to navigate between pages

if __name__ == "__main__":
    st.sidebar.title('Navigation')
    page = st.sidebar.radio("Go to", ('Stock Detail', 'Coin Comparission', 'Image Classifier'))

    if page == 'Stock Detail':
        stock_detail()
    elif page == 'Coin Comparission':
        coin_cmp()
    elif page == 'Image Classifier':
        img_classifier()
