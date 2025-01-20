import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

# Load data with caching
@st.cache
def load_data():
    try:
        data = pd.read_csv("zameen-updated.csv")
        return data
    except FileNotFoundError:
        st.error("Dataset not found. Please upload the dataset to the repository.")
        return pd.DataFrame()

# Function to train the model with caching
@st.cache(allow_output_mutation=True)
def train_model(data):
    features = ['property_type', 'bedrooms', 'baths', 'Area Size', 'latitude', 'longitude']
    X = data[features]
    y = data['price']
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_scaled, y)
    return model, scaler, features

# Load and validate data
data = load_data()
if data.empty:
    st.stop()

# Train model
model, scaler, features = train_model(data)

# Dashboard title
st.title("Real Estate Price Prediction Dashboard")

# Display dataset overview
st.header("Dataset Overview")
if st.checkbox("Show raw data"):
    st.write(data.head())

# Sidebar for user input
st.sidebar.header("User Input Features")
property_type = st.sidebar.selectbox("Property Type", sorted(data['property_type'].unique()))
bedrooms = st.sidebar.slider("Number of Bedrooms", 0, int(data['bedrooms'].max()), 3)
baths = st.sidebar.slider("Number of Baths", 0, int(data['baths'].max()), 2)
area_size = st.sidebar.slider("Area Size (normalized)", 
                               float(data['Area Size'].min()), 
                               float(data['Area Size'].max()), 
                               float(data['Area Size'].mean()))
latitude = st.sidebar.slider("Latitude", 
                              float(data['latitude'].min()), 
                              float(data['latitude'].max()), 
                              float(data['latitude'].mean()))
longitude = st.sidebar.slider("Longitude", 
                               float(data['longitude'].min()), 
                               float(data['longitude'].max()), 
                               float(data['longitude'].mean()))

# Prediction
st.header("Price Prediction")
if st.button("Predict Price"):
    input_data = pd.DataFrame({
        'property_type': [property_type],
        'bedrooms': [bedrooms],
        'baths': [baths],
        'Area Size': [area_size],
        'latitude': [latitude],
        'longitude': [longitude]
    })
    input_scaled = scaler.transform(input_data)
    predicted_price = model.predict(input_scaled)[0]
    st.success(f"Predicted Price: {predicted_price:,.2f}")

# Visualizations
st.header("Data Visualizations")

# Price distribution
st.subheader("Price Distribution")
fig, ax = plt.subplots()
sns.histplot(data['price'], kde=True, ax=ax, color="blue")
st.pyplot(fig)

# Geospatial analysis
st.subheader("Geospatial Analysis")
fig, ax = plt.subplots()
sns.scatterplot(x=data['longitude'], y=data['latitude'], hue=data['price'], palette='coolwarm', ax=ax)
plt.title("Property Prices by Location")
st.pyplot(fig)

# Feature importance
st.subheader("Feature Importance (Random Forest)")
feature_importances = model.feature_importances_
fig, ax = plt.subplots()
sns.barplot(x=feature_importances, y=features, ax=ax)
plt.title("Feature Importance")
st.pyplot(fig)
