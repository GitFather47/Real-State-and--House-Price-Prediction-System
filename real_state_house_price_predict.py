import streamlit as st
import pandas as pd
import xgboost as xgb
from sklearn.preprocessing import LabelEncoder

# Home Page: House Price Prediction
def home_page():
    st.title("Real Estate & House Price Prediction in Bangladesh")
    st.image('image.webp', use_container_width=True)  # Update with your image path

    # Load city and location data from the text file  
    @st.cache_data
    def load_city_location_data():
        # Load the city and location data
        data = pd.read_csv("city_and_location.txt", sep="\t")
        # Clean and convert the price column to numeric
        data['Price_in_taka'] = pd.to_numeric(data['Price_in_taka'].str.replace('à§³', '').str.replace(',', ''), errors='coerce')
        return data

    city_location_data = load_city_location_data()

    # Load pre-trained model
    @st.cache_resource
    def load_model():
        model = xgb.XGBRegressor()
        model.load_model("real_state_house_price_bd_model.json")  # Load your trained model
        return model

    model = load_model()

    # Sidebar for user input features for prediction
    st.sidebar.header('Input Features')

    # City selection
    selected_city = st.sidebar.selectbox("Select City", city_location_data['City'].unique())

    # Filter locations based on the selected city
    available_locations = city_location_data[city_location_data['City'] == selected_city]['Location'].unique()

    # Location selection
    selected_location = st.sidebar.selectbox("Select Location", available_locations)

    # User inputs for prediction (flexible for any user input)
    bedrooms_input = st.sidebar.number_input("Input Number of Bedrooms", min_value=1, value=3, step=1)
    bathrooms_input = st.sidebar.number_input("Input Number of Bathrooms", min_value=1, value=2, step=1)
    floor_area_input = st.sidebar.number_input("Input Floor Area (sq ft)", min_value=1.0, value=1200.0, step=1.0)
    floor_no_input = st.sidebar.number_input("Input Floor Number", min_value=1, value=3, step=1)

    # Prepare input data for prediction
    input_data = pd.DataFrame({
        'Bedrooms': [bedrooms_input],
        'Bathrooms': [bathrooms_input],
        'Floor_area': [floor_area_input],
        'Floor_no': [floor_no_input],
        'City': [selected_city],
        'Location': [selected_location]
    })

    # Prediction
    if st.button("Predict Price"):
        # Initialize LabelEncoder for City and Location
        label_encoder_city = LabelEncoder()
        label_encoder_location = LabelEncoder()

        # Fit the label encoder on the entire dataset for City and Location
        label_encoder_city.fit(city_location_data['City'])
        label_encoder_location.fit(city_location_data['Location'])

        # Apply the transformations to the input data
        input_data['City'] = label_encoder_city.transform(input_data['City'])
        input_data['Location'] = label_encoder_location.transform(input_data['Location'])

        # Make prediction on the input data using the pre-trained model
        prediction = model.predict(input_data)


        st.write(f"Predicted House Price: {prediction[0]:,.2f} BDT")


# About Page: Information and Dataset Link
def about_page(): 
    st.title("About")
    st.write("This project is a Real Estate House Price Prediction system built using machine learning techniques to predict the price of houses in Bangladesh. It utilizes a trained XGBoost model, alongside features such as city, location, floor area, and number of rooms to make accurate predictions based on historical data.")
    
    st.write("#### Credits:")
    image_path = "about.jpg"  # Update this to the correct image path if needed
    st.image(image_path)
    
    st.write("Arnob Aich Anurag")
    st.write("Research Intern at AMIR Lab (Advanced Machine Intelligence Research Lab)")
    st.write("Student at American International University Bangladesh")
    st.write("Dhaka, Bangladesh")
    
    st.write("For more information, please contact me at my email.")
    st.write("Email: aicharnob@gmail.com")

    st.write("#### Dataset:")
    st.write("The dataset used for this project is the **Real Estate & House Price Trends in Bangladesh** dataset, which can be found on Kaggle:")
    st.write("[Real Estate & House Price Trends in Bangladesh](https://www.kaggle.com/datasets/durjoychandrapaul/house-price-bangladesh)")

# Navigation Sidebar
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Home", "About"])

if page == "Home":
    home_page()
elif page == "About":
    about_page()
