import streamlit as st
import joblib
import pandas as pd

# -----------------------------
# Page Configuration
# -----------------------------
st.set_page_config(
    page_title="Chennai House Price Predictor",
    page_icon="🏠",
    layout="centered"
)

# -----------------------------
# Load the Saved Files
# -----------------------------
encoder = joblib.load("location_encoder.pkl")
model = joblib.load("house_price_model.pkl")
feature_columns = joblib.load("feature_columns.pkl")

# -----------------------------
# App Title
# -----------------------------
st.set_page_config(
    page_title="Chennai House Price Predictor",
    page_icon="🏠",
    layout="centered"
)

st.title("🏠 Chennai House Price Predictor")

st.caption(
    "Estimate Chennai house prices instantly using a Machine Learning model trained on real housing data."
)

st.markdown("---")

# -----------------------------
# User Inputs
# -----------------------------
st.subheader("🏡 Enter Property Details")
col1, col2 = st.columns(2)

with col1:
    area = st.number_input(
        "Area (sq.ft)",
        min_value=100,
        max_value=10000,
        value=1000
    )

    location = st.selectbox(
        "Location",
        encoder.categories_[0]
    )

    resale = st.checkbox("Resale")

    lift = st.checkbox("Lift Available")

    car_parking = st.checkbox("Car Parking")

    security = st.checkbox("24×7 Security")

with col2:
    bedrooms = st.selectbox(
        "Bedrooms",
        [1, 2, 3, 4, 5]
    )

    gym = st.checkbox("Gymnasium")

    swimming_pool = st.checkbox("Swimming Pool")

    power_backup = st.checkbox("Power Backup")

    clubhouse = st.checkbox("Club House")

    play_area = st.checkbox("Children's Play Area")

st.markdown("---")
st.subheader("📈 Generate Prediction")
if st.button("🔍 Predict House Price"):

    # Create one empty row with all 222 features
    input_df = pd.DataFrame(0, index=[0], columns=feature_columns)

    # Fill the values entered by the user
    input_df["Area"] = area
    input_df["No. of Bedrooms"] = bedrooms
    input_df["Resale"] = int(resale)
    input_df["LiftAvailable"] = int(lift)
    input_df["Gymnasium"] = int(gym)
    input_df["SwimmingPool"] = int(swimming_pool)
    input_df["CarParking"] = int(car_parking)
    input_df["PowerBackup"] = int(power_backup)
    input_df["24X7Security"] = int(security)
    input_df["ClubHouse"] = int(clubhouse)
    input_df["Children'splayarea"] = int(play_area)

    # Encode the selected location
    location_column = "Location_" + location

    if location_column in input_df.columns:
        input_df[location_column] = 1

    # Display only the columns that have values
    # Make prediction
    prediction = model.predict(input_df)

    price = prediction[0]

    # Convert into Lakhs / Crores
    if price >= 10000000:
        display_price = f"₹ {price/10000000:.2f} Crore"
    elif price >= 100000:
        display_price = f"₹ {price/100000:.2f} Lakhs"
    else:
        display_price = f"₹ {price:,.2f}"

    st.markdown("---")

    st.subheader("🏡 Estimated House Price")

    st.metric(
        label="Predicted Value",
        value=display_price
    )

    st.info(
        "💡 This prediction is generated using a trained Random Forest Machine Learning model."
    )

st.markdown("---")

st.caption(
    "Built using Python • Streamlit • Scikit-learn • Random Forest Regressor"
)