import streamlit as st
import numpy as np
import joblib
import datetime
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import img_to_array

# Load the trained PCOS detection model
pcos_model = joblib.load('/Users/karankadyan/Downloads/PCOS Detection/model/exported model/data_model.pkl')

# Load the trained image model
image_model = load_model('/Users/karankadyan/Downloads/PCOS Detection/model/exported model/image_model.keras')

# Main Title
st.title("Women's Health Assistant")
st.write("A one-stop solution for PCOS prediction, period tracking, diet planning, and image-based PCOS detection.")

# Sidebar Navigation
st.sidebar.title("Navigation")
st.sidebar.write("Use the options below to navigate:")
navigation = st.sidebar.selectbox(
    "Choose a section",
    ["🏠 Home", "🩺 PCOS Detection", "📅 Period Tracker", "🥗 Diet Plan", "🖼️ Image-based PCOS Detection"]
)

# Home Section
if navigation == "🏠 Home":
    st.header("Welcome to Women's Health Assistant")
    st.write("""
        This application provides several features:
        - 🖼️ **Image-based PCOS Detection**: Uses an image of an ultrasound to detect PCOS.
        - 🩺 **PCOS Detection**: Predicts if a person is at risk of Polycystic Ovary Syndrome (PCOS) based on medical parameters.
        - 🥗 **Diet Plan**: Suggests a personalized diet plan based on weight, activity level, and dietary preferences.
        - 📅 **Period Tracker**: Tracks menstrual cycles and predicts the next period date.
    """)
    st.image("home.jpg", caption="Women's Health Assistant", use_column_width=True)
    st.write("Select a section from the sidebar to get started.")

# PCOS Detection Section
elif navigation == "🩺 PCOS Detection":
    st.header("PCOS Disease Prediction")
    st.write("Fill in the details below to predict if a person has PCOS.")

    # User Inputs for PCOS Prediction
    age = st.number_input("Age", min_value=10, max_value=100, value=25)
    weight = st.number_input("Weight (kg)", min_value=10, max_value=200, value=64)
    height = st.number_input("Height (cm)", min_value=50, max_value=250, value=156)
    bmi = st.number_input("BMI", min_value=10.0, max_value=50.0, value=26.3)
    blood_group = st.selectbox("Blood Group", options=["A+", "A-", "B+", "B-", "O+", "O-", "AB+", "AB-"], index=0)
    pulse_rate = st.number_input("Pulse Rate (bpm)", min_value=30, max_value=200, value=70)
    rr = st.number_input("Respiratory Rate (breaths/min)", min_value=5, max_value=50, value=18)
    hb = st.number_input("Hemoglobin (g/dL)", min_value=5.0, max_value=20.0, value=11.2)
    cycle = st.number_input("Cycle Regularity (days)", min_value=0, max_value=40, value=2)
    cycle_length = st.number_input("Cycle Length (days)", min_value=0, max_value=40, value=6)
    marriage_status = st.selectbox("Marriage Status", options=["Married", "Unmarried"], index=0)
    pregnant = st.selectbox("Pregnant", options=["Yes", "No"], index=1)
    no_of_abortions = st.number_input("Number of Abortions", min_value=0, max_value=10, value=0)
    hip = st.number_input("Hip Circumference (cm)", min_value=20, max_value=200, value=39)
    waist = st.number_input("Waist Circumference (cm)", min_value=20, max_value=200, value=34)
    waist_hip_ratio = st.number_input("Waist-Hip Ratio", min_value=0.5, max_value=2.0, value=0.87)
    weight_gain = st.selectbox("Weight Gain", options=["Yes", "No"], index=1)
    hair_growth = st.selectbox("Excess Hair Growth", options=["Yes", "No"], index=1)
    skin_darkening = st.selectbox("Skin Darkening", options=["Yes", "No"], index=1)
    hair_loss = st.selectbox("Hair Loss", options=["Yes", "No"], index=1)
    pimples = st.selectbox("Pimples", options=["Yes", "No"], index=1)
    fast_food = st.selectbox("Frequent Fast Food Intake", options=["Yes", "No"], index=1)
    reg_exercise = st.selectbox("Regular Exercise", options=["Yes", "No"], index=1)
    bp_systolic = st.number_input("BP Systolic (mmHg)", min_value=50, max_value=200, value=110)
    bp_diastolic = st.number_input("BP Diastolic (mmHg)", min_value=30, max_value=150, value=80)

    # Prepare input data
    input_data = [
        age, weight, height, bmi, 0 if blood_group == "A+" else 1, pulse_rate, rr, hb,
        cycle, cycle_length, 0 if marriage_status == "Married" else 1,
        1 if pregnant == "Yes" else 0, no_of_abortions, hip, waist, waist_hip_ratio,
        1 if weight_gain == "Yes" else 0, 1 if hair_growth == "Yes" else 0,
        1 if skin_darkening == "Yes" else 0, 1 if hair_loss == "Yes" else 0,
        1 if pimples == "Yes" else 0, 1 if fast_food == "Yes" else 0,
        1 if reg_exercise == "Yes" else 0, bp_systolic, bp_diastolic
    ]
    input_data_as_numpy_array = np.asarray(input_data).reshape(1, -1)

    # Prediction Button
    if st.button("Predict"):
        prediction = pcos_model.predict(input_data_as_numpy_array)
        if prediction[0] == 0:
            st.success("The Person does not have PCOS Disease.")
        else:
            st.error("The Person has PCOS Disease.")

# Period Tracker Section
elif navigation == "📅 Period Tracker":
    st.header("Period Tracker")
    st.write("Track your periods and get predictions for the next cycle.")

    # Inputs for Period Tracking
    last_period_date = st.date_input("Last Period Date", datetime.date.today())
    average_cycle_length = st.number_input("Average Cycle Length (days)", min_value=20, max_value=40, value=28)

    # Calculate next period date
    if st.button("Track"):
        next_period_date = last_period_date + datetime.timedelta(days=average_cycle_length)
        st.success(f"Your next period is expected to start on: {next_period_date.strftime('%d %B, %Y')}")

# Diet Plan Section
elif navigation == "🥗 Diet Plan":
    st.header("PCOS Diet Plan Generator")
    st.write("Enter your weight, activity level, and dietary preferences to get a personalized PCOS-friendly diet plan.")

    weight = st.number_input("Enter your weight (kg):", min_value=30, max_value=200, value=70)
    activity_level = st.selectbox("Activity level", ["low", "moderate", "high"])
    preference = st.selectbox("Dietary preference", ["regular", "vegetarian", "vegan"])

    if st.button("Generate Diet Plan"):
        # Diet Plan Calculation Logic
        def calculate_macronutrients(calories):
            protein_grams = (calories * 0.30) / 4  # 4 calories per gram of protein
            carb_grams = (calories * 0.40) / 4    # 4 calories per gram of carbohydrates
            fat_grams = (calories * 0.30) / 9     # 9 calories per gram of fats
            return {
                "Protein (g)": round(protein_grams, 2),
                "Carbs (g)": round(carb_grams, 2),
                "Fats (g)": round(fat_grams, 2)
            }

        def get_diet_plan(weight, activity_level, preference):
            calories = weight * 10
            if activity_level == "low":
                calories *= 1.2
            elif activity_level == "moderate":
                calories *= 1.5
            elif activity_level == "high":
                calories *= 1.8

            macros = calculate_macronutrients(calories)

            diet_plan = {
                "Breakfast": "Oats with almond milk and chia seeds",
                "Snack": "Handful of nuts and a small apple",
                "Lunch": "Grilled chicken/fish with quinoa and mixed vegetables",
                "Snack": "Greek yogurt with berries",
                "Dinner": "Lentil soup with a side of roasted veggies"
            }

            return macros, diet_plan

        macros, plan = get_diet_plan(weight, activity_level, preference)
        st.write(f"Your estimated daily calorie intake: {weight * 10} kcal")
        st.write("Macronutrient Breakdown:")
        st.write(macros)
        st.write("Recommended Diet Plan:")
        for meal, recipe in plan.items():
            st.write(f"{meal}: {recipe}")

# Image-based PCOS Detection Section
elif navigation == "🖼️ Image-based PCOS Detection":
    st.header("PCOS Detection via Image (Ultrasound)")

    # Upload an image
    uploaded_image = st.file_uploader("Upload an Ultrasound Image", type=["jpg", "jpeg", "png"])

    if uploaded_image is not None:
        st.image(uploaded_image, caption="Uploaded Image", use_column_width=True)

        # Preprocess the image and make prediction
        img = image.load_img(uploaded_image, target_size=(224, 224))
        img_array = img_to_array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        prediction = image_model.predict(img_array)
        if np.argmax(prediction[0]) == 0:
            st.success("PCOS Detected!")
        else:
            st.error("No PCOS Detected!")