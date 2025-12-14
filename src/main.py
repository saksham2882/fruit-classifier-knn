import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import KNeighborsClassifier
import streamlit as st
import joblib
import os
import datetime
import warnings

# ignore warnings to keep the app interface clean.
warnings.filterwarnings("ignore")   

# --- Streamlit Config ---
st.set_page_config(page_title="Fruit Classifier AI", page_icon="🍎", layout="centered")


# --- Constants ---
# use absolute paths to ensure the app works regardless of where the command is run from.
BASE_DIR = os.path.dirname(os.path.abspath(__file__)) # The directory this file is in (src)
PROJECT_ROOT = os.path.dirname(BASE_DIR)              # The parent directory
DATA_PATH = os.path.join(PROJECT_ROOT, 'data', 'fruits_data_with_colors.csv')
MODEL_DIR = os.path.join(PROJECT_ROOT, 'models')
MODEL_PATH = os.path.join(MODEL_DIR, 'fruit_knn_model.pkl')
SCALER_PATH = os.path.join(MODEL_DIR, 'scaler.pkl')

current_year = datetime.datetime.now().year


# --- Model Loading/Training Logic ---
# @st.cache_resource is a decorator. It tells Streamlit:
# "Run this function ONCE and save the result in memory. Don't run it again unless the code changes."
# This makes the app super fast because we don't retrain the model on every button click.
@st.cache_resource
def load_or_train_model():
    """
    Main Logic:
    1. Try to load an existing smart brain (model) and its eyewear (scaler).
    2. If not found, create a fresh brain, teach it (train), size it (scale), and save it.
    """
    
    # 1. Check if we already have a trained model saved
    if os.path.exists(MODEL_PATH) and os.path.exists(SCALER_PATH):
        try:
            # Load the Model
            model = joblib.load(MODEL_PATH)
            # Load the Scaler
            scaler = joblib.load(SCALER_PATH)
            
            # We also load the dataset because we need it for the scatter plot visualization later.
            dataset = pd.read_csv(DATA_PATH)
            
            return model, scaler, None, dataset 

        except Exception as e:
            st.error(f"Error loading model: {e}")
            # If loading fails, we just continue down to train a new one.
    

    # --- TRAINING PHASE (If no model exists) ---

    # 2. Load the Dataset
    if not os.path.exists(DATA_PATH):
        st.error(f"Dataset not found at {DATA_PATH}! Please check your files.")
        st.stop()
        
    dataset = pd.read_csv(DATA_PATH)
    
    # 3. separate Features and Target
    # 'X' is what we give the model (Input: Dimensions/Color)
    X = dataset[['mass', 'width', 'height', 'color_score']]
    # 'y' is what we want the model to predict (Output: Fruit Name)
    y = dataset['fruit_name']
    

    # 4. Split Data (Training vs Testing)
    # Training Data: 75% of the data and Testing Data: 25% of the data
    # random_state=0 ensures we get the same shuffling every time (reproducibility).
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)
    

    # 5. Scaling (CRITICAL STEP FOR KNN)
    # Why? 
    #   - Mass is in hundreds (150g). Width is in ones (7cm).
    #   - KNN calculates "Distance". A difference of 5g in Mass looks huge (5 units), while a difference of 2cm in Width looks small (2 units).
    #   - Code would think Mass is more important.
    #   - Scaling shrinks ALL numbers to be between 0 and 1 so they are equally important.
    scaler = MinMaxScaler()
    
    # fit_transform: "Learn the range (min/max) of the training data AND scale it."
    X_train_scaled = scaler.fit_transform(X_train)
    
    # transform: "Use the SAME range you learned from training to scale the test data."
    X_test_scaled = scaler.transform(X_test)
    

    # 6. Train the KNN Model
    # K-Nearest Neighbors (KNN) works by looking at the 'k' closest points.
    # n_neighbors=5: logical choice, checks 5 closest similar fruits.
    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(X_train_scaled, y_train) # This is where the learning happens!
    

    # 7. Check Accuracy
    accuracy = knn.score(X_test_scaled, y_test)

    # 8. Save for Future Use
    os.makedirs(MODEL_DIR, exist_ok=True)      # Ensure folder exist
    joblib.dump(knn, MODEL_PATH)               # Save the model
    joblib.dump(scaler, SCALER_PATH)           # Save the Scaler (Must save this to scale user input later!)
    
    return knn, scaler, accuracy, dataset


# Initialize the app by loading logic
model, scaler, train_acc, dataset = load_or_train_model()


# --- UI Layout ---
st.title("🍎 Fruit Classifier AI")
st.write("""
This Machine Learning model estimates the type of fruit based on its physical properties. It uses the **K-Nearest Neighbors (KNN)** algorithm and is trained on the **Fruits with Colors Dataset**.
""")

st.write("---")
st.header("📝 Enter Fruit Properties")
st.info("Measure your fruit and enter the details below:")

st.write("---")
# Columns allow us to place inputs side-by-side for a cleaner look
col1, col2 = st.columns(2)

with col1:
    mass = st.number_input("Weight (Mass)", min_value=0.0, max_value=1000.0, value=150.0, step=1.0, help="Weight of the fruit in grams.")
    width = st.number_input("Width", min_value=0.0, max_value=20.0, value=7.5, step=0.1, help="Widest part of the fruit in cm.")

with col2:
    height = st.number_input("Height", min_value=0.0, max_value=20.0, value=7.5, step=0.1, help="Tallest part of the fruit in cm.")
    color_score = st.slider("Color Score", min_value=0.0, max_value=1.0, value=0.75, step=0.01, help="A value from 0.0 to 1.0 representing the fruit's color spectrum.")


st.write("---")

if st.button("🔍 Identify Fruit", type="primary", use_container_width=True):
    # 1. Prepare Input
    # We must provide the data in the exact same format as training: [[mass, width, height, color_score]]. It must be a 2D list (list of lists) because the model expects a batch of inputs.
    input_data = [[mass, width, height, color_score]]
    
    # 2. Scale Input
    # [CRITICAL]: We trained on numbers 0-1. If we feed raw numbers (like 150), the model will be extremely confused and give wrong answers.
    # We use the SAVED scaler to shrink the user's input exactly how we shrank the training data.
    input_scaled = scaler.transform(input_data)
    
    # 3. Predict
    prediction = model.predict(input_scaled)
    result = prediction[0]            # Get the first (and only) result
    
    # 4. Display Result
    st.balloons()                     # Fun animation
    st.success(f"### The AI thinks this is a: **{result.upper()}**")
    
    # 5. Confidence Levels
    # predict_proba gives the percentage chance for each fruit type (e.g., 80% Apple, 20% Orange)
    probabilities = model.predict_proba(input_scaled)[0]
    st.write("#### Confidence Levels:")
    
    # Get class names from the model (Apple, Mandarin, Orange, Lemon)
    classes = model.classes_
    
    # Show as a bar chart
    prob_df = pd.DataFrame({
        'Fruit': classes,
        'Confidence': probabilities
    })
    st.bar_chart(prob_df.set_index('Fruit'))


    # 6. Visualization with Matplotlib
    # This helps the user "see" "why" the model made its decision.
    # We plot all known fruits and then place the user's fruit on the map.
    st.write("---")
    st.write("### 📊 Visualizing the Prediction")
    st.write("See where your fruit stands among the known samples (Width vs Height).")
    
    # Create Matplotlib Figure
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Scatter plot of all known fruits from the dataset
    fruit_types = dataset['fruit_name'].unique()
    
    for fruit in fruit_types:
        # Filter data for this specific fruit
        subset = dataset[dataset['fruit_name'] == fruit]
        # Plot its points
        ax.scatter(subset['width'], subset['height'], label=fruit, alpha=0.6, s=100, edgecolors='k')

    # Plot the User's Fruit as a big Black Star
    ax.scatter([width], [height], color='black', marker='*', s=300, label='Your Fruit', zorder=10)
    
    # Labels make the graph readable
    ax.set_xlabel('Width (cm)')
    ax.set_ylabel('Height (cm)')
    ax.set_title('Fruit Distribution: Width vs Height')
    ax.legend()
    ax.grid(True, linestyle='--', alpha=0.5)
    
    st.pyplot(fig)         # Show the plot in Streamlit


# --- Footer ---
st.write("---")
with st.expander("ℹ️ How does this work?"):
    st.write("""
    1.  **Scaling**: First, we shrink your measurements to a 0-1 scale so 'Mass' (150) doesn't overshadow 'Width' (7.5).
    2.  **Distance Calculation**: The AI looks at the 5 closest fruits in its memory (Nearest Neighbors).
    3.  **Voting**: If 3 neighbors are Apples and 2 are Oranges, it votes 'Apple'.
    """)


st.markdown("---")
st.markdown(
    f"""
    <div style="text-align: center; font-size: 14px; color: gray;">
        © {current_year} Salary Prediction App. All rights reserved. <br>
        Made with ❤️ by <a style="text-decoration: none;" href="https://saksham-agrahari.vercel.app" target="_blank">Saksham Agrahari</a>
    </div>
    """,
    unsafe_allow_html=True
)