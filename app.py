import os
import pickle
import pandas as pd
import streamlit as st


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Hyderabad House Price Predictor",
    page_icon="🏠",
    layout="centered"
)


# ============================================================
# TITLE & SUBTITLE
# ============================================================

st.title("🏠 Hyderabad House Price Prediction")

st.markdown(
    "Enter the property details below to estimate "
    "the house price in Hyderabad."
)


# ============================================================
# FILE PATHS HELPER
# ============================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


def get_path(filename: str) -> str:
    return os.path.join(BASE_DIR, filename)


# ============================================================
# LOAD MODEL AND COLUMNS
# ============================================================

@st.cache_resource
def load_model():
    with open(get_path("house_price_model.pkl"), "rb") as file:
        model = pickle.load(file)

    with open(get_path("model_columns.pkl"), "rb") as file:
        model_columns = pickle.load(file)

    return model, model_columns


try:
    model, model_columns = load_model()
except FileNotFoundError:
    st.error("Model files were not found.")
    st.info(
        "Make sure 'house_price_model.pkl' and 'model_columns.pkl' "
        "are present in the same folder as app.py. "
        "Run train_model.py to generate them."
    )
    st.stop()
except Exception as error:
    st.error(f"Unable to load the model: {error}")
    st.stop()


# ============================================================
# GET LOCATIONS
# ============================================================

# Method A: Check for one-hot encoded columns (e.g. Location_Gachibowli)
locations = sorted([
    column.replace("Location_", "", 1)
    for column in model_columns
    if column.startswith("Location_")
])

# Method B & C: Check for raw "location" column pipeline
if not locations and "location" in model_columns:
    locations_path = get_path("locations.pkl")

    # Method B: Load from locations.pkl if it exists
    if os.path.exists(locations_path):
        try:
            with open(locations_path, "rb") as file:
                locations = sorted(pickle.load(file))
        except Exception:
            locations = []

    # Method C: Direct extraction from the fitted OneHotEncoder in the pipeline
    if not locations and hasattr(model, "named_steps") and "preprocessor" in model.named_steps:
        try:
            encoder = model.named_steps["preprocessor"].named_transformers_["location"]
            if hasattr(encoder, "categories_"):
                locations = sorted(encoder.categories_[0].tolist())
            elif hasattr(encoder, "named_steps") and "onehot" in encoder.named_steps:
                locations = sorted(encoder.named_steps["onehot"].categories_[0].tolist())
        except Exception:
            locations = []

if not locations:
    st.error(
        "No locations were found. Run train_model.py to generate "
        "'locations.pkl' or check your model files."
    )
    st.stop()


# ============================================================
# PROPERTY DETAILS (INPUT UI)
# ============================================================

st.subheader("📋 Enter Property Details")

selected_location = st.selectbox(
    "📍 Select Location in Hyderabad",
    locations
)

area = st.number_input(
    "📐 Total Area (Sq. Ft.)",
    min_value=300,
    max_value=20000,
    value=1200,
    step=50
)

bedrooms = st.slider(
    "🛏️ Number of Bedrooms",
    min_value=1,
    max_value=10,
    value=2
)


# ============================================================
# PREDICT BUTTON & INFERENCE
# ============================================================

if st.button("💰 Calculate Predicted Price", type="primary"):
    try:
        # Create input dataframe matching model columns
        input_data = pd.DataFrame(index=[0], columns=model_columns)
        input_data = input_data.fillna(0)

        # Set numerical features
        if "area" in input_data.columns:
            input_data["area"] = float(area)

        if "bedrooms" in input_data.columns:
            input_data["bedrooms"] = int(bedrooms)

        if "area_per_bedroom" in input_data.columns:
            input_data["area_per_bedroom"] = float(area / bedrooms)

        # Set location feature (supports both pipeline and one-hot encodings)
        location_column = f"Location_{selected_location}"
        if location_column in input_data.columns:
            input_data[location_column] = 1
        elif "location" in input_data.columns:
            input_data["location"] = selected_location
        else:
            st.error("The selected location does not match the model features.")
            st.stop()

        # Run inference
        prediction = float(model.predict(input_data)[0])

        # Display results
        st.markdown("---")
        st.subheader("💰 Estimated House Price")

        if prediction >= 100:
            crores = prediction / 100
            st.success(f"### ₹ {crores:.2f} Crores")
            st.write(f"Approximately ₹ {prediction:.2f} Lakhs")
        elif prediction > 0:
            st.success(f"### ₹ {prediction:.2f} Lakhs")
        else:
            st.warning(
                "The model predicted a non-positive price. "
                "Check your training data and model inputs."
            )

        if prediction > 0:
            price_per_sqft = (prediction * 100000) / area
            st.info(f"📊 Estimated price per sq.ft: ₹ {price_per_sqft:,.0f}")

        # Summary
        st.markdown("### 🏠 Property Details")
        st.write(f"**Location:** {selected_location}")
        st.write(f"**Area:** {area:,} sq.ft")
        st.write(f"**Bedrooms:** {bedrooms}")

    except Exception as error:
        st.error(f"Prediction failed: {error}")
        st.info("Ensure the model input schema matches what train_model.py used.")


# ============================================================
# MODEL PERFORMANCE AND METRICS
# ============================================================

st.markdown("---")
st.header("📊 Model Performance and Visualizations")

metrics_path = get_path("model_metrics.pkl")

if os.path.exists(metrics_path):
    try:
        with open(metrics_path, "rb") as file:
            metrics = pickle.load(file)

        col1, col2 = st.columns(2)

        with col1:
            st.metric("Mean Absolute Error (MAE)", f"{metrics.get('mae', 0):,.2f}")
            st.metric("Root Mean Squared Error (RMSE)", f"{metrics.get('rmse', 0):,.2f}")

        with col2:
            st.metric("Mean Squared Error (MSE)", f"{metrics.get('mse', 0):,.2f}")
            st.metric("R² Score", f"{metrics.get('r2', 0):.4f}")

        st.caption(
            "MAE, MSE, and RMSE are expressed in Lakhs. "
            "An R² score closer to 1 indicates better predictive accuracy."
        )
    except Exception as error:
        st.warning(f"Could not load model metrics: {error}")
else:
    st.info("Run train_model.py to generate 'model_metrics.pkl'.")


# ============================================================
# DISPLAY SAVED GRAPHS
# ============================================================

graph_dir = get_path("graphs")

graph_details = [
    ("actual_vs_predicted.png", "Actual vs Predicted House Prices"),
    ("residual_plot.png", "Prediction Error (Residuals)"),
    ("feature_importance.png", "Feature Importance"),
    ("price_distribution.png", "House Price Distribution"),
    ("correlation_heatmap.png", "Correlation Heatmap"),
    ("covariance_heatmap.png", "Covariance Matrix Heatmap"),
    ("error_distribution.png", "Prediction Error Distribution"),
    ("location_comparison.png", "House Prices by Location"),
    ("model_comparison.png", "Model Performance Comparison")
]

available_graphs = []
for filename, title in graph_details:
    graph_path = os.path.join(graph_dir, filename)
    if os.path.exists(graph_path):
        available_graphs.append((graph_path, title))

if available_graphs:
    for index in range(0, len(available_graphs), 2):
        col1, col2 = st.columns(2)

        with col1:
            graph_path, title = available_graphs[index]
            st.subheader(title)
            st.image(graph_path, caption=title, use_container_width=True)

        if index + 1 < len(available_graphs):
            with col2:
                graph_path, title = available_graphs[index + 1]
                st.subheader(title)
                st.image(graph_path, caption=title, use_container_width=True)
else:
    st.info("No visualization graphs found. Run train_model.py to generate them.")


# ============================================================
# FOOTER
# ============================================================

st.markdown("---")
st.caption("Hyderabad House Price Predictor | Machine Learning Project")