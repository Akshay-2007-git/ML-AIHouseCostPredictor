# 🏡 Hyderabad House Price Prediction

A Machine Learning project that predicts residential property prices in Hyderabad based on property features such as location, number of bedrooms, and area.

The project explores real-estate data, performs data preprocessing and exploratory data analysis (EDA), trains a regression model, evaluates its performance, and provides a foundation for interactive house price predictions.

## 📌 Project Overview

House prices vary depending on location, property size, and other factors. This project applies machine learning techniques to analyze housing data and estimate property prices using historical data.

**Objectives**
- Analyze and preprocess Hyderabad house price data.
- Explore relationships between property features and prices.
- Train and evaluate a regression model.
- Generate visualizations to understand the dataset and model predictions.
- Develop an interactive application using Streamlit.

## ✨ Features

- 📊 **Exploratory Data Analysis:** Understand distributions, trends, and relationships in the dataset.
- 🧹 **Data Preprocessing:** Clean the dataset and prepare features for model training.
- 🤖 **Machine Learning:** Train a regression model for house price prediction.
- 📈 **Model Evaluation:** Evaluate predictions using regression metrics.
- 📉 **Data Visualization:** Generate graphs to compare actual and predicted prices and analyze location-wise price trends.
- 🖥️ **Interactive Application:** Use Streamlit to build a user-facing prediction interface.
- 💾 **Model Persistence:** Save trained models and supporting files for reuse.

## 🛠️ Tech Stack

| Technology | Purpose |
|---|---|
| Python | Core programming language |
| Pandas | Data manipulation and preprocessing |
| NumPy | Numerical computations |
| Matplotlib | Data visualization |
| Seaborn | Statistical visualization |
| Scikit-learn | Machine learning and model evaluation |
| Streamlit | Interactive web application |
| Pickle | Saving and loading trained model artifacts |
| Git & GitHub | Version control and project hosting |

## 📂 Project Structure

```text
ML-AIHouseCostPredictor/
│
├── app.py                    # Streamlit application
├── train_model.py            # Model training and evaluation
├── price.json                # Supporting price/location data
├── house_price_model.pkl     # Trained model
├── model_columns.pkl         # Model feature information
├── locations.pkl             # Saved location information
├── model_metrics.pkl         # Model evaluation metrics
├── graphs/                   # Generated visualizations
├── README.md                 # Project documentation
│
├── House_Price_Prediction (3).pptx
├── House_Price_Prediction_ML_Documentation.docx
└── PBL_HousePricePrediction_rev2.pdf
```

*The structure above describes the intended project organization; some files may differ depending on your local setup.*

## ⚙️ Installation and Setup

### 1. Clone the repository

```bash
git clone https://github.com/Akshay-2007-git/ML-AIHouseCostPredictor.git
cd ML-AIHouseCostPredictor
```

### 2. Create a virtual environment

**Windows:**

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

If PowerShell blocks activation, use Command Prompt:

```cmd
venv\Scripts\activate.bat
```

### 3. Install dependencies

```bash
pip install pandas numpy matplotlib seaborn scikit-learn streamlit
```

### 4. Train the model

```bash
python train_model.py
```

This runs the training script, evaluates the model, and generates the visualizations configured in the project.

### 5. Launch the application

```bash
streamlit run app.py
```

Open the local URL displayed in the terminal to access the application.

**Note:** The trained model and supporting files must exist at the paths expected by the application. Run the training script first if those artifacts need to be generated.

## 📊 Model Evaluation

The project uses regression evaluation metrics to measure prediction performance.

| Metric | Description |
|---|---|
| MAE | Mean Absolute Error |
| MSE | Mean Squared Error |
| RMSE | Root Mean Squared Error |
| R² Score | Proportion of variance explained by the model |
| MAPE | Mean Absolute Percentage Error |

These metrics help assess prediction errors and identify opportunities to improve the model.

## 📉 Visualizations

The project generates visualizations to explore housing data and evaluate model behavior, including:

- Actual vs. predicted house prices
- House price distribution
- Location-wise price comparisons
- Relationships between numerical features
- Additional training and evaluation graphs

Visualizations help identify data patterns, potential outliers, and limitations in the model.

## 🚀 Future Improvements

- Improve prediction accuracy through feature engineering and model comparison.
- Evaluate advanced regression algorithms and hyperparameter tuning.
- Incorporate additional property features and more comprehensive datasets.
- Improve handling of unseen locations and missing values.
- Deploy the application to a cloud hosting platform.
- Add automated testing and model versioning.

## 🎓 Project Context

Developed as a machine learning project focused on applying data preprocessing, regression, model evaluation, and application development to a real-world real-estate prediction problem.

## 👨‍💻 Author

**Akshay Chandra Abburi**

- GitHub: [Akshay-2007-git](https://github.com/Akshay-2007-git)

## 📄 License

This project is intended for educational and learning purposes. Add a formal open-source license if you decide to distribute the project under specific reuse terms.
