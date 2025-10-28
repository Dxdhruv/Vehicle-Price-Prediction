# 🚗 Vehicle Price Prediction System

A comprehensive machine learning project that predicts vehicle prices using advanced data preprocessing techniques and ensemble learning algorithms. This system helps everyday people get accurate estimates for their vehicles' market value.

## 📋 Project Overview

This project implements an end-to-end machine learning pipeline for vehicle price prediction using the Craigslist Car Sales dataset. The system processes complex automotive data, applies sophisticated feature engineering, and delivers accurate price predictions with **94.13% accuracy** and **R² score of 0.94**.

### 🎯 Business Value
- **Consumer Empowerment**: Helps individuals make informed decisions when buying or selling vehicles
- **Market Transparency**: Provides data-driven insights into vehicle pricing trends
- **Time Efficiency**: Eliminates the need for manual price research and estimation

## 🚀 Key Features

- **Advanced Data Preprocessing**: Handles missing data intelligently without losing significant information
- **Sophisticated Feature Engineering**: Creates meaningful categorical features from raw data
- **Multiple Algorithm Testing**: Evaluated various ML algorithms to select the optimal model
- **Comprehensive Model Evaluation**: Detailed performance metrics and validation
- **Scalable Architecture**: Modular design for easy maintenance and enhancement
- **API-Ready**: FastAPI implementation for web service deployment

## 🛠️ Technical Stack

### Core Technologies
- **Python 3.x** - Primary programming language
- **Pandas** - Data manipulation and analysis
- **NumPy** - Numerical computing
- **Scikit-learn** - Machine learning algorithms and preprocessing
- **Category Encoders** - Advanced categorical encoding

### Machine Learning
- **Random Forest Regressor** - Primary prediction model
- **Target Encoding** - Advanced categorical feature encoding
- **Column Transformer** - Automated preprocessing pipeline
- **Standard Scaler** - Feature normalization

### Web Framework
- **FastAPI** - Modern, fast web framework for building APIs
- **Pydantic** - Data validation and settings management
- **Uvicorn** - ASGI server for FastAPI

### Development Tools
- **Jupyter Notebook** - Interactive development and analysis
- **Pickle** - Model serialization and persistence

## 📊 Dataset Information

- **Source**: [Craigslist Car Sales Dataset](https://www.kaggle.com/datasets/austinreese/craigslist-carstrucks-data)
- **Size**: 426,880 vehicle records
- **Features**: 26 attributes including manufacturer, model, year, condition, fuel type, odometer reading, and more
- **Time Period**: Comprehensive historical data covering multiple years
- **Data Quality**: High-quality dataset with realistic market pricing information

## 🔧 Installation & Setup

### Prerequisites
- Python 3.7 or higher
- pip package manager

### Installation Steps

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd Vehicle-Price-Prediction
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Download the dataset**
   - Visit the [Kaggle dataset page](https://www.kaggle.com/datasets/austinreese/craigslist-carstrucks-data)
   - Download `vehicles.csv.zip`
   - Extract and place `vehicles.csv` in the project directory

## 🏗️ Project Architecture

```
Vehicle-Price-Prediction/
├── preprocessing.py          # Data preprocessing and feature engineering
├── train.py                 # Model training pipeline
├── evaluate.py              # Model evaluation and metrics
├── api.py                   # FastAPI web service (ready for deployment)
├── Vehicle_Price_Prediction.ipynb  # Jupyter notebook analysis
├── requirements.txt         # Project dependencies
└── README.md               # Project documentation
```

## 🔬 Data Preprocessing & Feature Engineering

### Challenge Addressed
The dataset contained significant missing values across multiple columns. Traditional approaches of dropping missing data would have eliminated ~50% of the dataset, severely impacting model performance.

### Solution Implemented
Developed intelligent preprocessing strategies that preserve data integrity while maximizing information retention:

1. **Smart Missing Data Handling**: Implemented targeted imputation strategies for different data types
2. **Advanced Feature Engineering**: Created meaningful categorical features:
   - **Manufacturer Categories**: High/Medium/Low selling based on sales volume (>25K, 10K-25K, <10K)
   - **Vehicle Type Categories**: Popular/Medium/Less Popular based on market demand
   - **Paint Color Categories**: Popular/Less Popular based on consumer preferences

3. **Multi-Level Encoding Strategy**:
   - **Target Encoding**: For high-cardinality categorical variables (model names)
   - **Ordinal Encoding**: For ordered categorical variables (condition, cylinders)
   - **One-Hot Encoding**: For nominal categorical variables (fuel, transmission, drive)
   - **Standard Scaling**: For numerical features (year, odometer)

## 🤖 Model Development

### Algorithm Selection Process
Conducted comprehensive evaluation of multiple machine learning algorithms:

- **Linear Regression** - Baseline model
- **Ridge Regression (L2)** - Regularized linear model
- **Lasso Regression (L1)** - Feature selection with regularization
- **Decision Tree** - Interpretable single-tree model
- **Random Forest Regressor** - **Selected as optimal**
- **XGBoost** - Gradient boosting alternative

### Final Model: Random Forest Regressor
**Configuration**: 100 estimators, random state 42
**Rationale**: Superior performance in handling mixed data types, robust to outliers, and excellent generalization capability

## 📈 Performance Metrics

The model achieved exceptional performance on the test dataset:

| Metric | Value | Interpretation |
|--------|-------|----------------|
| **R² Score** | 0.94 | 94% of price variance explained |
| **Accuracy** | 94.13% | Overall prediction accuracy |
| **Mean Absolute Error** | $1,454.17 | Average prediction error |
| **Root Mean Square Error** | $2,754.13 | Standard deviation of errors |

### Performance Analysis
- **High Accuracy**: 94.13% accuracy indicates excellent predictive capability
- **Low Error Rate**: MAE of $1,454 represents less than 5% error for typical vehicle prices
- **Strong Correlation**: R² of 0.94 demonstrates the model captures 94% of price variance

## 🚀 Usage

### Training the Model
```python
from train import ModelTrainer

# Initialize trainer with dataset path
trainer = ModelTrainer("path/to/vehicles.csv")

# Train the model
trainer.train_model()
```

### Making Predictions
```python
from preprocessing import DataPreprocessor
import pickle

# Load trained model and preprocessor
with open('model.pkl', 'rb') as f:
    model = pickle.load(f)

with open('preprocessor.pkl', 'rb') as f:
    preprocessor = pickle.load(f)

# Prepare input data
vehicle_data = {
    'manufacturer': 'toyota',
    'model': 'camry',
    'year': 2020,
    'condition': 'excellent',
    'cylinders': '4 cylinders',
    'fuel': 'gas',
    'odometer': 25000,
    'title_status': 'clean',
    'transmission': 'automatic',
    'drive': 'fwd',
    'type': 'sedan',
    'paint_color': 'white'
}

# Transform and predict
df = pd.DataFrame([vehicle_data])
processed_data = preprocessor.transform(df)
predicted_price = model.predict(processed_data)[0]

print(f"Predicted Price: ${predicted_price:,.2f}")
```

### API Usage (FastAPI)
```python
# Start the API server
uvicorn api:app --reload

# Make prediction requests
curl -X POST "http://localhost:8000/predict" \
     -H "Content-Type: application/json" \
     -d '{
       "manufacturer": "toyota",
       "model": "camry",
       "year": 2020,
       "condition": "excellent",
       "cylinders": "4 cylinders",
       "fuel": "gas",
       "odometer": 25000,
       "title_status": "clean",
       "transmission": "automatic",
       "drive": "fwd",
       "type": "sedan",
       "paint_color": "white"
     }'
```

## 🔍 Technical Challenges & Solutions

### Challenge 1: Missing Data Management
**Problem**: 50%+ data loss with traditional missing data handling
**Solution**: Implemented intelligent imputation strategies preserving data integrity

### Challenge 2: High-Dimensional Categorical Data
**Problem**: Complex categorical variables with high cardinality
**Solution**: Multi-level encoding strategy combining target, ordinal, and one-hot encoding

### Challenge 3: Feature Engineering Complexity
**Problem**: Raw features insufficient for accurate predictions
**Solution**: Created domain-specific categorical features based on market insights

### Challenge 4: Model Selection
**Problem**: Multiple algorithms with varying performance
**Solution**: Systematic evaluation leading to Random Forest selection

## 🎓 Learning Outcomes & Skills Developed

### Technical Skills
- **Machine Learning**: Algorithm selection, hyperparameter tuning, model evaluation
- **Data Science**: Exploratory data analysis, feature engineering, data preprocessing
- **Python Programming**: Pandas, NumPy, Scikit-learn, object-oriented programming
- **API Development**: FastAPI, Pydantic, RESTful service design
- **Data Visualization**: Matplotlib, statistical analysis and plotting

### Soft Skills
- **Problem-Solving**: Tackled complex data quality issues
- **Analytical Thinking**: Developed domain-specific feature engineering strategies
- **Project Management**: Completed end-to-end ML pipeline in 10-15 days
- **Documentation**: Comprehensive code documentation and README creation

## 🔮 Future Enhancements

### Planned Improvements
1. **Engine Information Integration**: Add engine specifications for more accurate predictions
2. **Model Optimization**: Reduce MSE and MAE through advanced techniques
3. **Real-time Data Integration**: Connect to live market data feeds
4. **Web Application**: Develop user-friendly frontend interface
5. **Model Explainability**: Implement SHAP values for prediction interpretability

### Technical Roadmap
- **Deep Learning**: Experiment with neural networks for complex pattern recognition
- **Ensemble Methods**: Combine multiple models for improved accuracy
- **Feature Selection**: Implement automated feature importance analysis
- **Deployment**: Containerize application with Docker for scalable deployment

## 📚 Project Timeline

- **Duration**: 10-15 days
- **Phase 1**: Data exploration and preprocessing (3-4 days)
- **Phase 2**: Feature engineering and model development (4-5 days)
- **Phase 3**: Model evaluation and optimization (2-3 days)
- **Phase 4**: API development and documentation (2-3 days)

## 🤝 Contributing

This project was developed as part of an internship program and later adapted for academic purposes. Contributions and suggestions for improvements are welcome!

## 📄 License

This project is open source and available under the [MIT License](LICENSE).

## 📞 Contact

For questions about this project or collaboration opportunities, please reach out through the project repository.

---

**Note**: This project demonstrates proficiency in end-to-end machine learning development, from data preprocessing to model deployment, showcasing skills valuable for data science and machine learning engineering roles.
