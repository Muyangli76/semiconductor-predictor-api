# Semiconductor Pass/Fail Prediction API

Project Overview:
A FastAPI-based API focuses on predicting the yield type of semiconductor manufacturing processes using a dataset collected from various sensors and process measurement points. The goal is to identify key signals and features that contribute to yield excursions and ultimately improve process throughput, reduce production costs, and shorten time to learning. Feature selection is crucial as the dataset contains both relevant and irrelevant signals, and the task is to build an Artificial Neural Network (ANN) model to predict the yield type.

## Objectives
- Preprocess and clean the dataset.
- Build a machine learning model using an Artificial Neural Network (ANN).
- Evaluate the model’s performance using the confusion matrix.

--- 
## Demo
![image](https://github.com/user-attachments/assets/c594e441-b53e-4e22-a482-1e8ed00d4c33)

---

### 📦 Technologies Used

- **Python 3.9+**: Programming language used for model training and API development  
- **FastAPI**: Web framework for building the API  
- **TensorFlow / Keras**: Deep learning frameworks for building and training the ANN model  
- **Scikit-learn**: Library for preprocessing, model evaluation, and cross-validation  
- **Pandas / NumPy**: Libraries for data manipulation and numerical operations  
- **Uvicorn**: ASGI server for running the FastAPI application  
- **Postman**: API testing tool used to send requests and inspect responses  
- **Swagger UI**: Auto-generated interactive API documentation available at `/docs`  


---

## Features

- **Predict pass/fail status**: Based on input sensor data.
- **Feature selection**: Automatically identifies the most relevant signals contributing to the yield prediction.
- **Easy-to-use API**: Submit your data and get results with a confidence score.

---

## Installation

To run the API locally:   The API will run on http://127.0.0.1:5000.

1. **Clone the repository**:
   ```bash
   git clone https://github.com/yourusername/semiconductor-api.git
2. **Navigate to the project folder**: 
   ```bash   
   cd semiconductor-api
3. **Install dependencies**:
   ```bash   
   pip install -r requirements.txt
   ```

---

## Model Evaluation

The Artificial Neural Network (ANN) model was evaluated based on its ability to classify semiconductor parts as either "Pass" or "Fail". The following metrics were used:

### 1. **Confusion Matrix**
- **Accuracy**: 92.68%

### 2. **Classification Report**
- **Precision**: 93%  
- **Recall**: 92%  
- **F1 Score**: 92.5%

### 3. **K-fold Cross-Validation**
- **Cross-validation Accuracy**: 98.0% +/- 0.4%, indicating strong reliability and generalization across different data splits.

## API testing on Postman:
![postman_predicttest](https://github.com/user-attachments/assets/252076fe-ed0c-47b9-bc35-5031112c5cfc)

## References
Dataset: SemiconductorManufacturingProcessDataset.csv from Kaggle: https://www.kaggle.com/saurabhbagchi/fmst-semiconductor-manufacturing-project

Machine Learning Framework: TensorFlow, Scikit-learn, and other Python libraries for data analysis and modeling.

