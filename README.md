
-----

# AI-Powered Predictive Health Application (PranaChain)

This project is a proof-of-concept (PoC) web application developed for PranaChain's Industry-Academic Collaboration. Its purpose is to demonstrate the "AI" component of the company's "AI + Blockchain Synergy" vision by providing a user-friendly interface to run high-accuracy machine learning models for disease prediction.

The application is built in Python using **Streamlit** and hosts three separate **Scikit-learn** models (Random Forest) to predict the risk for Diabetes, Heart Disease, and Chronic Kidney Disease.

## Features

  * **Three High-Accuracy Models:** Integrates predictive models for three critical health conditions.
  * **Simple Web Interface:** A clean, user-friendly UI built with Streamlit that allows non-technical users to run predictions.
  * **File-Based Prediction:** Users can upload a text file (`.txt`) with the required patient data to receive an instant probability score.
  * **Integrated App:** All three models are hosted and accessible from a single, unified web application.

## Model Performance

All models were trained using a Random Forest algorithm and achieved high performance:

| Model | Accuracy | Precision | Recall | F1-Score |
| :--- | :--- | :--- | :--- | :--- |
| **Kidney Disease** | 98.9% | 98.0% | 99.7% | 98.8% |
| **Diabetes** | 98.6% | 99.0% | 99.0% | 99.0% |
| **Heart Disease** | 92.0% | 92.0% | 92.0% | 92.0% |

## 🛠Technical Stack

  * **Programming Language:** Python
  * **Web Framework:** Streamlit
  * **ML & Data Science:**
      * Scikit-learn (for building and running models)
      * Pandas (for data manipulation)
      * NumPy
  * **Model Persistence:** Joblib

## Setup and Installation

Follow these steps to set up and run the application on your local machine.

**Prerequisites:**

  * [Python 3.x](https://www.python.org/downloads/)
  * [Git](https://www.google.com/search?q=https://git-scm.com/downloads)

**Steps:**

1.  **Clone the repository:**

    ```bash
    git clone https://github.com/ShubhGohil/diseasePredictionModels.git
    ```

2.  **Navigate to the project directory:**

    ```bash
    cd diseasePredictionModels
    ```

3.  **Install required dependencies:**
    All necessary libraries are listed in `requirements.txt`.

    ```bash
    pip install -r requirements.txt
    ```

    *Note: The trained model files (`.joblib`) are already included in the `models/` directory.*

## How to Run the Application

1.  Ensure you are in the project's root directory (the one containing `app.py`).

2.  Run the following command in your terminal:

    ```bash
    streamlit run app.py
    ```

    \*\*

3.  Streamlit will automatically open a new tab in your default web browser, pointing to the running application (usually `http://localhost:8501`).

## Using the App

1.  Once the app is open, use the sidebar to select which predictive model you want to use:
      * Diabetes
      * Heart Disease
      * Kidney Disease
2.  On the corresponding page, click the "Browse files" button to upload a text file (`.txt`) containing the patient data.
3.  **Test files** are provided for your convenience in the `test_file/` directory.
4.  The application will process the file and display the prediction result clearly on the page.

## Project Structure

```
.
├── app.py                # The main Streamlit application file
├── models/
│   ├── ckd_model_pred.joblib
│   ├── diabetes.joblib
│   └── heart.joblib
├── code/                 # Jupyter Notebooks and .py scripts for model training
│   ├── chronic_kidney_disease/
│   ├── diabetes/
│   └── heart/
├── test_file/            # Example .txt files for testing
│   ├── test_diabetes_data.txt
│   ├── test_heart_data.txt
│   └── test_kidney_data.txt
├── requirements.txt      # List of Python dependencies
└── README.md             # This file
```

-----
