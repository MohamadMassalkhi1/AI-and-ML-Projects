# AI and ML Projects

This repository contains various Artificial Intelligence (AI) and Machine Learning (ML) projects that I am working on. Each project explores different aspects of AI and ML, from basic neural networks to applied AI in healthcare, finance, and traffic management.

## Project Overview

### 1. CNN on MNIST Dataset
- **File**: `CNN_MNIST.ipynb`
- **Description**: This project demonstrates the implementation of a Convolutional Neural Network (CNN) to classify handwritten digits using the MNIST dataset. The model is trained and evaluated to achieve high accuracy in digit recognition.
- **Key Features**:
  - Data preprocessing and augmentation
  - CNN architecture with multiple convolutional layers
  - Evaluation metrics such as accuracy and confusion matrix

---

### 2. Basic Neural Network Model
- **File**: `Basic_Neural_Network_Model.ipynb`
- **Description**: This project involves building a simple feedforward neural network for classification tasks. It serves as a fundamental introduction to neural networks and their implementation.
- **Key Features**:
  - Implementation of a fully connected neural network
  - Backpropagation algorithm
  - Hyperparameter tuning

---

### 3. Brain Tumor Detection
- **Objective**: Develop a deep learning model capable of identifying brain tumors from MRI images. The project uses Flask to provide a user-friendly interface for uploading images and visualizing predictions.
- **Workflow**:
  1. **Data Preprocessing**: The dataset is cleaned, resized, and normalized for input to the model.
  2. **Model Building**: A Convolutional Neural Network (CNN) was built using TensorFlow/Keras to classify images as tumor or non-tumor.
  3. **Deployment**: The model is integrated into a Flask app for an easy-to-use interface. Users can upload MRI images and get real-time predictions.
- **Files & Folders**:
  - `Brain Tumor Data Set/`: Contains the MRI images for training and testing.
  - `static/` & `templates/`: Flask assets for the web interface.
  - `app.py`: The Flask app.
  - `brain_tumor_model.h5`: Saved trained model for deployment.
- **How to Run**:
  1. Clone the repository.
  2. Install dependencies:
     ```bash
     pip install -r requirements.txt
     ```
  3. Run the app:
     ```bash
     python app.py
     ```
  4. Open `http://127.0.0.1:5000` in your browser to interact with the app.

---

### 4. Stock Prediction using LSTM
- **Objective**: Train an LSTM model to predict stock price trends using historical data. A Flask app serves as the front-end for users to view predictions.
- **Workflow**:
  1. **Data Collection**: Historical stock data is collected from a public API or dataset.
  2. **Model Training**: An LSTM (Long Short-Term Memory) model is used to learn time-series patterns from the stock data.
  3. **Deployment**: Flask is used to deploy the model, where users can input stock details and view predicted trends.
- **Files & Folders**:
  - `static/` & `templates/`: Flask assets for UI.
  - `app.py`: The Flask app for user interaction.
  - `stock_dl_model.h5`: Saved trained LSTM model.
- **How to Run**:
  1. Clone the repository.
  2. Install dependencies:
     ```bash
     pip install -r requirements.txt
     ```
  3. Run the app:
     ```bash
     python app.py
     ```
  4. Open `http://127.0.0.1:5000` to use the app.

---

### 5. Azure End-to-End Data Engineering Project
- **Objective**: Create a data pipeline leveraging Azure services to process, transform, and visualize data.
- **Workflow**:
  1. **Data Ingestion**: Data is ingested from an on-prem SQL database using Azure Data Factory.
  2. **Data Processing**: Data is staged in Azure Data Lake and processed through Bronze, Silver, and Gold layers using Azure Databricks.
  3. **Analytics and Visualization**: Azure Synapse Analytics is used to query the Gold layer, with insights visualized in Power BI.
  4. **Security and Governance**: Azure Key Vault and Azure Active Directory are used to secure access to the data.
- **Key Components**:
  - Azure Data Factory
  - Azure Data Lake (Bronze, Silver, Gold layers)
  - Azure Databricks
  - Azure Synapse Analytics
  - Power BI
- **How to Run**:
  1. Set up Azure Data Factory, Data Lake, and Databricks following the provided architecture.
  2. Configure Synapse Analytics to connect to the Gold layer.
  3. Use Power BI to visualize insights.

---

### 6. Traffic Management System (In Progress)
- **File**: `Traffic_Management_System.ipynb`
- **Description**: This project focuses on developing an intelligent traffic management system using AI techniques. The system aims to optimize traffic flow and reduce congestion in urban areas.
- **Key Features**:
  - Real-time traffic data analysis
  - Predictive modeling for traffic flow
  - Simulation of traffic scenarios and evaluation

---

## Getting Started

To get started with any of these projects:
1. Clone the repository:
   ```bash
   git clone https://github.com/MohamadMassalkhi1/AI-and-ML-Projects.git
