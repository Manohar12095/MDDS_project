
🥗 Nutrient Prediction from Food Packaging Images Using CNN
📌 Project Description

This project uses Convolutional Neural Networks (CNN) to analyze food packaging images and predict the nutritional health status of packaged food items. The system classifies the product into categories such as snacks, chocolates, biscuits, juices, and soft drinks, and provides a Healthy / Unhealthy recommendation.

🎯 Objective

Classify food packaging images using CNN

Predict health impact of packaged food

Provide user-friendly health recommendation

🧠 Technologies Used

Python

TensorFlow / Keras

CNN (MobileNetV2 – Transfer Learning)

Kaggle Dataset

Google Colab

Streamlit

📂 Dataset

Source: Kaggle

Dataset: Food Packaging Dataset

Link: https://www.kaggle.com/datasets/parjunwoo/fooddatasert

🧪 Model Details

Input size: 224 × 224

Epochs: 10

Optimizer: Adam

Output: Food category prediction

🌐 Web Application

The trained model is deployed using Streamlit, where users can upload a food packaging image and view:

Predicted food category

Health status (Healthy / Unhealthy)

▶ How to Run
pip install -r requirements.txt
streamlit run app.py

📁 Project Structure
food-nutrition-app/
├── app.py
├── requirements.txt
├── README.md

