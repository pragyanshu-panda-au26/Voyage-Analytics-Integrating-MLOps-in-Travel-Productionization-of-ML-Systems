# Voyage-Analytics-Integrating-MLOps-in-Travel-Productionization-of-ML-Systems

## Overview

This repository contains a comprehensive solution for a multi-task machine learning project, including gender classification, hotel recommendation, and price prediction models. The models have been trained on an extensive dataset and are available for inference via Docker images. Additionally, the prediction models are hosted and accessible via a shared link on Google Drive.

## Dataset

The dataset used in this project encompasses multiple aspects:
- **Gender Classification**: Features that contribute to predicting the gender of a user, which may include demographics and user preferences.
- **Hotel Recommendation**: Data related to user preferences, past bookings, and ratings, utilized for suggesting hotels.
- **Price Prediction**: A model that predicts the price of hotel bookings based on various factors, including time of booking, location, and demand.

## Models

### 1. Gender Classification Model
- **Task**: Classify the gender of users based on available features.
- **Model Type**: Classification
- **Performance Metrics**: Accuracy, Precision, Recall, F1-Score

### 2. Hotel Recommendation Model
- **Task**: Recommend hotels to users based on past preferences and behavioral data.
- **Model Type**: Recommendation System
- **Performance Metrics**: Precision@K, Recall@K, Mean Average Precision (MAP)

### 3. Price Prediction Model
- **Task**: Predict the price of hotel bookings.
- **Model Type**: Regression
- **Performance Metrics**: Mean Squared Error (MSE), Mean Absolute Error (MAE), R² Score

## Docker Image

To facilitate ease of deployment and testing, all models are encapsulated within a Docker image. This allows for seamless integration into various environments, whether for development, testing, or production purposes.

### Running the Docker Image
1. **Pull the Docker Image**:
   ```bash
   docker pull <docker-image-name>
2 Run the Docker Container:
bash
Copy code
docker run -p 8080:8080 <docker-image-name>
Accessing the Models
Once the Docker container is running, the models can be accessed via the provided API endpoints:

Gender Classification: http://localhost:8080/predict/gender
Hotel Recommendation: http://localhost:8080/recommend/hotel
Price Prediction: http://localhost:8080/predict/price
Access the Prediction Models
The trained prediction models are also available for download. You can access them via the following link:

Download Prediction Models from Google Drive

License
This project is licensed under the MIT License. See the LICENSE file for more details.

Acknowledgments
We extend our gratitude to the contributors and the open-source community for providing the tools and frameworks that made this project possible.

Contact
For any inquiries or issues, please feel free to open an issue in this repository or contact the project maintainers.
