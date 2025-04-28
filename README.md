# Australia Weather Rain Prediction

This project predicts whether it will rain tomorrow based on weather data collected from Australia. The project involves training a machine learning model using historical weather data and deploying the model to a web application. The model predicts whether it will rain the following day based on the given weather features.

## Table of Contents

- [Overview](#overview)
- [Technologies Used](#technologies-used)
- [Project Setup](#project-setup)
- [Model Training](#model-training)
- [Web Application](#web-application)
- [Deployment](#deployment)
- [Contributing](#contributing)
- [License](#license)

## Overview

This repository contains code to:
1. **Data Collection**: Load and preprocess historical weather data from CSV files.
2. **Data Processing**: Clean and prepare the data for training the machine learning model.
3. **Model Training**: Train a machine learning model to predict whether it will rain tomorrow.
4. **Web Application**: Deploy the model into a Flask-based web application to allow users to input weather data and receive predictions.

## Technologies Used

- **Flask**: A lightweight Python web framework used for building the web application.
- **scikit-learn**: A machine learning library used to train the model.
- **Pandas**: A library for data manipulation and analysis.
- **Docker**: For containerizing the web application and model.
- **Google Cloud**: Used for hosting the application in Google Kubernetes Engine (GKE).
- **GitHub Actions**: For continuous integration and deployment (CI/CD).

## Project Setup

### Prerequisites

Before running the project, ensure you have the following installed:

- Python 3.8 or higher
- Docker
- Google Cloud SDK
- Git
- Flask

### Clone the repository

To get started, clone the repository to your local machine:

```bash
git clone https://github.com/your-username/your-repository-name.git
cd your-repository-name
