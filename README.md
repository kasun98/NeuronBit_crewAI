# NeuronBit: Bitcoin Price Direction Prediction

NeuronBit is a deep learning project designed to predict the direction of Bitcoin prices. It uses a fully connected neural network (DNN) with LeakyReLU activation functions. The model is trained to determine whether the next day's Bitcoin opening price will be higher or lower than today's opening price. The project also includes a Flask web application to serve predictions.

## Table of Contents

- [Project Overview](#project-overview)
- [Model Architecture](#model-architecture)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)


## Project Overview

NeuronBit uses a fully connected neural network to predict the direction of Bitcoin prices. The model is trained on historical Bitcoin data with 26 features to determine if the next day's opening price will be higher or lower than the current day's opening price. The project also includes a Flask web application that allows users to interact with the model and get real-time predictions.

## Model Architecture

The architecture of the neural network is as follows:

- **Input Layer**: 26 features
- **Dense Layer 1**: 256 units, LeakyReLU activation
- **Dense Layer 2**: 128 units, LeakyReLU activation
- **Dense Layer 3**: 64 units, LeakyReLU activation
- **Dense Layer 4**: 32 units, LeakyReLU activation
- **Dense Layer 5**: 16 units, LeakyReLU activation
- **Dense Layer 6**: 8 units, LeakyReLU activation
- **Output Layer**: 1 unit, sigmoid activation

  ![fcnn](https://github.com/kasun98/btcprice/assets/63708260/505cdf77-beb1-4b57-a012-587f1dc03c84)


## Features

- **Prediction**: Predict whether the next day's Bitcoin opening price will be higher or lower.
- **Flask Web App**: User-friendly web interface for interacting with the model.

## Installation

To get started with NeuronBit, follow these steps:
1. **Clone the repository**:
   ```bash
   git clone https://github.com/kasun98/btcprice.git
2. **Create and activate a virtual environment**:
   ```bash
   -m venv venv
   source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
3. **Install the required dependencies**:
   ```bash
   pip install -r requirements.txt
4. **Set up environment variables**:
   Create a .env file in the root directory and add necessary environment variables.

## Usage

Once the Flask application is running, you can access the web app in your browser at http://127.0.0.1:5000/.

