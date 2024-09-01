
# LSTM Time Series Forecasting

This project demonstrates how to build a Long Short-Term Memory (LSTM) neural network using TensorFlow and Keras to perform time series forecasting. The model is trained on a simple sequence of data and is used to predict future values.

## Table of Contents

- [Overview](#overview)
- [Installation](#installation)
- [Data Preparation](#data-preparation)
- [Model Architecture](#model-architecture)
- [Training](#training)
- [Prediction](#prediction)
- [Visualization](#visualization)
- [Usage](#usage)
- [Contributing](#contributing)
- [License](#license)

## Overview

This repository contains code for a simple time series forecasting model using an LSTM neural network. The model is trained to predict the next values in a sequence given the previous values. The example dataset is a small sequence of integers, but the model can be adapted to any univariate time series dataset.

## Installation

To get started, clone the repository and install the necessary dependencies.

```bash
git clone https://github.com/your-username/lstm-time-series-forecasting.git
cd lstm-time-series-forecasting
pip install numpy tensorflow matplotlib
```

## Data Preparation

The data preparation involves converting the time series data into input-output pairs suitable for training an LSTM model.

```python
def prepare_data(timeseries_data, n_features):
    X, y = [], []
    for i in range(len(timeseries_data)):
        end_ix = i + n_features
        if end_ix > len(timeseries_data) - 1:
            break
        seq_x, seq_y = timeseries_data[i:end_ix], timeseries_data[end_ix]
        X.append(seq_x)
        y.append(seq_y)
    return np.array(X), np.array(y)

timeseries_data = [110, 125, 133, 146, 158, 172, 187, 196, 210]
n_steps = 3
X, y = prepare_data(timeseries_data, n_steps)
X = X.reshape((X.shape[0], X.shape[1], 1))
```

## Model Architecture

The LSTM model consists of two LSTM layers followed by a Dense layer. The model is designed to predict the next value in a sequence based on the previous values.

```python
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

model = Sequential()
model.add(LSTM(50, activation='relu', return_sequences=True, input_shape=(n_steps, 1)))
model.add(LSTM(50, activation='relu'))
model.add(Dense(1))
model.compile(optimizer='adam', loss='mse')
```

## Training

The model is trained for 300 epochs with a mean squared error (MSE) loss function.

```python
model.fit(X, y, epochs=300, verbose=0)
```

## Prediction

The model predicts the next 10 values in the sequence using the trained LSTM model.

```python
x_input = np.array([187, 196, 210])
temp_input = list(x_input)
lst_output = []
i = 0
while(i < 10):
    if (len(temp_input) > 3):
        x_input = np.array(temp_input[1:])
        x_input = x_input.reshape((1, n_steps, 1))
        yhat = model.predict(x_input, verbose=0)
        temp_input.append(yhat[0][0])
        temp_input = temp_input[1:]
        lst_output.append(yhat[0][0])
        i += 1
    else:
        x_input = x_input.reshape((1, n_steps, 1))
        yhat = model.predict(x_input, verbose=0)
        temp_input.append(yhat[0][0])
        lst_output.append(yhat[0][0])
        i += 1
```

## Visualization

The predictions are plotted alongside the original time series data for visualization.

```python
import matplotlib.pyplot as plt 

day_new = np.arange(1, 10)
day_pred = np.arange(10, 20)

plt.plot(day_new, timeseries_data)
plt.plot(day_pred, lst_output)
plt.show()
```

## Usage

1. **Run the script:** Execute the Python script to train the model and generate predictions.
2. **Visualize the output:** The script will display a plot of the original time series data and the predicted values.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.


 
