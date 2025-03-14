import pandas as pd
import numpy as np
import PricePredictionNet
import torch
import torch.optim as optim

file_path = 'data/BTC.json'

# Load data
# [[timestamp, open, close, low, high, volume, number of trades]]
data = pd.read_json(file_path)

# Convert to numpy arrays
data = np.array(data)
data = data[:, 1:5]  # Only keep OHLC
data = data.astype(float)
candles = []
for row in data:
    candles.append({
        'open': row[0],
        'high': row[1],
        'low': row[2],
        'close': row[3]
    })

# Initialize model and optimizer
model = PricePredictionNet.PricePredictionNet()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
for epoch in range(100):  # Number of epochs
    for i in range(len(candles) - 4):  # Loop through the data, need 3 days + 1 target day
        # Get input data from 3 days
        input_data = PricePredictionNet.prepare_input_data(candles[i:i+3])
        
        # Calculate specific deltas for lambda calculations
        high_deltas = []
        close_deltas = []
        low_deltas = []
        closes = []
        
        # Use the last 3 days of data to calculate lambdas
        for j in range(i, i+3):
            if j > 0:  # Skip first day for deltas that need previous day
                high_deltas.append(candles[j]['high'] - candles[j]['close'])
                close_deltas.append(candles[j]['close'] - candles[j-1]['close'])
                low_deltas.append(candles[j]['close'] - candles[j]['low'])
                closes.append(candles[j-1]['close'])
        
        # Calculate lambdas using the calculate_lambda function
        lambda_h = PricePredictionNet.calculate_lambda(high_deltas, closes)
        lambda_c = PricePredictionNet.calculate_lambda(close_deltas, closes)
        lambda_l = PricePredictionNet.calculate_lambda(low_deltas, closes)
        
        # Prepare target values using the next day's data
        targets = PricePredictionNet.prepare_target_data(
            next_candle=candles[i+3],     # i+1 day
            current_candle=candles[i+2],   # i day
            lambda_h=lambda_h,
            lambda_c=lambda_c,
            lambda_l=lambda_l
        )
        
        # Train the model
        loss = PricePredictionNet.train_step(model, optimizer, input_data, targets)
        
    print(f'Epoch {epoch+1}, Loss: {loss}')
