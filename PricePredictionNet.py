import torch
import torch.nn as nn
import numpy as np

class PricePredictionNet(nn.Module):
    def __init__(self):
        super(PricePredictionNet, self).__init__()
        
        # Layer definitions
        self.hidden_layer = nn.Linear(10, 8)  # 10 inputs -> 8 hidden nodes
        self.output_layer = nn.Linear(8, 3)   # 8 hidden nodes -> 3 outputs
        
        # Activation function
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        # Forward pass through the network
        x = self.hidden_layer(x)
        x = self.sigmoid(x)
        x = self.output_layer(x)
        x = self.sigmoid(x)
        return x

def calculate_lambda(deltas, closes):
    """
    Calculate the lambda normalization factor
    
    Args:
        deltas: List of price deltas
        closes: List of corresponding closing prices
    Returns:
        lambda_factor: Normalization factor
    """
    # Calculate sum of |delta|/close for each pair
    sum_delta_close_ratio = sum(abs(delta)/close 
                               for delta, close in zip(deltas, closes))
    
    # Calculate lambda using the formula: (2/P) * sum(|delta|/close)
    P = len(deltas)
    lambda_factor = (2/P) * sum_delta_close_ratio
    
    return lambda_factor

def prepare_input_data(candles):
    """
    Convert 3 days of candlesticks to 10 input nodes using the delta calculations
    
    candles: List of dictionaries containing OHLC data for 3 days
    returns: Normalized input tensor
    """
    deltas = []
    
    # Calculate all required deltas
    for i in range(len(candles)):
        c = candles[i]
        # Within day deltas
        deltas.append(c['high'] - c['open'])    # Delta 1
        deltas.append(c['high'] - c['low'])     # Delta 2
        deltas.append(c['close'] - c['low'])    # Delta 3
        
        # Add close-to-close delta for days except first
        if i > 0:
            deltas.append(c['close'] - candles[i-1]['close'])
    
    # Calculate lambda using the new function
    closes = [c['close'] for c in candles] * 3  # Repeat closes to match delta count
    closes = closes[:len(deltas)]  # Trim to match number of deltas
    lambda_factor = calculate_lambda(deltas, closes)
    
    # Normalize deltas to [-0.95, 0.95] range
    X = [min(max(delta/(c*lambda_factor), -0.95), 0.95) 
         for delta, c in zip(deltas, closes)]
    
    return torch.tensor(X, dtype=torch.float32)

def convert_outputs(output, current_close, lambda_c, lambda_h, lambda_l):
    """
    Convert network outputs to predicted prices
    """
    # Extract outputs (y1, y2, y3 from the document)
    y1, y2, y3 = output.detach().numpy()
    
    # Calculate predicted values using formulas from the document
    pred_close = 2 * current_close * lambda_c * (y2 - 0.5) + current_close
    pred_high = pred_close + current_close * lambda_h * y1
    pred_low = pred_close + current_close * lambda_l * y3
    
    return pred_close, pred_high, pred_low

def prepare_target_data(next_candle, current_candle, lambda_h, lambda_c, lambda_l):
    """
    Prepare target values for training using the d1,i, d2,i, d3,i equations
    
    Args:
        next_candle: Dictionary with OHLC data for the target day (i+1)
        current_candle: Dictionary with OHLC data for current day (i)
        lambda_h, lambda_c, lambda_l: Lambda factors for high, close, and low
    Returns:
        torch tensor with normalized target values [d1,i, d2,i, d3,i]
    """
    # Calculate d1,i = min(max((h_{i+1} - c_{i+1})/(c_i * lambda_h), 0.05), 0.95)
    d1 = min(max((next_candle['high'] - next_candle['close']) / 
                 (current_candle['close'] * lambda_h), 0.05), 0.95)
    
    # Calculate d2,i = min(max((c_{i+1} - c_i)/(c_i * lambda_c), 0.05), 0.95)
    d2 = min(max((next_candle['close'] - current_candle['close']) / 
                 (current_candle['close'] * lambda_c), 0.05), 0.95)
    
    # Calculate d3,i = min(max((c_{i+1} - l_{i+1})/(c_i * lambda_l), 0.05), 0.95)
    d3 = min(max((next_candle['close'] - next_candle['low']) / 
                 (current_candle['close'] * lambda_l), 0.05), 0.95)
    
    return torch.tensor([d1, d2, d3], dtype=torch.float32)

# Example usage:
def train_step(model, optimizer, inputs, targets):
    optimizer.zero_grad()
    outputs = model(inputs)
    
    # Calculate loss as described in the document
    loss = torch.mean((outputs - targets)**2)
    
    loss.backward()
    optimizer.step()
    return loss.item()