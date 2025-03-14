# Day Trading with a Price Prediction Model

## Overview
This project implements a neural network-based day trading model that predicts stock price movements. The system uses historical candlestick data to forecast the next day's close, high, and low prices, which are then used to execute trading decisions.

## Architecture

### Neural Network Component
- **Input**: 10 delta values derived from 3 daily candlesticks
- **Hidden Layer**: 8 nodes + bias
- **Output**: 3 nodes (predicted close, high, and low prices)

### Price Delta Calculation
The model converts candlestick data into normalized input nodes using:
- λ = (2/P) * Σ(|Δ|/cᵢ)
- Xᵢ = Δ/(cᵢ * λ)

Where:
- Δ represents price changes
- cᵢ is the closing price
- Xᵢ is normalized between -0.95 and 0.95

### Price Prediction Formulas
$c_{i+1} = 2c_i λ_c (y_2 - 0.5) + c_i$ # Predicted close
$h_{i+1} = c_{i+1} + c_i λ_h y_1$ # Predicted high
$l_{i+1} = c_{i+1} + c_i λ_l y_3$ # Predicted low

## Trading Strategy
The basic trading strategy implements:
1. Buy when predicted close > opening price
2. Sell when predicted close < opening price
3. Implements stop-loss protection using:
   - Buy Stop = $o_{i+1} - c_i * StopLoss%$
   - Sell Stop = $o_{i+1} + c_i * StopLoss%$

## Project Structure
AlgyAlgo/
├── idea/ \
│ └── main.tex \
├── images/ \
│ ├── inputDeltaChart.png \
│ ├── networkDiagram.png \
│ └── tradingStrat.png \
└── ref/ \
└── ref.bib \

## Requirements
- LaTeX environment
- Neural network implementation (TBD)
- Financial data source (TBD)

## Future Improvements
- Scale up neural network complexity
- Incorporate multiple stock tickers
- Optimize number of hidden layer nodes
- Implement advanced trading strategies

## Authors
- Clark - Neural Network Implementation
- Parth - Trading Strategy Development

## License
[TBD]

## References
See ref.bib for complete bibliography