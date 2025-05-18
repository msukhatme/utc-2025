# UTC 2025
Welcome to the UTC 2025 repository, which contains my team's code and supporting documents for the 2025 UChicago Trading Competition. In this competition, teams build trading algorithms to compete in two separate cases: a Live Trading Bot and Portfolio Optimization. This repo includes all necessary code, configuration, and documentation to run our strategies on the provided exchange platform.

---

## Table of Contents
1. [Overview](#overview)
2. [Repository Structure](#repository-structure)
3. [Case 1: Live Trading Bot](#case-1:-live-trading-bot)
    1. [Competition Details](#competition-details)
    2. [Our Strategy](#our-strategy)
4. [Case 2: Portfolio Optimization and Asset Return Prediction](#case-2:-portfolio-optimization-and-asset-return-prediction)
    1. [Competition Details](#competition-details)
    2. [Our Strategy](#our-strategy)
5. [utcxchangelib Library](#utcxchangelib-library)
6. [Development and Contributions](#development-and-contributions)

---

## Overview
UTC 2025 challenges participants to design high-performance trading algorithms operating on two distinct cases:
1. ### Case 1: Live Trading Bot
Compete against other competitors and bots that exist on the exchange to generate as high of a PnL as possible by trading three stocks and two ETFs.
2. ### Case 2: Portfolio Optimization and Asset Return Prediction
Build a portfolio allocation algorithm that invests over a long time horizon using intraday daya. The challenge is to maximize risk-adjusted returns, primarily measured by the annual Sharpe ratio.

---

## Repository Structure
```bash
utc-2025/
├── case1/
│   ├── case1_algo.py         # Algorithmic trading bot for Case 1
│   ├── config.json           # Configuration file for parameters and risk limits
│   └── case1_doc.pdf         # Detailed documentation for Case 1 rules and guidelines
├── case2/
│   ├── case2_algo.ipynb      # Jupyter notebook with our portfolio allocation strategy for Case 2
│   ├── case2.csv             # Example historical data or output used in analysis
│   └── case2_doc.pdf         # Documentation of Case 2 specifications and analysis guidelines
└── utcxchangelib/
    └── xchange_client.py     # Library providing connectivity to the UTC exchange platform
```
- `case1/`: Contains our live trading bot, a JSON config file to adjust parameters at run-time, and a PDF with official case documentation.
- `case2/`: Contains our portfolio allocation algorithm implemented as a Jupyter Notebook (for interactive exploration), the training output data as a CSV file, and a PDF with official case documentation.
- `utcxchangelib/`: Provides the core exchange client implementation needed to interface with the UTC exchange. This module handles authentication, order placement, order cancellation, receiving market data, and processing news events.

---

## Case 1: Live Trading Bot

### Competition Details
The live trading bot competition involves trading three stocks (APT, DLR, MKJ) and two ETFs (AKAV, AKIM) on a simulated exchange. The goal is to maximize PnL through:
- Market making and liquidity provision
- Statistical arbitrage between ETFs and their underlying components
- News-based trading strategies
- Risk management and position sizing

### Our Strategy
Our trading bot implements several sophisticated strategies:
1. **Market Making**: Dynamic quote pricing based on volatility and order book imbalance
2. **Statistical Arbitrage**: Exploiting price discrepancies between ETFs and their components
3. **News Analysis**: Real-time sentiment analysis of news events affecting MKJ
4. **Risk Management**: Position limits, profit taking, and dynamic hedging
5. **Adaptive Parameters**: Runtime configuration through `config.json`

## Case 2: Portfolio Optimization and Asset Return Prediction

### Competition Details
The portfolio optimization challenge requires:
- Building a portfolio allocation algorithm for long-term investment
- Using intraday data to maximize risk-adjusted returns
- Optimizing for Sharpe ratio as the primary performance metric
- Managing portfolio weights across multiple assets

### Our Strategy
Our portfolio optimization approach combines:
1. **ARIMA Forecasting**: Time series modeling for return prediction
2. **Dynamic Portfolio Allocation**: Mean-variance optimization with constraints
3. **Risk Management**: Rolling window covariance estimation
4. **Adaptive Rebalancing**: Continuous portfolio weight updates based on new data

## utcxchangelib Library
The `utcxchangelib` provides essential functionality for interacting with the UTC exchange:
- WebSocket-based market data streaming
- Order placement and management
- News event processing
- Authentication and session management
- Error handling and reconnection logic

## Development and Contributions

### Code Structure
- `case1/`: Live trading implementation
  - `case1_algo.py`: Main trading bot
  - `config.json`: Runtime parameters
  - `case1_doc.pdf`: Competition rules
- `case2/`: Portfolio optimization
  - `case2_algo.ipynb`: Interactive analysis
  - `case2.csv`: Historical data
  - `case2_doc.pdf`: Competition specifications
- `utcxchangelib/`: Exchange interface
  - `xchange_client.py`: Core exchange functionality

### Logging
The trading bot generates several log files:
- `market_values.log`: Asset price tracking
- `theo_values.log`: Theoretical value calculations
- `arb.log`: Arbitrage opportunities
- `news.log`: News event processing
- `orders.log`: Order execution details
- `risk.log`: Risk management metrics
