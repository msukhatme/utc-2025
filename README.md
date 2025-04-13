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
6. [Setup and Installation](#setup-and-installation)
7. [Usage](#usage)
8. [Development and Contributions](#development-and-contributions)
9. [License](#license)

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
