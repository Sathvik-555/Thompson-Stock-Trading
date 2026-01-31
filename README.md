# 📈 Thompson Stock Trading

A stock trading application using Thompson Sampling algorithm to optimize stock selection from Indian market portfolios. The system uses Bayesian inference to learn and adapt stock selection strategies based on historical performance.

## 🚀 Features

- **Thompson Sampling Algorithm**: Uses Bayesian bandit approach for intelligent stock selection
- **Real-time Data**: Fetches live stock data from Yahoo Finance
- **Interactive Dashboard**: Streamlit-based web interface for visualization
- **Multiple Simulations**: Run and compare multiple trading strategies
- **Portfolio Analysis**: Compare different stock portfolios (Large-cap vs Top Performers)
- **Performance Metrics**: Track returns, Sharpe ratios, and portfolio values

## 📊 Portfolios

### Portfolio 1: Large-cap Stocks
Includes major Indian companies like Reliance, TCS, HDFC Bank, Infosys, ICICI Bank, and more.

### Portfolio 2: Top Performers
Includes high-growth stocks like Cochin Shipyard, IRFC, Suzlon, Adani Green, Trent, and more.

## 🛠️ Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/Thompson-Stock-Trading.git
cd Thompson-Stock-Trading
```

2. Install required dependencies:
```bash
pip install -r requirements.txt
```

## 🎯 Usage

### Run the Streamlit Dashboard

```bash
streamlit run src/app.py
```

This will launch an interactive web dashboard where you can:
- Adjust simulation parameters
- Select date ranges
- Compare portfolio performances
- Visualize stock selection patterns
- Analyze returns and Sharpe ratios

### Run Command-line Trading Bot

```bash
python src/trading_bot.py
```

### Run Tests

```bash
python src/test.py
```

## 📁 Project Structure

```
Thompson-Stock-Trading/
├── src/
│   ├── app.py                 # Streamlit dashboard application
│   ├── thompson_trader.py     # Thompson Sampling trader implementation
│   ├── strategy_engine.py     # Trading strategy engine
│   ├── trading_bot.py         # Command-line trading bot
│   ├── utils.py               # Utility functions
│   └── test.py                # Test scripts
├── data/
│   └── thompson_sample.csv    # Sample trading data
├── requirements.txt           # Python dependencies
└── README.md                  # Project documentation
```

## 🔬 How It Works

### Thompson Sampling Algorithm

Thompson Sampling is a probabilistic algorithm that balances exploration and exploitation:

1. **Initialize**: Start with prior distributions for each stock's expected return
2. **Sample**: Draw a random sample from each stock's posterior distribution
3. **Select**: Choose the stock with the highest sampled value
4. **Update**: Update the posterior distribution based on observed returns
5. **Repeat**: Continue the process for each trading day

### Bayesian Updates

The algorithm uses Bayesian inference to update beliefs about stock performance:
- **Prior**: Initial estimates based on historical statistics
- **Likelihood**: Observed daily returns
- **Posterior**: Updated beliefs combining prior and observed data

## 📈 Key Metrics

- **Portfolio Value**: Total value of investment over time
- **Daily Returns**: Percentage change in portfolio value
- **Sharpe Ratio**: Risk-adjusted return metric
- **Stock Selection Frequency**: How often each stock is selected

## 🔧 Configuration

Adjust simulation parameters in the Streamlit dashboard:
- Number of simulations (10-500)
- Random seed for reproducibility
- Start and end dates for historical data

## 📦 Dependencies

- numpy
- pandas
- yfinance
- streamlit
- altair

## 🤝 Contributing

Contributions are welcome! Feel free to submit issues or pull requests.

## 📝 License

This project is open source and available under the MIT License.

## ⚠️ Disclaimer

This application is for educational purposes only. Do not use it for actual trading without proper research and risk management. Past performance does not guarantee future results.
