# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a quantitative trading research project built on the RQAlpha framework. The repository contains:

- **Trading strategies** (strategy_1.py, strategy_2.py) that implement algorithmic trading logic
- **RQAlpha framework source code** (src/rqalpha/) - a complete Python backtesting platform
- **Baseline testing** (baseline/test.py) for validation
- **Documentation and research notes** (docs/)

The project focuses on developing and testing quantitative trading strategies for Chinese stock markets using technical indicators, volatility controls, and risk management techniques.

## Key Commands

### Environment Setup
```bash
# Install RQAlpha platform
pip install -i https://pypi.douban.com/simple rqalpha

# Verify installation
rqalpha version

# Download stock data bundle
rqalpha download-bundle
```

### Running Strategies
```bash
# Run baseline test (verify setup)
rqalpha run -f /path/to/baseline/test.py -s 2016-06-01 -e 2016-12-01 --account stock 100000 --benchmark 000300.XSHG --plot

# Run custom strategy
rqalpha run -f /path/to/strategy_1.py -s 2018-01-01 -e 2025-07-01 --account stock 1000000 --benchmark 000300.XSHG --plot

# Run with specific parameters
rqalpha run -f strategy_file.py -s start_date -e end_date --account stock initial_capital --benchmark benchmark_index --plot
```

### Testing and Validation
```bash
# Test baseline functionality
cd baseline && python test.py

# Run RQAlpha's built-in tests
cd src/rqalpha && python -m pytest tests/
```

## Architecture Overview

### Core Components

1. **Strategy Files** (`strategy_1.py`, `strategy_2.py`):
   - Implement trading logic using RQAlpha API
   - Define `init(context)` for initialization and `handle_bar(context, bar_dict)` for bar-by-bar execution
   - Include risk management, position sizing, and entry/exit logic

2. **RQAlpha Framework** (`src/rqalpha/`):
   - Complete backtesting engine with modular architecture
   - Core modules: `core/`, `data/`, `mod/`, `portfolio/`
   - API layer: `apis/` for strategy interaction
   - Built-in modules for accounts, analysis, simulation, risk management

3. **Strategy Components**:
   - **Market filtering**: Uses MA200/MA50 for bull/bear market detection
   - **Volatility targeting**: Dynamic position sizing based on target volatility
   - **Risk controls**: ATR-based stops, drawdown protection, time-based exits
   - **Stock selection**: Quality filters, liquidity requirements, fundamental screening

### Key Strategy Patterns

**Strategy 1** (Conservative Quality Rebound):
- Focuses on stocks recovering from drawdowns with low volatility
- Implements ATR-based stop losses and trailing stops
- Uses portfolio-level drawdown protection
- Market regime filtering with CSI 300 MA200

**Strategy 2** (Trend Following with Risk Controls):
- Trend-following approach with candlestick risk assessment
- Breakout and pullback entry patterns
- State synchronization for handling suspended trading
- K-line pattern risk detection

### Configuration
- Strategies use `__config__` dictionary for backtesting parameters
- RQAlpha configuration in `src/rqalpha/rqalpha/config.yml`
- Modular configuration in `mod_config.yml`

## Important Notes

- **Data Requirements**: Strategies require Chinese stock market data bundle
- **File Paths**: Use absolute paths when running strategies via command line
- **Dependencies**: Project uses pandas, numpy, talib for technical analysis
- **Benchmark**: Default benchmark is CSI 300 (000300.XSHG)
- **Risk Management**: Both strategies implement multiple layers of risk controls
- **State Management**: Strategy 2 includes position state synchronization for handling market suspensions