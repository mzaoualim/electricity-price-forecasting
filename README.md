# Electricity Price Forecasting

Arbitrage in trading is such a powerful strategy to gain profits by spotting market imbalances.

In electricity market, it is realized by [...take positions in the day-ahead market and then offload them in the balancing market (also known as intraday market)...](https://arxiv.org/pdf/2301.08360v3.pdf).

So, How profitable this strategy can reach, using machine learning algorithms to spot imbalances and turn it into profit&loss signals?

This is a full notebook, with step by step implementation of this strategy to Spanish Electricity market covering:
## Getting and preprocessing historical data

 * ### Loading more than 6 years of hourly prices (day-ahead and intraday).

 * ### Cleaning and formatting data to suite model requirements.

## Optimizing and Modeling

 * ### Using Optimized (Hyper-parameter tuned) Facebook Prophet Model.

## Evaluation

 * ### Evaluate predictions using Cross-validation Technique and sMAPE metric.

## Testing Against live data

* ### Perform Prediction of last 24h of available spreads and compare it with the realized ones.

# [Have Fun!](https://github.com/mzaoualim/electricity-price-forecasting/blob/039904123c2925a0499fdeb4e8cd28acddfa1cfc/electricity_price_forcasting.ipynb)




