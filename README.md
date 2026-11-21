# EV Charging Station Demand & Throughput Forecasting (End-to-End DS & ML)

**Module 06: Final Data Science Project** | Codomax Digital Solutions Data Science Internship

## Problem Statement
Planning electric vehicle (EV) charging expansion requires estimating daily energy demand (`kWh`) across different charger tiers, site types, and traffic conditions so capital is allocated where utilization is highest.

## Workflow & Methodology
1. **Data Cleaning & Feature Engineering:** Imputed missing operational metrics (`daily_traffic_k`, `uptime_pct`) and engineered `kwh_per_port` and `estimated_daily_revenue_inr`.
2. **Exploratory Data Analysis:** Compared energy throughput across charger tiers and evaluated feature correlations.
3. **Machine Learning Pipeline:** Built a reproducible `ColumnTransformer` + `Pipeline` workflow benchmarking **Ridge Regression**, **Random Forest Regressor**, and **Gradient Boosting Regressor** using MAE, RMSE, and $R^2$.

## Key Takeaways
- Charging throughput scales strongly with fast/ultra-fast charger deployment, local EV density, corridor traffic, and station uptime.
- Tree/boosting models capture non-linear interactions between site attributes and daily energy demand more accurately than linear baselines.

## Tech Stack
- **Python**, **Pandas**, **NumPy**
- **Matplotlib**, **Seaborn**
- **Scikit-learn** (`Pipeline`, `ColumnTransformer`, `Ridge`, `RandomForestRegressor`, `GradientBoostingRegressor`)
