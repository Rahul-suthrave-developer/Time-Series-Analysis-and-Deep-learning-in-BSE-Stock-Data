# Time-Series-Analysis-and-Deep-learning-in-BSE-Stock-Data
Time Series Analysis of BSE Stock Index — cleaned and analyzed historical BSE index data, built and compared statistical (ARIMA/SARIMA) and machine-learning (LSTM) forecasting models, N-BEATS Algorithm and produced an interactive notebook with model evaluation and actionable visualizations using Pandas, NumPy, Statsmodels, Scikit-learn and Keras.
This project investigates historical price dynamics of the Bombay Stock Exchange (BSE) index and builds forecasting models to help anticipate near-term index movements. The work is structured as an executable Jupyter notebook that walks through:

Data ingestion & cleaning: loading historical BSE index data, handling missing values, resampling and frequency alignment, feature engineering (log returns, rolling statistics, lag features).

Exploratory data analysis (EDA): visualization of trends, seasonality and volatility; correlation checks and identification of structural breaks.

Stationarity & decomposition: Augmented Dickey-Fuller (ADF) test, KPSS (optional), time series decomposition into trend/seasonal/residual components to guide modeling decisions.

Modeling & forecasting: comparison of classical statistical models (ARIMA / SARIMA) and machine learning / deep learning approaches (e.g., LSTM). Hyperparameter tuning, cross-validation on time splits, and model selection based on performance metrics.

Evaluation & backtesting: model evaluation using RMSE, MAE and MAPE on hold-out test sets and visual backtests that overlay actual vs forecasted values.

Visualization & interpretation: publication-quality plots of forecasts, residual diagnostics, and feature importance / sensitivity analysis where applicable.

Deliverables: reproducible notebook, model artifacts (pickles/h5), and suggestions for deployment (rolling forecast pipelines, API endpoints, or integration with dashboards).

Key technical stack

Python: Pandas, NumPy for data wrangling

Visualization: Matplotlib, Seaborn, Plotly (optional interactive plots)

Statistical modeling: Statsmodels (ARIMA/SARIMA), pmdarima (auto_arima)

Machine learning / Deep learning: Scikit-learn (scaling, metrics), TensorFlow / Keras or PyTorch for LSTM

Utilities: scikit-optimize or GridSearchCV for tuning, joblib/h5py for saving models

Environment: Jupyter Notebook, Git for version control
