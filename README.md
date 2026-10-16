# Retail Demand Forecasting — React + FastAPI (2024–2026 dataset)

This project uses a React/Vite frontend, a FastAPI backend, and an Extra Trees regression model. The updated dataset has 76,000 rows and dates from 2024-01-01 through 2026-01-30.

> **Data note:** The dataset dates were shifted forward by two years. It is not newly collected real-world 2026 sales data; demand values remain from the original dataset.

## Project structure

```text
Retail_Demand_Forecasting_React_FastAPI_2026/
├── backend/
│   ├── main.py
│   ├── train_model.py
│   ├── requirements.txt
│   └── demand_forecasting.csv
└── frontend/
    ├── index.html
    ├── package.json
    ├── vite.config.js
    └── src/
        ├── main.jsx
        ├── App.jsx
        └── styles.css
```

## 1. Train the ML model

Open a terminal in the `backend` folder:

```bash
python -m venv .venv
```

Activate the environment:

- Windows PowerShell: `.venv\\Scripts\\Activate.ps1`
- macOS/Linux: `source .venv/bin/activate`

Install and train:

```bash
pip install -r requirements.txt
python train_model.py
```

This creates `demand_forecasting_model.pkl`, `model_metrics.json`, and `test_predictions.csv` in `backend/`.

## 2. Start the API

Keep the terminal in `backend/` and run:

```bash
uvicorn main:app --reload --port 8000
```

API docs: http://127.0.0.1:8000/docs

## 3. Start React

Open a **second terminal** in the `frontend` folder and run:

```bash
npm install
npm run dev
```

Open the local URL printed by Vite, usually http://localhost:5173.

## Dashboard features

- Overview: record, store, product and demand KPIs; historical demand charts.
- Forecast: separate Store, Category, Month and Year dropdowns; daily category forecast and CSV download.
- Model evaluation: MAE, RMSE, R², WAPE and actual-versus-predicted test chart.

## Forecast assumptions and limitations

- Forecasts aggregate predictions for products in the selected store/category.
- Forecasting a month uses only records dated before that month's first day.
- For future days, the model recursively uses earlier predictions as lag features.
- Future inventory, price, discount, promotion, competitor pricing and epidemic values are held at the latest available values for each product.
- R² is not the same as prediction accuracy. Forecasts are estimates, not guarantees.
- The API's default CORS policy permits the local Vite frontend. Update `allow_origins` in `backend/main.py` when deploying the frontend to a different domain.
