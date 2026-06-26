# FUTURE_ML_01
# Retail Sales Demand Forecasting Pipeline & Interactive Dashboard

An end-to-end Machine Learning and Business Intelligence solution that transforms raw historical transaction logs into structured, actionable demand forecasts. 

This project builds a robust predictive model using **XGBoost** to smooth out volatile retail purchase spikes, capturing the steady, operational baseline demand of the business. The final predictive insights are exported automatically into an interactive **Power BI Dashboard** for executive strategic planning.

---

##  Business Problem & Solution Design

In retail supply chains, sudden bulk corporate orders create drastic, unpredictable revenue spikes. Forcing an inventory system to chase these random spikes leads to panic-buying, overstocking, and expensive warehouse overhead.

### The Solution:
***The AI Weather Forecast:** Instead of trying to guess exact daily order anomalies, our pipeline processes historical data into clean weekly regional buckets and models the **underlying operational demand trend**. 
***The Outcome:** Management can look at the smooth forecast trajectory to confidently scale supply chains up or down, minimizing unsold inventory and avoiding empty shelves.



##  Project Structure & Architecture

```text
FUTURE_ML_01/
│
├── data/
│   ├── superstore.csv.csv            # Raw historical transaction source data
│   └── dashboard_feed.csv            # Final ML model predictions exported for Power BI
│
├── notebooks/
│   └── EDA_and_Forecasting.ipynb     # Main workspace (EDA, Log Transformation, Training, GridSearch)
│
├── Sales_Demand_Forecasting_Dashboard.pbix  # Final interactive Power BI visualization file
│
├── requirements.txt                  # Local execution environment dependencies
└── README.md                         # Project documentation and presentation guide