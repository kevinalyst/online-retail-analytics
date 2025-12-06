# 🚀 Customer Value & Retention Strategy Dashboard

![Power BI](https://img.shields.io/badge/power_bi-F2C811?style=for-the-badge&logo=powerbi&logoColor=black)
![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54)
![Databricks](https://img.shields.io/badge/Databricks-FF3621?style=for-the-badge&logo=databricks&logoColor=white)
![Scikit-Learn](https://img.shields.io/badge/scikit_learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)

An end-to-end advanced analytics solution designed to transform raw retail transaction data into actionable business strategy. This project moves beyond standard reporting to predict Customer Lifetime Value (CLV) and identify high-probability cross-sell opportunities.

---

## 📺 Project Walkthrough

**Watch the Executive Presentation Simulation:**
I skip the coding tutorial and simulate a real-world C-Suite presentation, pitching the strategic value of this dashboard to leadership.

[![Watch the Video](https://img.youtube.com/vi/qwvafbBRzrM/maxresdefault.jpg)](https://www.youtube.com/watch?v=qwvafbBRzrM)

*(Click the image above to watch the video)*

---

## 📊 Dashboard Preview

![Dashboard Screenshot](Dashboard%20Screenshot.png)
*A view of the Strategic Overview page featuring RFM Segmentation, CLV Prediction Funnels, and Cohort Retention Analysis.*

---

## 🛠️ Technical Architecture & Workflow

This project simulates a modern data stack, moving from raw data ingestion to machine learning and final BI visualization.

### 1. Data Ingestion & Engineering
* **Source:** [UCI Online Retail Dataset](https://archive.ics.uci.edu/dataset/352/online+retail)
* **Platform:** Databricks
* **Process:** Ingested raw `.csv` data into Databricks for cleaning, outlier removal, and exploratory data analysis (EDA).
* **Feature Engineering:** Calculated R, F, and M (Recency, Frequency, Monetary) scores to segment customers.

### 2. Advanced Analytics (Python)
* **CLV Forecasting:** Trained a **Gradient Boosting Regressor** model (using `scikit-learn`) on local IDE to forecast future Customer Lifetime Value based on historical purchasing behavior.
* **Market Basket Analysis:** Implemented the **Apriori Algorithm** to identify product associations (bundles), calculating Confidence and Lift metrics to find high-value cross-sell opportunities.

### 3. Business Intelligence (Power BI)
* **Transformation:** Imported processed data and model outputs; used **Power Query** for final transformation and deduplication.
* **Modeling:** Built a Star Schema data model to connect Transactional Data, Customer Dimensions, and ML Outputs.
* **DAX Measures:** Created dynamic measures for "Revenue at Risk," "Opportunity Gaps," and complex Time Intelligence calculations.
* **UI/UX:** Designed a stakeholder-focused interface using custom tooltips, drill-throughs, and bookmark navigators.

---

## 💡 Key Business Insights & Takeaways

The goal of this project was to turn data into **actionable solutions**. Here is the strategic value delivered:

### 📉 From "Reporting" to "Prescribing"
Instead of just showing historical sales, the dashboard prescribes actions:
* **Opportunity Gap:** Quantified a **$1.48M** revenue opportunity by targeting "Growth Potential" customers who are under-monetized compared to "High Value" peers.
* **Churn Risk:** Identified **$329k** in customer equity at risk within the "Needs Attention" segment.

### 🤖 Operationalizing Machine Learning
* **Predictive CLV:** Allows marketing to allocate budget based on *future* value rather than past spend.
* **Market Basket Bundles:** Provides specific "Next Best Offer" recommendations (e.g., *Green Teacup + Pink Teacup*) with 83% statistical confidence, ready for implementation in checkout recommendation engines.

### 🔄 Solving the "Month 2 Cliff"
* **Cohort Analysis:** Revealed a critical retention drop-off in Month 2 for recent cohorts.
* **Strategic Fix:** Proposed an automated "Day 30" re-engagement campaign using the Market Basket recommendations to bridge the gap between the first and second purchase.

---

## 📂 Repository Structure

* `train_clv_model.py`: Python script for training the Gradient Boosting Regressor.
* `market_basket_analysis.py`: Script for generating association rules (Bundles).
* `Online_Retail_RFM_Cohort.pbix`: The final Power BI file.
* `Explore workspace...`: Databricks notebook exports.
