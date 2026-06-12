# Anomaly Attribution Engine for Indian Agricultural Yield Data

An end-to-end data science pipeline that automatically detects major structural breaks (change points) in historical Indian crop yield time series and explains the underlying drivers of these shifts using machine learning and SHAP (Shapley Additive exPlanations) attribution.

---

## 🌟 Executive Summary

In agriculture, sudden shifts in crop yields can be driven by climate changes, policy shifts, or changes in farming inputs (fertilizer, pesticide, cultivation area). Identifying *when* these shifts happen and *why* they happen is critical for policy-making and food security. 

This engine solves this by:
1. **Detecting Sudden Shifts**: Scanning historical yield data to find statistically significant structural breaks (change points) in yield trends across various states and crops.
2. **Explaining the Causes**: Training a localized machine learning model around each detected break and running SHAP feature attribution to explain *which* specific inputs (such as fertilizer, pesticides, or rainfall) drove the transition.
3. **Adding Context**: Mapping descriptive context from a massive secondary dataset of NDVI (satellite-derived vegetation index) and temperature to check environmental conditions.
4. **Providing an Interactive Dashboard**: Enabling users to select any state and crop to inspect yield charts, break points, and visual AI explanations instantly.

---

## 🔍 Methodology (Simple English)

### Stage 1: Finding the Break Points
For every crop and state combination that has at least 8 years of historical records, we analyze the yield time series. Using a mathematical algorithm called **PELT (Pruned Exact Linear Time)** from the `ruptures` library, we find years where the yield trend suddenly and permanently shifts (up or down).

### Stage 2: Filtering Out Noise
Some shifts are just small, normal year-to-year fluctuations. To ignore this noise, we filter our detected shifts. A break is only kept if it is highly dramatic—either the difference in average yield before vs. after exceeds **1.0 standard deviation** of that crop's historic variance, and/or a two-sample statistical **t-test** confirms the shift is highly unlikely to be random chance ($p < 0.05$). Out of 714 raw breaks, **602** were confirmed as statistically significant.

### Stage 3: Training Local AI Models to Explain the Break
For each significant break, we zoom in on a 6-year window (3 years before the break + 3 years after). We train a small, localized Random Forest machine learning model specifically on this window. The model learns how crop inputs (Area, Rainfall, Fertilizer, Pesticide, Season, and Year) relate to Yield.
We then calculate **SHAP values**—a game-theory method that shows exactly how much credit each input feature deserves for a given yield prediction. By comparing SHAP values before vs. after the break, we identify the feature with the largest shift in contribution and programmatically generate a plain-English explanation.

### Stage 4: Integrating Satellite NDVI Context
To add environmental context, we utilize a secondary dataset containing 1 million records of satellite-derived vegetation indexes (NDVI), Land Surface Temperatures (LST), and reference evapotranspiration (ETo). Since this dataset lacks common keys like "State" or "Crop", we map it using **precipitation overlap**: we take the average rainfall of the crop around the break and query the NDVI dataset for regions with similar rainfall ($\pm 5\%$). If the rainfall falls outside the NDVI dataset's bounds, we explicitly report that no direct overlap is available to prevent fabricating false links.

---

## 🏆 Top 5 National Findings

Across all 602 detected anomalies, these are the 5 most dramatic and statistically significant crop yield breaks detected in India, ranked by shift magnitude ($\sigma$):

1. **Uttarakhand — Potato (2002)**
   - **Yield Shift**: Fell from **20.67 t/ha** to **10.98 t/ha** (a **46.9% decrease**; magnitude: **2.50 σ**)
   - **AI Explanation**: Shifts in seasonal cropping patterns (a transition in seasonal reporting/outputs) were the primary drivers of the yield decline.
   - **Environmental Context**: N/A (Rainfall of 1,513 mm was outside the NDVI dataset's bounds).

2. **Andhra Pradesh — Banana (2017)**
   - **Yield Shift**: Rose from **27.78 t/ha** to **100.87 t/ha** (a **263.1% increase**; magnitude: **2.39 σ**)
   - **AI Explanation**: An increase in pesticide application post-break supported a strong yield recovery/increase.
   - **Environmental Context**: N/A (Rainfall of 918 mm was outside the NDVI dataset's bounds).

3. **Assam — Turmeric (2017)**
   - **Yield Shift**: Rose from **0.78 t/ha** to **1.23 t/ha** (a **57.8% increase**; magnitude: **2.39 σ**)
   - **AI Explanation**: An increase in fertilizer usage post-break supported a strong yield recovery.
   - **Environmental Context**: Matched precipitation regions (2,151 mm) show that **33.9%** of observations fall into Good/Excellent health categories, with an average Land Surface Temperature of **26.1°C** and reference evapotranspiration (ETo) of **412.6**.

4. **Delhi — Barley (2003)**
   - **Yield Shift**: Rose from **0.39 t/ha** to **2.80 t/ha** (a **621.2% increase**; magnitude: **2.37 σ**)
   - **AI Explanation**: Changes in pesticide application post-break marked a significant shift in crop input efficiency and yield contribution.
   - **Environmental Context**: N/A (Rainfall of 670 mm was outside the NDVI dataset's bounds).

5. **Gujarat — Moong / Green Gram (2018)**
   - **Yield Shift**: Rose from **0.51 t/ha** to **0.95 t/ha** (an **85.6% increase**; magnitude: **2.37 σ**)
   - **AI Explanation**: Long-term technology improvements and temporal trends drove the yield recovery post-break.
   - **Environmental Context**: N/A (Rainfall of 925 mm was outside the NDVI dataset's bounds).

---

## 💾 Project Structure

- `change_point_detection.py` — Runs the PELT change point algorithm and filters for statistical significance (Stage 1-2).
- `shap_attribution.py` — Extracts local windows, trains local Random Forest models, and calculates SHAP attributions (Stage 3).
- `ndvi_context.py` — Integrates and queries the secondary satellite environmental context dataset (Stage 4).
- `report_generator.py` — Aggregates all steps into a single report containing all anomaly cards (Stage 5, outputs `anomaly_report.json`).
- `app.py` — Streamlit interactive web dashboard showing yield time series and AI explanations (Stage 6).
- `README.md` — This project summary and portfolio guide.

---

## 🧪 Limitations

- **NDVI Dataset Mapping**: The NDVI dataset has no direct spatial (state) or temporal (year) keys matching the yield dataset. To avoid fabricating data relationships, **we do not row-merge them**. The NDVI data is used solely as a descriptive environmental lookup based on matching rainfall ranges. If a crop's average rainfall falls outside the NDVI precipitation range (1666 mm to 4734 mm), the engine reports this limitation explicitly.

---

## 🚀 How to Run Locally

### 1. Prerequisites
Ensure you have Python installed. You can install all dependencies via pip:
```bash
pip install pandas numpy ruptures scikit-learn shap streamlit plotly scipy
```

### 2. Run the Dashboard
To start the interactive dashboard, run:
```bash
streamlit run app.py
```
This will start a local web server and open the dashboard in your default browser. You can select any State and Crop from the sidebar, view the historical yield line, click on marked break points, and see full statistical and SHAP-based attributions instantly.
