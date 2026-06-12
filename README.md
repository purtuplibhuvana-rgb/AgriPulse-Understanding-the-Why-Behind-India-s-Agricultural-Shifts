# AgriPulse-Understanding-the-Why-Behind-India-s-Agricultural-Shifts
An explainable AI system that automatically detects structural breaks in Indian crop yield trends and uses SHAP-based feature attribution to explain why they happened — backed by statistical significance testing and environmental context from satellite-derived NDVI data.
Overview

Most agricultural analytics projects stop at prediction — "here's what the yield will be." This project asks a different question:


When did yield patterns fundamentally change, and what drove that change?



For every state–crop combination in India's agricultural yield records, the engine automatically scans the historical time series for statistically significant structural breaks — points where the underlying yield trend shifted meaningfully rather than just fluctuating year to year. For each break detected, it trains a localized model on the surrounding years and uses SHAP (SHapley Additive exPlanations) to identify which factor — cultivation area, fertilizer usage, pesticide application, rainfall, or seasonal patterns — shifted most in importance across that break.

Where possible, the system cross-references satellite-derived NDVI (Normalized Difference Vegetation Index) data to add environmental context — and is explicit when no such context can be reliably mapped, rather than fabricating a connection.

The result: an interactive dashboard where every detected anomaly comes with a yield trajectory, statistical confidence score, SHAP-driven explanation, and supporting evidence.


Why this approach

Two independent agricultural datasets — a multi-decade crop yield record and a satellite-based NDVI crop health dataset — share no common keys (no state, year, or crop identifiers in common). Rather than forcing an invalid row-level merge, this project uses each dataset for what it is genuinely suited for:


Yield dataset → the primary signal, used for change-point detection and feature attribution
NDVI dataset → a secondary environmental reference, used only when its precipitation range genuinely overlaps with the conditions of a detected break


This methodological honesty — clearly stating when environmental context is and isn't available — is a core design principle of the system.


How it works

1. Change point detection

Each state–crop yield series (sorted by year and season) is standardized and scanned using the PELT algorithm (ruptures library) to detect structural breaks — points where the mean yield level shifts.

2. Statistical significance filtering

Detected breaks are filtered to retain only those where:


the shift exceeds 1 standard deviation of the series, and/or
a Welch's t-test between the before/after segments returns p < 0.05


This separates genuine structural shifts from year-to-year noise.

3. Local SHAP attribution

For each significant break, a localized model is trained on a window of years surrounding the break, using cultivation area, rainfall, fertilizer usage, pesticide application, and seasonal patterns as features. SHAP values are computed to identify which feature's contribution to yield changed most before vs. after the break — and a natural-language explanation is generated automatically.

4. NDVI environmental context

The system checks whether the average rainfall around a given break falls within the range covered by the NDVI dataset. If it does, NDVI-derived crop health indicators are reported as supporting context. If not, the system explicitly states that no environmental context is available — rather than guessing.

5. Interactive dashboard

A Streamlit application lets users explore yield trends by state and crop, view detected breaks on an interactive time series, and drill into the full attribution card — including the SHAP feature-shift chart, statistical metrics, and NDVI context — for each anomaly.


Key findings

The engine processed yield data across India and surfaced its most statistically significant findings, ranked by shift magnitude:

RankState — CropBreak YearYield ShiftDriving Factor1Uttarakhand — Potato2002 (Rabi)−9.69 t/ha (−46.9%)Shift in seasonal cropping patterns2Andhra Pradesh — Banana2017 (Whole Year)+73.09 t/ha (+263.1%)Increased pesticide application3Assam — Turmeric2017 (Whole Year)+0.45 t/ha (+57.8%)Increased fertilizer usage4Delhi — Barley2003 (Kharif)+2.41 t/ha (+621.2%)Shift in crop input efficiency (pesticide)5Gujarat — Moong (Green Gram)2018 (Summer)+0.44 t/ha (+85.6%)Long-term technology and temporal trends

Each of these findings is statistically validated (Welch's t-test, shift magnitude in standard deviations) and accompanied by a SHAP-driven explanation of the underlying driver — not just a description of what changed, but why.


Example: Karnataka — Coconut (2008)

A representative anomaly card from the dashboard:


Break Year: 2008 (Whole Year)
Yield Before: 4,411.7 t/ha → Yield After: 8,474.6 t/ha (+92.1%)
Statistical confidence: Welch's t-test p = 1.11e-08, shift magnitude = 1.87σ
SHAP attribution: Cultivation area showed the largest post-break shift in feature importance, identified as the primary driver of the yield transition
NDVI context: Average rainfall for this period (1,264.8 mm) fell outside the NDVI dataset's coverage range — the system correctly reports that no direct environmental context is available, rather than forcing an unsupported claim



Tech stack


Python — core pipeline
Pandas — data processing
ruptures — change point detection (PELT algorithm)
scikit-learn — local regression models
SHAP — feature attribution and explainability
Streamlit — interactive dashboard
Plotly / Matplotlib — visualizations



Project structure

├── change_point_detection.py   # Stage 1-2: PELT detection + significance filtering
├── shap_attribution.py         # Stage 3: Local model training + SHAP explanations
├── ndvi_context.py             # Stage 4: NDVI precipitation-based context mapping
├── report_generator.py         # Stage 5: Compiles anomaly cards into JSON
├── app.py                       # Stage 6: Streamlit dashboard
├── anomaly_report.json          # Generated anomaly cards
└── README.md


Running locally

bashpip install -r requirements.txt
streamlit run app.py

Use the sidebar to select a state and crop. If a structural break is detected, select the change point year to view the full attribution card — including the SHAP feature-shift chart, statistical metrics, and NDVI context.


Limitations and design choices


The NDVI dataset and the crop yield dataset share no common keys (state, year, or crop identifiers). Rather than performing an invalid merge, NDVI context is mapped only via precipitation-range overlap, and is explicitly omitted when no overlap exists.
Local SHAP models are trained on small windows of data around each break; where data is sparse, the window is expanded or supplemented with comparable records, with this clearly documented in the methodology.
This project is exploratory and diagnostic in nature — it surfaces where and what changed, with statistically grounded explanations, rather than making causal claims about policy interventions.
