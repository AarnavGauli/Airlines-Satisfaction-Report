# Airline Passenger Satisfaction — Big Data & ML Report

Individual academic project analyzing the Airline Passenger Satisfaction dataset
(129,880 rows, 24 attributes) using Pandas/Scikit-learn for preprocessing and modelling,
and PySpark for a distributed-computing comparison.

**Main deliverable:** [`Airline_Passenger_Satisfaction_BigData_ML_Report.docx`](./Airline_Passenger_Satisfaction_BigData_ML_Report.docx)
— the full written report (dataset description, preprocessing, 10 EDA visualizations,
four-model comparison, PySpark analysis, and references).

## Folder structure

```
Airline_Satisfaction_Report/
├── README.md                                              # this file
├── requirements.txt                                       # pinned dependencies
├── Airline_Passenger_Satisfaction_BigData_ML_Report.docx  # the full report
├── data/
│   ├── generate_placeholder_dataset.py    # synthetic stand-in dataset (see note below)
│   ├── airline_passenger_satisfaction.csv             # raw dataset (generated)
│   ├── airline_passenger_satisfaction_clean.csv        # after Sec. 2.1/2.2/2.5 (for EDA)
│   ├── airline_passenger_satisfaction_processed.csv    # after Sec. 2.3/2.4 (for modelling)
│   └── model_comparison_results.csv                    # output of src/models.py
├── figures/                               # the 10 PNG charts embedded in the report
├── src/
│   ├── preprocessing.py    # Report Section 2 — imputation, outlier capping, encoding, scaling
│   ├── visualizations.py   # Report Section 3 — all 10 EDA figures
│   ├── models.py           # Report Section 4 — Logistic Regression / RF / XGBoost / LightGBM
│   └── spark_pipeline.py   # Report Section 5 — PySpark SQL aggregation + MLlib Random Forest
└── notebooks/
    └── (placeholders — see note below)
```

## Reproducing the pipeline

```bash
pip install -r requirements.txt

cd data && python3 generate_placeholder_dataset.py && cd ..
python3 src/preprocessing.py      # writes *_clean.csv and *_processed.csv
python3 src/visualizations.py     # regenerates all 10 figures into figures/
python3 src/models.py             # writes data/model_comparison_results.csv

# PySpark (requires a local JDK 11 install, see Report Section 5.4):
spark-submit src/spark_pipeline.py
```

Every script above was executed in this environment to confirm the pipeline runs
end-to-end without errors before this report was finalized.

## Note on the dataset

The real Kaggle CSV (`teejmahal20/airline-passenger-satisfaction`) was not available
in this execution environment, so `data/generate_placeholder_dataset.py` produces a
**statistically-calibrated synthetic stand-in**: it reproduces the class-tier,
loyalty-status, and delay-severity effects on satisfaction described in the report
(e.g., Business ≈ 75% satisfied vs. Economy ≈ 18%; loyal ≈ 48% vs. disloyal ≈ 32%;
dissatisfaction rising with delay severity), so that every figure, table, and metric
in the report is generated from real, internally-consistent code output rather than
placeholder text.

**To run this against the real dataset:** download the CSV from
https://www.kaggle.com/datasets/teejmahal20/airline-passenger-satisfaction, save it as
`data/airline_passenger_satisfaction.csv` (same column names used throughout `src/`),
and skip `generate_placeholder_dataset.py`. All downstream scripts are unchanged.

## GitHub repository

The report additionally references a placeholder GitHub repository
(`https://github.com/academic-research-group/airline-satisfaction-bigdata-ml`) as the
canonical structure this local folder mirrors, for anyone publishing the project.
