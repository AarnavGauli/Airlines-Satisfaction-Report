"""
Ten EDA visualizations for the Airline Passenger Satisfaction dataset.
Corresponds to Report Section 3. Each function saves one figure to ../figures/.
Run directly to regenerate every figure referenced in the report.
"""
import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

_BASE_DIR = os.path.dirname(os.path.abspath(__file__))
FIG_DIR = os.path.join(_BASE_DIR, "..", "figures")
DATA_DIR = os.path.join(_BASE_DIR, "..", "data")
SERVICE_COLS = [
    "Inflight wifi service", "Food and drink", "Seat comfort",
    "Inflight entertainment", "On-board service", "Leg room service",
    "Baggage handling", "Checkin service", "Inflight service", "Cleanliness",
]


def fig1_age_satisfaction(df):
    plt.figure(figsize=(9, 5))
    sns.histplot(data=df, x="Age", hue="satisfaction", kde=True, element="step",
                 stat="density", common_norm=False, palette="Set2")
    plt.title("Distribution of Customer Age by Satisfaction Status")
    plt.xlabel("Age (years)"); plt.ylabel("Density")
    plt.tight_layout(); plt.savefig(f"{FIG_DIR}/fig1_age_satisfaction.png", dpi=300); plt.close()


def fig2_class_satisfaction(df):
    plt.figure(figsize=(8, 5))
    sns.countplot(data=df, x="Class", hue="satisfaction", palette="muted",
                  order=["Eco", "Eco Plus", "Business"])
    plt.title("Satisfaction Level by Travel Class")
    plt.xlabel("Class"); plt.ylabel("Passenger Count")
    plt.tight_layout(); plt.savefig(f"{FIG_DIR}/fig2_class_satisfaction.png", dpi=300); plt.close()


def fig3_service_corr(df):
    plt.figure(figsize=(11, 9))
    extra_cols = ["Inflight wifi service", "Online boarding", "Ease of Online booking"]
    cols = list(dict.fromkeys(SERVICE_COLS + extra_cols))
    corr = df[cols].corr()
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", center=0, square=True, linewidths=0.5)
    plt.title("Correlation Matrix of Service Quality Ratings")
    plt.tight_layout(); plt.savefig(f"{FIG_DIR}/fig3_service_corr.png", dpi=300); plt.close()


def fig4_delay_customertype(df):
    plt.figure(figsize=(9, 5))
    sns.boxplot(data=df, x="satisfaction", y="Arrival Delay in Minutes",
                hue="Customer Type", showfliers=False, palette="pastel")
    plt.title("Arrival Delay vs. Satisfaction across Customer Types")
    plt.tight_layout(); plt.savefig(f"{FIG_DIR}/fig4_delay_customertype.png", dpi=300); plt.close()


def fig5_distance_delay(df):
    sample = df.sample(min(5000, len(df)), random_state=42)
    plt.figure(figsize=(9, 6))
    sns.scatterplot(data=sample, x="Flight Distance", y="Arrival Delay in Minutes",
                     hue="satisfaction", alpha=0.4, s=15)
    sns.regplot(data=sample, x="Flight Distance", y="Arrival Delay in Minutes",
                scatter=False, color="black", lowess=True)
    plt.title("Flight Distance vs. Arrival Delay Coloured by Satisfaction")
    plt.tight_layout(); plt.savefig(f"{FIG_DIR}/fig5_distance_delay.png", dpi=300); plt.close()


def fig6_inflight_traveltype(df):
    plt.figure(figsize=(8, 6))
    sns.violinplot(data=df, x="Type of Travel", y="Inflight service",
                    hue="Type of Travel", palette="Set3", inner="quartile", legend=False)
    plt.title("Inflight Service Rating by Type of Travel")
    plt.tight_layout(); plt.savefig(f"{FIG_DIR}/fig6_inflight_traveltype.png", dpi=300); plt.close()


def fig7_customertype_stacked(df):
    ct = pd.crosstab(df["Customer Type"], df["satisfaction"], normalize="index")
    ct.plot(kind="bar", stacked=True, figsize=(7, 5), colormap="viridis")
    plt.title("Satisfaction Ratio by Customer Type")
    plt.ylabel("Proportion"); plt.xticks(rotation=0)
    plt.tight_layout(); plt.savefig(f"{FIG_DIR}/fig7_customertype_stacked.png", dpi=300); plt.close()


def fig8_pairplot_digital(df):
    digital_cols = ["Inflight wifi service", "Online boarding", "Ease of Online booking"]
    g = sns.pairplot(df.sample(min(3000, len(df)), random_state=42), vars=digital_cols,
                      hue="satisfaction", palette="husl", diag_kind="kde", plot_kws={"alpha": 0.4})
    g.fig.suptitle("Digital Service Interactions", y=1.02)
    g.savefig(f"{FIG_DIR}/fig8_pairplot_digital.png", dpi=300); plt.close("all")


def fig9_facet_entertainment_comfort(df):
    g = sns.FacetGrid(df, col="Class", height=4.5, col_order=["Eco", "Eco Plus", "Business"])
    g.map_dataframe(sns.scatterplot, x="Seat comfort", y="Inflight entertainment",
                     hue="satisfaction", alpha=0.3, s=12)
    g.add_legend()
    g.fig.suptitle("Entertainment vs. Seat Comfort by Class", y=1.05)
    g.savefig(f"{FIG_DIR}/fig9_facet_entertainment_comfort.png", dpi=300); plt.close("all")


def fig10_friction_points(df):
    df = df.copy()
    df["Delay_Category"] = pd.cut(
        df["Total_Delay"], bins=[-1, 0, 15, 60, 180, 3200],
        labels=["No Delay", "Minor (1-15m)", "Moderate (16-60m)", "Major (61-180m)", "Severe (>180m)"])
    friction = df.groupby("Delay_Category", observed=True)["satisfaction"].apply(
        lambda s: (s == "neutral or dissatisfied").mean()).sort_values(ascending=False)
    plt.figure(figsize=(8, 5))
    sns.barplot(x=friction.values, y=friction.index, hue=friction.index, palette="rocket", legend=False)
    plt.title("Dissatisfaction Rate by Operational Delay Category")
    plt.xlabel("Dissatisfaction Rate")
    plt.tight_layout(); plt.savefig(f"{FIG_DIR}/fig10_friction_points.png", dpi=300); plt.close()


ALL_FIGURES = [
    fig1_age_satisfaction, fig2_class_satisfaction, fig3_service_corr, fig4_delay_customertype,
    fig5_distance_delay, fig6_inflight_traveltype, fig7_customertype_stacked, fig8_pairplot_digital,
    fig9_facet_entertainment_comfort, fig10_friction_points,
]

if __name__ == "__main__":
    sns.set_theme(style="whitegrid", palette="Set2")
    # Uses the CLEAN (unencoded, unscaled) dataset so category labels and delay
    # minutes remain human-readable in the plots - see src/preprocessing.py.
    data = pd.read_csv(os.path.join(DATA_DIR, "airline_passenger_satisfaction_clean.csv"))
    for fn in ALL_FIGURES:
        fn(data)
        print("saved:", fn.__name__)
