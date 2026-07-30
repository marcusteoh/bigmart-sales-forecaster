"""BigMart Sales Forecaster - Streamlit web application.

Loads the exported scikit-learn Pipeline from models/ and serves two things to a
retail planner:

  1. A revenue forecast for one product at one outlet, with an 80% planning range.
  2. A price-sensitivity curve, so the user can see how the forecast responds to price.

The pipeline carries its own cleaning, imputation and encoding, so this file never
re-implements preprocessing - it only collects raw inputs and calls .predict().

Currency: the source dataset does not state its units, so amounts are labelled as
rupees and the SGD figures are a display-only conversion at a rate the user sets in
the sidebar. Nothing in the model depends on the currency - a monotonic change of
units leaves every metric and every comparison identical.

Run locally:  streamlit run streamlit_app.py
"""
from pathlib import Path

import json
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

# joblib.load() rebuilds the custom transformers by importing them from this module,
# so the import must be present even though nothing here calls it directly.
import bigmart_preprocessing  # noqa: F401
import joblib

APP_DIR = Path(__file__).parent
MODELS_DIR = APP_DIR / "models"

# Display units. The dataset's own figures are treated as rupees; SGD is derived for
# readability only, never fed back into the model.
CURRENCY_SYMBOL = "Rs"
DEFAULT_RUPEES_PER_SGD = 105.0

st.set_page_config(
    page_title="BigMart Sales Forecaster",
    page_icon="🛒",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Light styling only - keeps the app readable in both Streamlit themes rather than
# hard-coding colours that break in dark mode.
st.markdown(
    """
    <style>
      .main .block-container { padding-top: 2rem; max-width: 1200px; }
      div[data-testid="stMetricValue"] { font-size: 2.1rem; }
      .forecast-range {
          border-left: 4px solid #4C72B0;
          padding: 0.6rem 1rem;
          margin-top: 0.5rem;
          background: rgba(76, 114, 176, 0.08);
          border-radius: 0 6px 6px 0;
      }
      .footnote { font-size: 0.85rem; opacity: 0.75; }
      .sgd { opacity: 0.72; font-size: 0.92rem; }
    </style>
    """,
    unsafe_allow_html=True,
)


# --------------------------------------------------------------------------------
# Artefact loading
# --------------------------------------------------------------------------------
@st.cache_resource(show_spinner="Loading forecasting model...")
def load_artifacts():
    """Load the pipelines and metadata once per session.

    Returns (point_model, lower_model, upper_model, metadata). Raises FileNotFoundError
    with a readable message if the export step in the notebook has not been run.
    """
    required = {
        "point": MODELS_DIR / "best_pipeline.joblib",
        "lower": MODELS_DIR / "quantile_lower_pipeline.joblib",
        "upper": MODELS_DIR / "quantile_upper_pipeline.joblib",
        "meta": MODELS_DIR / "feature_metadata.json",
    }
    missing = [str(p.relative_to(APP_DIR)) for p in required.values() if not p.exists()]
    if missing:
        raise FileNotFoundError(
            "Missing model files: " + ", ".join(missing)
            + ". Run section 7.8 of BigMart_Sales_Prediction.ipynb to export them."
        )

    with open(required["meta"], encoding="utf-8") as fh:
        meta = json.load(fh)

    return (
        joblib.load(required["point"]),
        joblib.load(required["lower"]),
        joblib.load(required["upper"]),
        meta,
    )


try:
    point_model, lower_model, upper_model, META = load_artifacts()
except Exception as exc:  # surfaced to the user rather than a raw traceback
    st.error("The forecasting model could not be loaded.")
    st.caption(str(exc))
    st.stop()

METRICS = META["metrics"]
OPTIONS = META["category_options"]
RANGES = META["numeric_ranges"]
APP_COLUMNS = META["app_input_columns"]


def predict_frame(frame: pd.DataFrame) -> pd.DataFrame:
    """Score a raw input frame, returning point forecast and 80% interval.

    Predictions are clipped at zero: a negative revenue forecast is not a meaningful
    business output even when the underlying regressor produces one.
    """
    point = np.clip(point_model.predict(frame), 0, None)
    lower = np.clip(lower_model.predict(frame), 0, None)
    upper = np.clip(upper_model.predict(frame), 0, None)
    # A quantile model can occasionally cross; enforce lower <= point <= upper so the
    # displayed range is always coherent.
    lower = np.minimum(lower, point)
    upper = np.maximum(upper, point)
    return pd.DataFrame(
        {"forecast": point, "lower_bound": lower, "upper_bound": upper},
        index=frame.index,
    )


# --------------------------------------------------------------------------------
# Sidebar - product and outlet inputs, then the display-currency control
# --------------------------------------------------------------------------------
with st.sidebar:
    st.header("Forecast inputs")
    st.caption("The forecast updates as you change any value.")

    st.subheader("Product")
    item_type = st.selectbox(
        "Product category", OPTIONS["Item_Type"],
        index=OPTIONS["Item_Type"].index("Snack Foods")
        if "Snack Foods" in OPTIONS["Item_Type"] else 0,
        help="The department the product sits in.",
    )

    # Fat content is meaningless for non-consumables, and the model was trained with a
    # dedicated 'Non-Edible' label for them. Mirror that rule here so the user cannot
    # submit a combination the model never saw.
    non_edible_types = list(bigmart_preprocessing.NON_EDIBLE_ITEM_TYPES)
    if item_type in non_edible_types:
        fat_content = bigmart_preprocessing.NON_EDIBLE_LABEL
        st.selectbox("Fat content", [fat_content], disabled=True,
                     help="Not applicable to household and hygiene products.")
    else:
        edible_options = [o for o in OPTIONS["Item_Fat_Content"]
                          if o != bigmart_preprocessing.NON_EDIBLE_LABEL]
        fat_content = st.radio("Fat content", edible_options, horizontal=True)

    item_mrp = st.slider(
        f"Retail price ({CURRENCY_SYMBOL})",
        min_value=float(round(RANGES["Item_MRP"]["min"], 2)),
        max_value=float(round(RANGES["Item_MRP"]["max"], 2)),
        value=float(round(RANGES["Item_MRP"]["median"], 2)),
        step=1.0,
        help="Maximum retail price. This is the strongest single driver of revenue.",
    )

    item_weight = st.number_input(
        "Unit weight (kg)",
        min_value=0.0, max_value=100.0,
        value=float(round(RANGES["Item_Weight"]["median"], 2)),
        step=0.5,
        help=f"Training data ranged {RANGES['Item_Weight']['min']:.2f}"
             f"-{RANGES['Item_Weight']['max']:.2f} kg.",
    )

    item_visibility = st.slider(
        "Shelf visibility (share of display area)",
        min_value=0.0,
        max_value=float(round(RANGES["Item_Visibility"]["max"], 3)),
        value=float(round(RANGES["Item_Visibility"]["median"], 3)),
        step=0.005, format="%.3f",
        help="Proportion of the outlet's total display area given to this product.",
    )

    st.subheader("Outlet")
    outlet_type = st.selectbox(
        "Store format", OPTIONS["Outlet_Type"],
        index=OPTIONS["Outlet_Type"].index("Supermarket Type1")
        if "Supermarket Type1" in OPTIONS["Outlet_Type"] else 0,
        help="The second strongest driver after price.",
    )
    outlet_size = st.selectbox(
        "Store size", OPTIONS["Outlet_Size"],
        help="'Unknown' is a real category - three outlets have no size on record.",
    )
    outlet_location = st.selectbox(
        "City tier", OPTIONS["Outlet_Location_Type"],
        help="Has little effect on the forecast: in this data the city tier is largely "
             "determined by the store format, so it adds no independent signal.",
    )
    outlet_year = st.selectbox(
        "Year the outlet opened", META["establishment_years"],
        index=len(META["establishment_years"]) // 2,
    )

    st.divider()
    st.subheader("Display currency")
    # Exposed as an input rather than hard-coded: an embedded rate silently goes stale,
    # and the assumption belongs to whoever is reading the forecast.
    rupees_per_sgd = st.number_input(
        f"{CURRENCY_SYMBOL} per 1 SGD",
        min_value=1.0, max_value=1000.0,
        value=DEFAULT_RUPEES_PER_SGD, step=0.5,
        help="Used only to display an SGD equivalent. It does not affect the model.",
    )
    st.caption(
        f"Forecasts are produced in the dataset's own units, shown as "
        f"{CURRENCY_SYMBOL}. Set the rate to today's value before quoting SGD figures."
    )


def money(value: float) -> str:
    """Format an amount in the dataset's own currency."""
    return f"{CURRENCY_SYMBOL} {value:,.2f}"


def sgd(value: float) -> str:
    """Convert to SGD for display only, at the user-set rate."""
    return f"S${value / rupees_per_sgd:,.2f}"


# Input validation - warn rather than block, but say plainly what is unusual.
warnings = []
if not RANGES["Item_Weight"]["min"] <= item_weight <= RANGES["Item_Weight"]["max"]:
    warnings.append(
        f"Unit weight {item_weight:.2f} kg is outside the training range "
        f"({RANGES['Item_Weight']['min']:.2f}-{RANGES['Item_Weight']['max']:.2f} kg). "
        "The forecast is an extrapolation."
    )
if item_visibility == 0:
    warnings.append(
        "Shelf visibility of 0 was treated as a recording error during training and is "
        "replaced by the typical value. Set a real share for a more specific forecast."
    )

single_row = pd.DataFrame([{
    "Item_Weight": item_weight,
    "Item_Fat_Content": fat_content,
    "Item_Visibility": item_visibility,
    "Item_Type": item_type,
    "Item_MRP": item_mrp,
    "Outlet_Establishment_Year": outlet_year,
    "Outlet_Size": outlet_size,
    "Outlet_Location_Type": outlet_location,
    "Outlet_Type": outlet_type,
}])[APP_COLUMNS]


# --------------------------------------------------------------------------------
# Header
# --------------------------------------------------------------------------------
st.title("🛒 BigMart Sales Forecaster")
st.markdown(
    "Forecast the revenue a **single product** will generate at a **single outlet**, "
    "so inventory and shelf-space decisions can be made before the selling period "
    "rather than reviewed after it."
)

head = st.columns(4)
head[0].metric("Typical forecast error", money(METRICS["test_mae"]),
               help="Mean absolute error on 1,705 held-out product-outlet pairs.")
head[0].caption(f"≈ {sgd(METRICS['test_mae'])} per decision")
head[1].metric("Improvement vs flat average",
               f"{(1 - METRICS['test_mae'] / METRICS['baseline_mae']) * 100:.0f}%",
               help="Against planning with a single chain-wide average.")
head[1].caption(f"flat average errs by {money(METRICS['baseline_mae'])}")
head[2].metric("Variance explained (R²)", f"{METRICS['test_r2']:.3f}",
               help="Share of sales variation the model accounts for.")
head[2].caption("1.000 would be a perfect forecast")
head[3].metric("Planning range accuracy",
               f"{METRICS['interval_coverage'] * 100:.0f}%",
               help="How often actual sales fell inside the 80% range on held-out data.")
head[3].caption("against an 80% target")

st.divider()


# --------------------------------------------------------------------------------
# Tabs
# --------------------------------------------------------------------------------
tab_forecast, tab_about = st.tabs(["📈 Forecast", "ℹ️ About this model"])

with tab_forecast:
    for message in warnings:
        st.warning(message, icon="⚠️")

    try:
        result = predict_frame(single_row).iloc[0]
    except Exception as exc:
        st.error("The forecast could not be produced for these inputs.")
        st.caption(str(exc))
        st.stop()

    left, right = st.columns([1, 1.4])

    with left:
        st.subheader("Forecast")
        st.metric(
            label=f"{item_type} at a {outlet_type}",
            value=money(result["forecast"]),
            help="Expected revenue for this product at this outlet over the period.",
        )
        st.markdown(
            f'<p class="sgd">≈ <strong>{sgd(result["forecast"])}</strong> '
            f'at {CURRENCY_SYMBOL} {rupees_per_sgd:,.2f} per SGD</p>',
            unsafe_allow_html=True,
        )
        st.markdown(
            f"""<div class="forecast-range">
                <strong>Planning range (80% confidence)</strong><br/>
                {money(result['lower_bound'])} &nbsp;–&nbsp; {money(result['upper_bound'])}
                <br/><span class="sgd">{sgd(result['lower_bound'])} &nbsp;–&nbsp;
                {sgd(result['upper_bound'])}</span>
                </div>""",
            unsafe_allow_html=True,
        )
        st.caption(
            f"On held-out data, actual sales fell inside this range "
            f"{METRICS['interval_coverage'] * 100:.0f}% of the time. Order against the "
            "lower bound to protect cash, the upper bound to protect against stockouts."
        )

        st.markdown("**Your selection**")
        # Values are cast to text before display: a transposed row mixes numbers and
        # strings in one column, which Arrow cannot serialise natively.
        st.dataframe(
            pd.DataFrame({
                "Input": [c.replace("_", " ") for c in single_row.columns],
                "Value": [str(v) for v in single_row.iloc[0]],
            }),
            width="stretch",
            hide_index=True,
        )

    with right:
        st.subheader("How price changes the forecast")
        st.caption(
            "Every other input is held at your current selection, so this isolates the "
            f"effect of price alone. Axes are in {CURRENCY_SYMBOL}."
        )

        # Re-score across the price range to build the sensitivity curve.
        price_grid = np.linspace(RANGES["Item_MRP"]["min"], RANGES["Item_MRP"]["max"], 40)
        grid_frame = pd.concat([single_row] * len(price_grid), ignore_index=True)
        grid_frame["Item_MRP"] = price_grid
        grid_result = predict_frame(grid_frame)

        fig, ax = plt.subplots(figsize=(7, 4.2))
        fig.patch.set_alpha(0)
        ax.patch.set_alpha(0)
        ax.fill_between(price_grid, grid_result["lower_bound"],
                        grid_result["upper_bound"], alpha=0.18, color="#4C72B0",
                        label="80% planning range")
        ax.plot(price_grid, grid_result["forecast"], lw=2.5, color="#4C72B0",
                label="Forecast")
        ax.axvline(item_mrp, color="#C44E52", ls="--", lw=2,
                   label=f"Your price ({CURRENCY_SYMBOL} {item_mrp:,.0f})")
        ax.set_xlabel(f"Retail price ({CURRENCY_SYMBOL})")
        ax.set_ylabel(f"Forecast revenue ({CURRENCY_SYMBOL})")
        ax.set_title("Price sensitivity", fontweight="bold")
        ax.legend(frameon=False, fontsize=9)
        ax.spines[["top", "right"]].set_visible(False)
        ax.tick_params(colors="grey")
        for spine in ax.spines.values():
            spine.set_color("grey")
        ax.xaxis.label.set_color("grey")
        ax.yaxis.label.set_color("grey")
        ax.title.set_color("grey")
        plt.tight_layout()
        st.pyplot(fig, width="stretch")
        plt.close(fig)


with tab_about:
    st.subheader("About this model")
    st.markdown(
        f"""
**Model.** {META['model_name']}, built with scikit-learn {META['sklearn_version']}.
Preprocessing, imputation, encoding and the estimator are exported as a single
pipeline, so a forecast made here follows exactly the same path as one made during
training.

**Performance on 1,705 held-out product-outlet pairs**

| Measure | Model | Planning on a flat average |
|---|---|---|
| Typical error (MAE) | {money(METRICS['test_mae'])} | {money(METRICS['baseline_mae'])} |
| RMSE | {money(METRICS['test_rmse'])} | {money(METRICS['baseline_rmse'])} |
| R² | {METRICS['test_r2']:.3f} | ~0.00 |

At {CURRENCY_SYMBOL} {rupees_per_sgd:,.2f} per SGD, the typical error is about
**{sgd(METRICS['test_mae'])}** per product-outlet decision, against
{sgd(METRICS['baseline_mae'])} for a flat average.

**Currency.** The source dataset does not state its currency. Amounts are shown in the
dataset's own units, labelled {CURRENCY_SYMBOL}, and the SGD figures are a display
conversion at the rate set in the sidebar. The model is unit-agnostic: changing
currency rescales every figure identically and changes no conclusion.

**What drives the forecast.** Retail price and store format account for almost all of
the model's predictive power. Product attributes such as weight, fat content and
category contribute very little once price and format are known — a finding confirmed
by two independent methods in the notebook.

**Limits you should know about.**
- Outlet age is measured against **{META['data_year']}**, the year the data was
  collected. The model has never seen an outlet older than 28 years.
- The data contains no footfall, promotion calendar, competitor pricing or
  stock-on-hand. Roughly 38% of sales variation is therefore unexplained.
- Accuracy is weakest in relative terms for Grocery Stores, and the forecast spread
  widens for high-value lines.
- City tier has almost no effect once store format is known, because the two are
  confounded across the ten outlets in the data.
- Forecasts for a store format not present in the training data are extrapolation.
        """
    )
    st.caption(
        "Tuned hyperparameters: "
        + ", ".join(f"{k}={v}" for k, v in sorted(META["model_params"].items()))
    )

st.divider()
st.markdown(
    '<p class="footnote">BigMart Sales Forecaster · Machine Learning for Developers '
    "(CAI2C08) · Model trained on the BigMart Sales dataset (8,523 product-outlet "
    "records). Amounts in the dataset's own currency units; SGD shown for reference "
    "only.</p>",
    unsafe_allow_html=True,
)
