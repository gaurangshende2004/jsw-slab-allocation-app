import streamlit as st
import pandas as pd
import numpy as np
from io import BytesIO
import matplotlib.pyplot as plt

# Try to import statsmodels; if unavailable we'll fallback to linear regression
try:
    from statsmodels.tsa.arima.model import ARIMA
    from statsmodels.tools.sm_exceptions import ConvergenceWarning
    import warnings
    warnings.simplefilter('ignore', ConvergenceWarning)
    STATSMODELS_AVAILABLE = True
except Exception:
    STATSMODELS_AVAILABLE = False
    from sklearn.linear_model import LinearRegression

# ---------------------------------------------------------
# PAGE CONFIG & BASIC STYLING
# ---------------------------------------------------------
st.set_page_config(
    page_title="JSW Slab Allocation System",
    layout="wide"
)

def inject_css():
    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    /* ================= Base App ================= */
    .stApp {
        background: #0f172a;
        color: #e5e7eb;
        font-family: 'Inter', sans-serif;
    }

    /* ================= Header ================= */
    .header {
        background: linear-gradient(90deg, #0f172a, #1e293b);
        padding: 28px 32px;
        border-radius: 18px;
        margin-bottom: 28px;
        box-shadow: 0 10px 25px rgba(0,0,0,0.35);
    }
    .header h1 {
        margin: 0;
        color: #f8fafc;
        font-weight: 700;
    }
    .header p {
        margin-top: 6px;
        color: #94a3b8;
    }

    /* ================= Section Cards ================= */
    .section-card {
        background: #020617;
        border-radius: 18px;
        padding: 24px 26px;
        margin-bottom: 22px;
        border: 1px solid rgba(148,163,184,0.15);
        box-shadow: 0 8px 20px rgba(0,0,0,0.35);
    }
    .section-title {
        font-size: 1.25rem;
        font-weight: 600;
        margin-bottom: 16px;
        color: #38bdf8;
    }

    /* ================= Sidebar ================= */
    section[data-testid="stSidebar"] {
        background: #020617;
        border-right: 1px solid rgba(148,163,184,0.2);
    }
    section[data-testid="stSidebar"] * {
        color: #e5e7eb !important;
    }

    /* ================= Buttons ================= */
    .stButton>button {
        background: linear-gradient(90deg, #38bdf8, #0ea5e9);
        color: #020617;
        border-radius: 12px;
        padding: 0.6rem 1.4rem;
        border: none;
        font-weight: 600;
        transition: all 0.2s ease;
    }
    .stButton>button:hover {
        transform: translateY(-1px);
        box-shadow: 0 6px 18px rgba(56,189,248,0.4);
    }

    /* ================= Tables ================= */
    .stDataFrame, .stTable {
        background: #020617;
        border-radius: 14px;
    }

    /* =================================================
       🟢 Highlight Customer Input Parameters
       ================================================= */
    label:has(+ div[data-baseweb="select"]) {
        color: #22c55e !important;
        font-weight: 600;
    }
    div[data-baseweb="select"] > div {
        border: 2px solid #22c55e !important;
        border-radius: 10px;
    }
    label:has(+ div input) {
        color: #22c55e !important;
        font-weight: 600;
    }
    div[data-baseweb="select"] span {
        color: #22c55e !important;
        font-weight: 600;
    }
    div[data-baseweb="select"] svg {
        fill: #22c55e !important;
    }
    div[data-baseweb="select"] > div:focus-within {
        box-shadow: 0 0 0 2px rgba(34,197,94,0.5);
    }
    ul[role="listbox"] li[aria-selected="true"] {
        background-color: rgba(34,197,94,0.15) !important;
        color: #22c55e !important;
        font-weight: 600;
    }

    /* =================================================
       🟢 Final Allocation KPI Metrics Highlight
       ================================================= */
    div[data-testid="stMetric"] {
        background: #020617;
        border-radius: 14px;
        padding: 14px;
        border: 2px solid #22c55e !important;
        box-shadow: 0 0 12px rgba(34,197,94,0.4);
    }
    div[data-testid="stMetric"] label {
        color: #22c55e !important;
        font-weight: 600;
    }
    div[data-testid="stMetric"] div {
        color: #eafff3 !important;
        font-weight: 700;
    }

    /* =================================================
       🟢 Forecast Table Highlight
       ================================================= */
    .stTable table {
        border-collapse: collapse;
        width: 100%;
    }
    .stTable thead tr th {
        background-color: rgba(34,197,94,0.2) !important;
        color: #22c55e !important;
        font-weight: 700;
        border: 1px solid #22c55e !important;
        text-align: center;
    }
    .stTable tbody tr td {
        color: #eafff3 !important;
        font-weight: 600;
        border: 1px solid rgba(34,197,94,0.4) !important;
        text-align: center;
    }

    </style>
    """, unsafe_allow_html=True)





inject_css()

# ---------------------------------------------------------
# DATA LOADING
# ---------------------------------------------------------
DATA_PATH = "Slab Data for JSW Project.xlsx"  # file should be next to app.py

@st.cache_data
def load_data(path: str) -> pd.DataFrame:
    try:
        df = pd.read_excel(path)
        return df
    except Exception as e:
        st.error(f"Failed to load Excel file from '{path}'. Error: {e}")
        return pd.DataFrame()

df = load_data(DATA_PATH)
if df.empty:
    st.stop()

# ---------------------------------------------------------
# HEADER
# ---------------------------------------------------------

col1, col2 = st.columns([6, 1])

with col1:
    st.markdown("""
    <div class="header">
        <h1>JSW Slab Allocation Dashboard</h1>
        <p>Smart inventory allocation & sales intelligence platform</p>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.image("logo.png", width=210)


# ---------------------------------------------------------
# SIDEBAR – quick info
# ---------------------------------------------------------
with st.sidebar:
    st.subheader("📁 Data Source")
    st.write(f"Path: `{DATA_PATH}`")
    st.write(f"Rows: {df.shape[0]}, Columns: {df.shape[1]}")
    if st.checkbox("Show sample data"):
        st.dataframe(df.head(30), use_container_width=True)

# ---------------------------------------------------------
# Ensure required columns exist
# ---------------------------------------------------------
required_cols = [
    "THICKNESS", "WIDTH", "LENGTH", "Grade", "MATERIAL",
    "BATCH NO", "Order No", "Weight(in Tons)", "INVOICE DATE"
]
missing = [c for c in required_cols if c not in df.columns]
if missing:
    st.error(f"Missing required columns: {missing}")
    st.stop()

# ---------------------------------------------------------
# CUSTOMER REQUIREMENTS SECTION
# ---------------------------------------------------------
st.markdown('<div class="section-card">', unsafe_allow_html=True)
st.markdown('<div class="section-title">🧾 Customer Requirements</div>', unsafe_allow_html=True)

thickness_options = sorted(df["THICKNESS"].dropna().unique().tolist())
width_options = sorted(df["WIDTH"].dropna().unique().tolist())
length_options = sorted(df["LENGTH"].dropna().unique().tolist())
grade_options = sorted(df["Grade"].dropna().unique().tolist())

c1, c2, c3, c4 = st.columns(4)
with c1:
    thickness_input = st.selectbox("Select Thickness", thickness_options)
with c2:
    width_input = st.selectbox("Select Width", width_options)
with c3:
    length_input = st.selectbox("Select Length", length_options)
with c4:
    grade_input = st.selectbox("Select Grade", grade_options)

st.markdown("<br>", unsafe_allow_html=True)

req_tons = st.number_input(
    "Required Quantity (in Tons)",
    min_value=0.1,
    step=0.1,
    format="%.2f"
)

st.markdown(
    """
    <p class="muted">
    Allocation rule: slabs will first try to match <b>exact</b> THICKNESS, WIDTH, LENGTH, and Grade.
    If exact available quantity is insufficient, the system will apply a fixed tolerance of <b>±10</b>
    on THICKNESS, WIDTH, and LENGTH and try allocation again. If still insufficient, a nearest-fit
    allocation will be attempted (nearest thickness/width/length — grade is ignored in nearest-fit).
    </p>
    """,
    unsafe_allow_html=True
)
st.markdown('</div>', unsafe_allow_html=True)

# ---------------------------------------------------------
# ORIGINAL EXACT ALLOCATION FUNCTION (PRESERVED)
# ---------------------------------------------------------
def allocate_slabs(data: pd.DataFrame,
                   t_val,
                   w_val,
                   l_val,
                   g_val,
                   required_tons: float):
    df_local = data.copy()
    df_local["Weight(in Tons)"] = pd.to_numeric(df_local["Weight(in Tons)"], errors="coerce")
    df_local = df_local.dropna(subset=["Weight(in Tons)"])

    mask = (
        (df_local["THICKNESS"] == t_val) &
        (df_local["WIDTH"] == w_val) &
        (df_local["LENGTH"] == l_val) &
        (df_local["Grade"] == g_val)
    )
    matched = df_local[mask].copy()

    if matched.empty:
        return pd.DataFrame(), 0.0, required_tons

    sort_cols = [c for c in ["BATCH NO","Order No","MATERIAL"] if c in matched.columns]
    if sort_cols:
        matched = matched.sort_values(sort_cols)

    matched = matched.reset_index(drop=True)
    matched["cum_weight"] = matched["Weight(in Tons)"].cumsum()
    meet = matched[matched["cum_weight"] >= required_tons]

    if not meet.empty:
        idx = meet.index[0]
        allocated = matched.loc[:idx]
    else:
        allocated = matched

    return allocated, float(allocated["Weight(in Tons)"].sum()), required_tons

# ---------------------------------------------------------
# TOLERANCE ALLOCATION (±10)
# ---------------------------------------------------------
def allocate_from_candidates(candidates: pd.DataFrame, required_tons: float):
    if candidates is None or candidates.empty:
        return pd.DataFrame(), 0.0
    dfc = candidates.copy()
    dfc["Weight(in Tons)"] = pd.to_numeric(dfc["Weight(in Tons)"], errors="coerce")
    dfc = dfc.dropna(subset=["Weight(in Tons)"])
    sort_cols = [c for c in ["BATCH NO", "Order No", "MATERIAL"] if c in dfc.columns]
    if sort_cols:
        dfc = dfc.sort_values(by=sort_cols)
    dfc = dfc.reset_index(drop=True)
    dfc["cum_weight"] = dfc["Weight(in Tons)"].cumsum()
    meet = dfc[dfc["cum_weight"] >= required_tons]
    if not meet.empty:
        idx = meet.index[0]
        allocated = dfc.loc[:idx]
    else:
        allocated = dfc
    return allocated, float(allocated["Weight(in Tons)"].sum())

# ---------------------------------------------------------
# UPDATED NEAREST-FIT (GRADE IGNORED) - Option C chosen
# ---------------------------------------------------------
import random

def allocate_nearest_fit(data: pd.DataFrame,
                         t_val,
                         w_val,
                         l_val,
                         g_val,
                         required_tons: float):

    df_local = data.copy()

    df_local["Weight(in Tons)"] = pd.to_numeric(
        df_local["Weight(in Tons)"], errors="coerce"
    )
    df_local = df_local.dropna(subset=["Weight(in Tons)"])

    # 🚫 Enforce WIDTH must be same
    df_local = df_local[df_local["WIDTH"] == w_val]

    # 🚫 Remove exact-dimension slabs
    df_local = df_local[
        ~(
            (df_local["THICKNESS"] == t_val) &
            (df_local["LENGTH"] == l_val)
        )
    ]

    if df_local.empty:
        return pd.DataFrame(), 0.0

    # Decide changed parameter PER SLAB (ONLY thickness or length)
    def decide_change(row):
        dt = abs(row["THICKNESS"] - t_val)
        dl = abs(row["LENGTH"] - l_val)

        if dt == 0 and dl == 0:
            return None, None

        # Choose nearest change (tie → thickness)
        if dt > 0 and (dl == 0 or dt <= dl):
            return "THICKNESS", row["THICKNESS"] - t_val
        else:
            return "LENGTH", row["LENGTH"] - l_val

    df_local[["Changed_Parameter", "Change_Value"]] = df_local.apply(
        lambda r: pd.Series(decide_change(r)),
        axis=1
    )

    df_local = df_local[df_local["Changed_Parameter"].notna()]

    if df_local.empty:
        return pd.DataFrame(), 0.0

    # Distance score based on chosen parameter only
    df_local["diff_score"] = df_local["Change_Value"].abs()

    df_local = df_local.sort_values("diff_score").reset_index(drop=True)

    df_local["cum_weight"] = df_local["Weight(in Tons)"].cumsum()
    meet = df_local[df_local["cum_weight"] >= required_tons]

    if not meet.empty:
        idx = meet.index[0]
        allocated = df_local.loc[:idx]
    else:
        allocated = df_local

    cols_to_drop = [c for c in ["Changed_Parameter", "Change_Value", "diff_score"] if c in allocated.columns]
    allocated = allocated.drop(columns=cols_to_drop)

    return allocated, float(allocated["Weight(in Tons)"].sum())



# ---------------------------------------------------------
# ALLOCATION UI & staged flow
# ---------------------------------------------------------
st.markdown('<div class="section-card">', unsafe_allow_html=True)
st.markdown('<div class="section-title">✅ Allocation Result</div>', unsafe_allow_html=True)

if st.button("Allocate Material"):
    with st.spinner("Allocating slabs..."):
        df_available = df.copy()
        remaining = float(req_tons)

        # Stage 1: Exact
        exact_alloc_df, exact_alloc_tons, _ = allocate_slabs(
            df_available, thickness_input, width_input, length_input, grade_input, remaining
        )

        if exact_alloc_df is None or exact_alloc_df.empty:
            exact_alloc_df = pd.DataFrame()
            exact_alloc_tons = 0.0
            st.info("Exact allocation: no exact-match slabs found.")
        else:
            st.success(f"Stage 1: Allocated {exact_alloc_tons:.2f} tons using exact-match slabs.")
            st.markdown("### 📋 Stage 1 — Exact Allocated Slabs")
            st.dataframe(exact_alloc_df.reset_index(drop=True), use_container_width=True, height=250)
            # remove exact allocations from available dataset by index-merge method
            df_available = df_available.reset_index().rename(columns={"index":"_orig_index"})
            merge_cols = [c for c in ["THICKNESS","WIDTH","LENGTH","Grade","Weight(in Tons)","BATCH NO","Order No","MATERIAL"] if c in exact_alloc_df.columns and c in df_available.columns]
            if not merge_cols:
                merge_cols = [c for c in ["THICKNESS","WIDTH","LENGTH","Grade","Weight(in Tons)"] if c in df_available.columns]
            to_drop = df_available.merge(exact_alloc_df[merge_cols].drop_duplicates(), on=merge_cols, how='inner')["_orig_index"].unique().tolist()
            df_available = df.copy().drop(index=to_drop).reset_index(drop=True)
            remaining = max(0.0, remaining - exact_alloc_tons)

        # Stage 2: Tolerance ±20
        tol_alloc_df = pd.DataFrame()
        tol_alloc_tons = 0.0
        if remaining > 0:
            tol = 20
            tol_mask = (
                (df_available["THICKNESS"].between(thickness_input - tol, thickness_input + tol)) &
                (df_available["WIDTH"].between(width_input - tol, width_input + tol)) &
                (df_available["LENGTH"].between(length_input - tol, length_input + tol)) &
                (df_available["Grade"] == grade_input)
            )
            tol_candidates = df_available[tol_mask].copy()
            if tol_candidates.empty:
                st.info("Stage 2: No slabs found within ±10 tolerance range.")
            else:
                tol_alloc_df, tol_alloc_tons = allocate_from_candidates(tol_candidates, remaining)
                if not tol_alloc_df.empty:
                    st.info(f"Stage 2: Allocated {tol_alloc_tons:.2f} tons using ±20 tolerance slabs.")
                    st.markdown("### 📋 Stage 2 — Tolerance Allocated Slabs (±10)")
                    st.dataframe(tol_alloc_df.reset_index(drop=True), use_container_width=True, height=250)
                    # remove tolerance allocated rows
                    df_available = df_available.reset_index().rename(columns={"index":"_orig_index"})
                    merge_cols = [c for c in ["THICKNESS","WIDTH","LENGTH","Grade","Weight(in Tons)","BATCH NO","Order No","MATERIAL"] if c in tol_alloc_df.columns and c in df_available.columns]
                    if not merge_cols:
                        merge_cols = [c for c in ["THICKNESS","WIDTH","LENGTH","Grade","Weight(in Tons)"] if c in df_available.columns]
                    to_drop = df_available.merge(tol_alloc_df[merge_cols].drop_duplicates(), on=merge_cols, how='inner')["_orig_index"].unique().tolist()
                    df_available = df.copy().drop(index=to_drop).reset_index(drop=True)
                    remaining = max(0.0, remaining - tol_alloc_tons)
                else:
                    st.info("Stage 2: Candidates existed but allocation returned 0 tons.")
        
        # Stage 3: Nearest-fit (grade ignored per Option C)
        # Remove slabs with exact same dimensions before nearest-fit
        df_available = df_available[
            ~(
                (df_available["THICKNESS"] == thickness_input) &
                (df_available["WIDTH"] == width_input) &
                (df_available["LENGTH"] == length_input)
            )
        ].reset_index(drop=True)    
        nf_alloc_df = pd.DataFrame()
        nf_alloc_tons = 0.0
        if remaining > 0:
            nf_alloc_df, nf_alloc_tons = allocate_nearest_fit(
                df_available, thickness_input, width_input, length_input, grade_input, remaining
            )
            if nf_alloc_df is None or nf_alloc_df.empty:
                nf_alloc_df = pd.DataFrame()
                nf_alloc_tons = 0.0
                st.info("Stage 3: No nearest-fit slabs could be allocated.")
            else:
                st.info(f"Stage 3: Allocated {nf_alloc_tons:.2f} tons using nearest-fit slabs.")
                st.markdown("### 📋 Stage 3 — Nearest-Fit Allocated Slabs")
                st.dataframe(nf_alloc_df.reset_index(drop=True), use_container_width=True, height=250)
                remaining = max(0.0, remaining - nf_alloc_tons)

        # Final summary & download (inside button block)
        st.markdown("## ✅ Final Allocation Summary")

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Exact Allocated", f"{exact_alloc_tons:.2f} Tons")
        c2.metric("Tolerance Allocated (±10)", f"{tol_alloc_tons:.2f} Tons")
        c3.metric("Nearest-Fit Allocated", f"{nf_alloc_tons:.2f} Tons")
        total_allocated = exact_alloc_tons + tol_alloc_tons + nf_alloc_tons
        c4.metric("Total Allocated", f"{total_allocated:.2f} Tons")

        st.markdown("---")

        c5, c6 = st.columns(2)
        c5.metric("Required Tons", f"{req_tons:.2f}")
        c6.metric("Remaining Shortfall", f"{remaining:.2f}")

        final_alloc = pd.concat([
            exact_alloc_df if not exact_alloc_df.empty else pd.DataFrame(),
            tol_alloc_df if not tol_alloc_df.empty else pd.DataFrame(),
            nf_alloc_df if not nf_alloc_df.empty else pd.DataFrame()
        ], ignore_index=True, sort=False)

        if not final_alloc.empty:
            buffer = BytesIO()
            final_alloc.to_excel(buffer, index=False)
            buffer.seek(0)
            st.download_button(
                label="⬇️ Download Final Allocation (Excel)",
                data=buffer,
                file_name="final_allocation.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )
else:
    st.info("Set the customer requirements above and click **Allocate Material** to see allocation details.")

st.markdown('</div>', unsafe_allow_html=True)  # close allocation section-card

# ---------------------------------------------------------
# SALES FORECASTING SECTION (ARIMA fallback to LR) - Next 2 months
# ---------------------------------------------------------
st.markdown('<div class="section-card">', unsafe_allow_html=True)
st.markdown('<div class="section-title">📈 Sales Forecasting (Next 2 months)</div>', unsafe_allow_html=True)

if st.button("Run Sales Forecast (ARIMA)"):
    with st.spinner("Preparing monthly series and fitting model..."):
        df_dates = df.copy()
        df_dates["INVOICE DATE"] = pd.to_datetime(df_dates["INVOICE DATE"], errors="coerce")
        monthly = df_dates.dropna(subset=["INVOICE DATE"]).set_index("INVOICE DATE")["Weight(in Tons)"].resample("M").sum()
        monthly = monthly.rename_axis("Month").reset_index()
        monthly["Month"] = pd.to_datetime(monthly["Month"])
        monthly = monthly.sort_values("Month").reset_index(drop=True)

        if len(monthly) < 6:
            st.warning("Less than 6 months of data — results may be unreliable.")

        ts = monthly.set_index("Month")["Weight(in Tons)"].astype(float)

        try:
            if STATSMODELS_AVAILABLE:
                model = ARIMA(ts, order=(1,1,1))
                res = model.fit()
                fc = res.get_forecast(steps=2)
                fc_mean = fc.predicted_mean
                fc_ci = fc.conf_int(alpha=0.05)
                last_month = ts.index.max()
                future_months = [last_month + pd.DateOffset(months=1), last_month + pd.DateOffset(months=2)]
                future_str = [d.strftime("%Y-%m") for d in future_months]
                forecast_df = pd.DataFrame({
                    "Month": future_str,
                    "Predicted Tons": fc_mean.values
                    
                })
            else:
                # fallback linear regression
                st.info("statsmodels not found — using linear regression fallback.")
                monthly_numeric = monthly.copy()
                monthly_numeric["t"] = np.arange(len(monthly_numeric))
                X = monthly_numeric[["t"]].values
                y = monthly_numeric["Weight(in Tons)"].values
                lr = LinearRegression().fit(X, y)
                future_t = np.array([[len(monthly_numeric)], [len(monthly_numeric)+1]])
                preds = lr.predict(future_t)
                last_month = monthly_numeric["Month"].max()
                future_months = [last_month + pd.DateOffset(months=1), last_month + pd.DateOffset(months=2)]
                future_str = [d.strftime("%Y-%m") for d in future_months]
                forecast_df = pd.DataFrame({
                    "Month": future_str,
                    "Predicted Tons": preds,
                    "CI Lower": preds,
                    "CI Upper": preds
                })

            st.markdown("### 📋 Forecast Table (Next 2 months)")
            st.table(forecast_df.round(2))

            # smaller compact plot
            fig, ax = plt.subplots(figsize=(4,2))
            ax.plot(ts.index, ts.values, marker="o", label="Historical")
            ax.plot(pd.to_datetime(forecast_df["Month"]), forecast_df["Predicted Tons"].values, marker="x", linestyle="--", label="Forecast")
            try:
                ax.fill_between(pd.to_datetime(forecast_df["Month"]),
                                forecast_df["CI Lower"].values,
                                forecast_df["CI Upper"].values,
                                color='gray', alpha=0.2)
            except Exception:
                pass
            ax.set_xlabel("Month", fontsize=8)
            ax.set_ylabel("Tonnage", fontsize=8)
            ax.legend(fontsize=7)
            plt.xticks(rotation=45, fontsize=7)
            plt.yticks(fontsize=7)
            plt.tight_layout()
            st.pyplot(fig, use_container_width=False)

        except Exception as e:
            st.error(f"Forecasting failed: {e}")

st.markdown('</div>', unsafe_allow_html=True)
