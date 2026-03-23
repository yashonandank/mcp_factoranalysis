import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from factor_analyzer import FactorAnalyzer
import io
import warnings
warnings.filterwarnings("ignore")

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Factor Analysis · Goizueta",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Serif+Display:ital@0;1&family=DM+Sans:wght@300;400;500;600&display=swap');

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
}

/* App background */
.stApp {
    background: #f7f5f0;
}

/* Sidebar */
[data-testid="stSidebar"] {
    background: #1a1a2e;
    border-right: none;
}
[data-testid="stSidebar"] * {
    color: #e8e4dc !important;
}
[data-testid="stSidebar"] .stSelectbox label,
[data-testid="stSidebar"] .stMultiSelect label,
[data-testid="stSidebar"] .stSlider label {
    color: #a8a4a0 !important;
    font-size: 0.78rem;
    letter-spacing: 0.08em;
    text-transform: uppercase;
}

/* Hero header */
.hero {
    background: linear-gradient(135deg, #1a1a2e 0%, #16213e 60%, #0f3460 100%);
    border-radius: 16px;
    padding: 2.5rem 2.8rem;
    margin-bottom: 2rem;
    position: relative;
    overflow: hidden;
}
.hero::before {
    content: '';
    position: absolute;
    top: -40px; right: -40px;
    width: 220px; height: 220px;
    background: radial-gradient(circle, rgba(233,196,106,0.18) 0%, transparent 70%);
    border-radius: 50%;
}
.hero-title {
    font-family: 'DM Serif Display', serif;
    font-size: 2.4rem;
    color: #f4ede4;
    margin: 0 0 0.4rem 0;
    line-height: 1.1;
}
.hero-subtitle {
    color: #e9c46a;
    font-size: 0.88rem;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    font-weight: 500;
    margin: 0;
}
.hero-desc {
    color: #a8b2c8;
    font-size: 0.95rem;
    margin-top: 0.8rem;
    max-width: 520px;
    line-height: 1.6;
}

/* Step cards */
.step-card {
    background: white;
    border-radius: 12px;
    padding: 1.6rem 1.8rem;
    margin-bottom: 1.4rem;
    border: 1px solid #ece8e1;
    box-shadow: 0 2px 12px rgba(0,0,0,0.04);
}
.step-label {
    font-size: 0.72rem;
    letter-spacing: 0.14em;
    text-transform: uppercase;
    color: #e9c46a;
    font-weight: 600;
    margin-bottom: 0.3rem;
}
.step-title {
    font-family: 'DM Serif Display', serif;
    font-size: 1.35rem;
    color: #1a1a2e;
    margin: 0 0 0.6rem 0;
}
.step-desc {
    color: #6b7280;
    font-size: 0.88rem;
    line-height: 1.55;
    margin: 0;
}

/* Insight boxes */
.insight-box {
    background: linear-gradient(135deg, #f0f9f4, #e8f5ec);
    border-left: 4px solid #2d9e6b;
    border-radius: 0 10px 10px 0;
    padding: 1rem 1.2rem;
    margin: 0.8rem 0;
    font-size: 0.9rem;
    color: #1a4731;
    line-height: 1.5;
}
.warning-box {
    background: #fff8e7;
    border-left: 4px solid #e9c46a;
    border-radius: 0 10px 10px 0;
    padding: 1rem 1.2rem;
    margin: 0.8rem 0;
    font-size: 0.9rem;
    color: #7a5c00;
    line-height: 1.5;
}

/* Metric pill */
.metric-row {
    display: flex;
    gap: 1rem;
    flex-wrap: wrap;
    margin: 1rem 0;
}
.metric-pill {
    background: #1a1a2e;
    color: #e9c46a;
    border-radius: 999px;
    padding: 0.45rem 1.1rem;
    font-size: 0.88rem;
    font-weight: 600;
    letter-spacing: 0.04em;
}

/* Download button */
.stDownloadButton > button {
    background: #1a1a2e !important;
    color: #e9c46a !important;
    border: none !important;
    border-radius: 8px !important;
    font-weight: 600 !important;
    letter-spacing: 0.04em !important;
    padding: 0.55rem 1.4rem !important;
    font-family: 'DM Sans', sans-serif !important;
}
.stDownloadButton > button:hover {
    background: #0f3460 !important;
    transform: translateY(-1px);
    box-shadow: 0 4px 16px rgba(0,0,0,0.18) !important;
}

/* Divider */
hr { border-color: #ece8e1; margin: 1.5rem 0; }

/* Multiselect tags */
[data-baseweb="tag"] {
    background-color: #1a1a2e !important;
}
</style>
""", unsafe_allow_html=True)

# ── Hero ──────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero">
    <p class="hero-subtitle">Goizueta Business School · Marketing Analytics</p>
    <h1 class="hero-title">Factor Analysis</h1>
    <p class="hero-desc">
        Reduce your survey variables into meaningful underlying factors using
        PCA with Varimax rotation — no statistics background required.
    </p>
</div>
""", unsafe_allow_html=True)

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚙️ Configuration")
    st.markdown("---")

    uploaded_file = st.file_uploader(
        "Upload your dataset",
        type=["xlsx", "csv"],
        help="Upload an Excel (.xlsx) or CSV file exported from Qualtrics or similar."
    )

    st.markdown("---")
    st.markdown("### 📖 How it works")
    st.markdown("""
1. **Upload** your survey data  
2. **Select** the question columns to analyze  
3. **Drop** any columns that don't fit  
4. **Run** — we handle the math  
5. **Name** your factors  
6. **Download** scores for cluster analysis
    """)
    st.markdown("---")
    st.markdown("<small style='color:#666'>Built for MKT 557 · Spring 2025</small>", unsafe_allow_html=True)

# ── Main logic ────────────────────────────────────────────────────────────────
if uploaded_file is None:
    # Welcome / instruction state
    col1, col2, col3 = st.columns(3)
    cards = [
        ("01", "Upload", "Drop your Excel or CSV file in the sidebar. Works with any Qualtrics export."),
        ("02", "Configure", "Pick which question columns to include, drop outliers, let the app detect factors."),
        ("03", "Download", "Export factor scores as a clean Excel file ready for cluster analysis."),
    ]
    for col, (num, title, desc) in zip([col1, col2, col3], cards):
        with col:
            st.markdown(f"""
            <div class="step-card">
                <div class="step-label">Step {num}</div>
                <div class="step-title">{title}</div>
                <p class="step-desc">{desc}</p>
            </div>
            """, unsafe_allow_html=True)
    st.info("👈 Upload your dataset in the sidebar to get started.")
    st.stop()

# ── Load data ─────────────────────────────────────────────────────────────────
@st.cache_data
def load_data(file):
    if file.name.endswith(".csv"):
        return pd.read_csv(file)
    else:
        return pd.read_excel(file)

df_raw = load_data(uploaded_file)
st.success(f"✅ Loaded **{uploaded_file.name}** — {df_raw.shape[0]} rows, {df_raw.shape[1]} columns")

# ── STEP 1: Select prefix / columns ──────────────────────────────────────────
st.markdown("""
<div class="step-card">
    <div class="step-label">Step 1</div>
    <div class="step-title">Select Your Variables</div>
    <p class="step-desc">
        Choose the survey question prefix (e.g. <code>Q8_</code>, <code>Q41_</code>) or 
        manually pick individual columns. Only numeric columns will be used.
    </p>
</div>
""", unsafe_allow_html=True)

numeric_cols = df_raw.select_dtypes(include=[np.number]).columns.tolist()

# Detect common prefixes
prefixes = sorted(set(
    col.split("_")[0] + "_"
    for col in numeric_cols
    if "_" in col
))

col_left, col_right = st.columns([1, 2])
with col_left:
    prefix_option = st.selectbox(
        "Quick-select by prefix",
        ["(choose manually)"] + prefixes,
        help="Select a prefix to auto-fill all matching columns"
    )

if prefix_option != "(choose manually)":
    default_cols = [c for c in numeric_cols if c.startswith(prefix_option)]
else:
    default_cols = []

with col_right:
    selected_cols = st.multiselect(
        "Columns to include in factor analysis",
        options=numeric_cols,
        default=default_cols,
        help="Select all the survey items you want to factor analyze together"
    )

if len(selected_cols) < 3:
    st.warning("⚠️ Please select at least 3 columns to run factor analysis.")
    st.stop()

df_work = df_raw[selected_cols].dropna()
st.markdown(f"""
<div class="insight-box">
    📋 Working with <strong>{len(selected_cols)} variables</strong> across 
    <strong>{len(df_work)} complete responses</strong>.
</div>
""", unsafe_allow_html=True)

# ── STEP 2: Drop columns ──────────────────────────────────────────────────────
st.markdown("""
<div class="step-card">
    <div class="step-label">Step 2</div>
    <div class="step-title">Drop Poorly-Fitting Columns (Optional)</div>
    <p class="step-desc">
        After reviewing your initial results, you can remove columns that don't load 
        clearly onto any factor — just like the teaching team did in the example notebooks.
    </p>
</div>
""", unsafe_allow_html=True)

drop_cols = st.multiselect(
    "Columns to drop",
    options=selected_cols,
    default=[],
    help="Remove items that cross-load or don't fit well. You can always come back and change this."
)

final_cols = [c for c in selected_cols if c not in drop_cols]
df_final = df_work[final_cols]

if len(final_cols) < 3:
    st.warning("⚠️ You need at least 3 columns after dropping. Please unselect some.")
    st.stop()

# ── STEP 3: PCA & Kaiser Criterion ───────────────────────────────────────────
st.markdown("""
<div class="step-card">
    <div class="step-label">Step 3</div>
    <div class="step-title">Determine the Number of Factors</div>
    <p class="step-desc">
        We use <strong>Kaiser's Criterion</strong>: keep factors whose eigenvalue is above 1. 
        The scree plot below shows this visually — the red dashed line is your cutoff.
    </p>
</div>
""", unsafe_allow_html=True)

# Standardize
scaler = StandardScaler()
df_scaled = scaler.fit_transform(df_final)

# PCA for eigenvalues
pca = PCA()
pca.fit(df_scaled)
eigenvalues = pca.explained_variance_
optimal_components = int(np.sum(eigenvalues > 1))

# ── Scree plot ────────────────────────────────────────────────────────────────
fig_scree, ax = plt.subplots(figsize=(8, 4))
fig_scree.patch.set_facecolor('#f7f5f0')
ax.set_facecolor('#f7f5f0')

bars = ax.bar(
    range(1, len(eigenvalues) + 1), eigenvalues,
    color='#1a1a2e', alpha=0.85, width=0.6, zorder=3
)
# Highlight factors above threshold
for i, (bar, ev) in enumerate(zip(bars, eigenvalues)):
    if ev > 1:
        bar.set_color('#0f3460')
        bar.set_alpha(1.0)

ax.axhline(y=1, color='#e9c46a', linestyle='--', linewidth=2,
           label="Kaiser's Criterion (Eigenvalue = 1)", zorder=4)
ax.set_xlabel('Principal Component', fontsize=11, color='#333')
ax.set_ylabel('Eigenvalue', fontsize=11, color='#333')
ax.set_title('Scree Plot', fontsize=14, fontweight='bold', color='#1a1a2e', pad=12)
ax.legend(fontsize=9, framealpha=0.9)
ax.grid(axis='y', alpha=0.3, zorder=0)
ax.spines[['top', 'right']].set_visible(False)
ax.spines[['left', 'bottom']].set_color('#ccc')
ax.tick_params(colors='#555')
plt.tight_layout()
st.pyplot(fig_scree)

st.markdown(f"""
<div class="insight-box">
    🎯 Kaiser's Criterion suggests <strong>{optimal_components} factor(s)</strong> — 
    these are the components with eigenvalue above 1.
</div>
""", unsafe_allow_html=True)

# Allow override
n_factors = st.slider(
    "Override number of factors (optional)",
    min_value=1,
    max_value=max(2, len(final_cols) - 1),
    value=optimal_components,
    help="The app recommends the number above, but you can adjust based on your judgment."
)

# ── STEP 4: Varimax Factor Analysis ──────────────────────────────────────────
st.markdown("""
<div class="step-card">
    <div class="step-label">Step 4</div>
    <div class="step-title">Varimax Rotation & Factor Loadings</div>
    <p class="step-desc">
        Varimax rotation makes factors easier to interpret by maximizing the difference 
        between high and low loadings. Each row shows how strongly a variable relates to each factor.
        Values above <strong>0.5</strong> are considered strong loadings.
    </p>
</div>
""", unsafe_allow_html=True)

# Run factor analyzer
fa = FactorAnalyzer(n_factors=n_factors, rotation='varimax')
fa.fit(df_scaled)
loadings = fa.loadings_
loadings_df = pd.DataFrame(
    loadings,
    index=final_cols,
    columns=[f"Factor {i+1}" for i in range(n_factors)]
)

# Heatmap
fig_heat, ax2 = plt.subplots(figsize=(max(6, n_factors * 2.5), max(5, len(final_cols) * 0.55)))
fig_heat.patch.set_facecolor('white')

cmap = sns.diverging_palette(220, 20, as_cmap=True)
sns.heatmap(
    loadings_df,
    cmap="RdBu_r",
    annot=True,
    fmt=".2f",
    linewidths=0.5,
    linecolor='#f0ece6',
    vmin=-1, vmax=1,
    ax=ax2,
    annot_kws={"size": 9, "weight": "bold"}
)
ax2.set_title("Factor Loadings Heatmap (Varimax Rotation)",
              fontsize=13, fontweight='bold', color='#1a1a2e', pad=14)
ax2.set_xlabel("Factors", fontsize=10, color='#555')
ax2.set_ylabel("Survey Variables", fontsize=10, color='#555')
ax2.tick_params(axis='x', rotation=0, labelsize=9)
ax2.tick_params(axis='y', rotation=0, labelsize=8)
plt.tight_layout()
st.pyplot(fig_heat)

st.markdown("""
<div class="warning-box">
    💡 <strong>How to read this:</strong> Dark red = strong positive loading (variable strongly belongs to this factor). 
    Dark blue = strong negative loading. Light colors near 0 = the variable doesn't relate to this factor much. 
    Each variable should ideally load strongly on <em>one</em> factor.
</div>
""", unsafe_allow_html=True)

# Show loadings table with highlighting
with st.expander("📋 View full loadings table"):
    def highlight_high(val):
        if abs(val) >= 0.5:
            return 'background-color: #fff3cd; font-weight: bold'
        return ''
    st.dataframe(loadings_df.style.applymap(highlight_high).format("{:.3f}"))

# ── STEP 5: Name your factors ─────────────────────────────────────────────────
st.markdown("""
<div class="step-card">
    <div class="step-label">Step 5</div>
    <div class="step-title">Name Your Factors</div>
    <p class="step-desc">
        Look at the heatmap above — which variables load strongly on each factor? 
        Give each factor a meaningful name that captures the theme.
    </p>
</div>
""", unsafe_allow_html=True)

factor_names = []
cols_name = st.columns(n_factors)
for i, col in enumerate(cols_name):
    with col:
        # Find the top loading variables for this factor as hints
        top_vars = loadings_df[f"Factor {i+1}"].abs().nlargest(3).index.tolist()
        hint = ", ".join([v.split("_", 1)[-1] if "_" in v else v for v in top_vars])
        name = st.text_input(
            f"Factor {i+1} name",
            value=f"Factor_{i+1}",
            help=f"Top loading variables: {hint}"
        )
        factor_names.append(name)
        st.caption(f"📌 Top items: {hint}")

# ── STEP 6: Factor Scores ─────────────────────────────────────────────────────
st.markdown("""
<div class="step-card">
    <div class="step-label">Step 6</div>
    <div class="step-title">Factor Scores</div>
    <p class="step-desc">
        Each respondent now gets a score on every factor. These scores summarize 
        how strongly each person aligns with each underlying theme — and are 
        ready to use as inputs for cluster analysis.
    </p>
</div>
""", unsafe_allow_html=True)

factor_scores = fa.transform(df_scaled)
scores_df = pd.DataFrame(factor_scores, columns=factor_names)

# Add uuid/id if available
id_cols = [c for c in ['uuid', 'record', 'id', 'ID', 'ResponseId'] if c in df_raw.columns]
if id_cols:
    # Align index with df_work (which dropped NAs)
    id_series = df_raw.loc[df_work.index, id_cols[0]].reset_index(drop=True)
    scores_df.insert(0, id_cols[0], id_series.values)

st.dataframe(scores_df.head(10), use_container_width=True)
st.caption(f"Showing first 10 of {len(scores_df)} rows · {n_factors} factors")

# ── STEP 7: Variance explained ────────────────────────────────────────────────
ev_df = pd.DataFrame({
    "Factor": factor_names,
    "Eigenvalue": eigenvalues[:n_factors].round(3),
    "Variance Explained (%)": (eigenvalues[:n_factors] / len(final_cols) * 100).round(1)
})
st.table(ev_df.set_index("Factor"))

total_var = (eigenvalues[:n_factors] / len(final_cols) * 100).sum()
st.markdown(f"""
<div class="insight-box">
    📊 Your {n_factors} factor(s) explain <strong>{total_var:.1f}%</strong> of the 
    total variance in your selected variables.
</div>
""", unsafe_allow_html=True)

# ── Download ──────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown("### 📥 Download Results")

col_dl1, col_dl2 = st.columns(2)

with col_dl1:
    # Factor scores
    buf1 = io.BytesIO()
    scores_df.to_excel(buf1, index=False)
    st.download_button(
        label="⬇️ Download Factor Scores (.xlsx)",
        data=buf1.getvalue(),
        file_name="factor_scores.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        help="Use this file as input for the Cluster Analysis app"
    )
    st.caption("Factor scores for each respondent — ready for cluster analysis")

with col_dl2:
    # Loadings table
    buf2 = io.BytesIO()
    loadings_df.to_excel(buf2)
    st.download_button(
        label="⬇️ Download Factor Loadings (.xlsx)",
        data=buf2.getvalue(),
        file_name="factor_loadings.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )
    st.caption("Variable weights on each factor — useful for your report")

# ── Full merged output ────────────────────────────────────────────────────────
with st.expander("⬇️ Download full dataset with factor scores appended"):
    try:
        merged = df_raw.copy()
        # Drop original factor columns and add scores
        merged = merged.drop(columns=[c for c in final_cols if c in merged.columns], errors='ignore')
        scores_reset = scores_df.reset_index(drop=True)
        merged = pd.concat([merged.reset_index(drop=True), scores_reset], axis=1)
        buf3 = io.BytesIO()
        merged.to_excel(buf3, index=False)
        st.download_button(
            label="⬇️ Download Full Dataset with Factor Scores",
            data=buf3.getvalue(),
            file_name="full_data_with_factors.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        )
        st.caption("Your original dataset with factor score columns appended — use this for cross-tabulations")
    except Exception as e:
        st.warning(f"Could not merge: {e}")

st.markdown("---")
st.markdown(
    "<small style='color:#aaa'>Factor Analysis App · Goizueta Business School · "
    "Powered by scikit-learn & factor_analyzer</small>",
    unsafe_allow_html=True
)
