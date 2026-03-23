import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
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
.stApp {
    background: #f7f5f0;
}
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
    box-shadow: 0 4px 16px rgba(0,0,0,0.18) !important;
}
hr { border-color: #ece8e1; margin: 1.5rem 0; }
</style>
""", unsafe_allow_html=True)

# ── Hero ──────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero">
    <p class="hero-subtitle">Goizueta Business School · Marketing Analytics</p>
    <h1 class="hero-title">Factor Analysis</h1>
    <p class="hero-desc">
        Reduce your survey variables into underlying factors using PCA with Varimax rotation.
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
    )
    st.markdown("---")
    st.markdown("### Steps")
    st.markdown("""
1. Upload your data  
2. Select columns  
3. Drop columns  
4. View Scree Plot  
5. Choose number of factors  
6. Name your factors  
7. Download scores
    """)
    st.markdown("---")
    st.markdown("<small style='color:#666'>Built for MKT 557 · Spring 2025</small>", unsafe_allow_html=True)

if uploaded_file is None:
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

numeric_cols = df_raw.select_dtypes(include=[np.number]).columns.tolist()

# ── STEP 1: Select columns ────────────────────────────────────────────────────
st.markdown("""
<div class="step-card">
    <div class="step-label">Step 1</div>
    <div class="step-title">Select Your Variables</div>
    <p class="step-desc">Choose the numeric columns you want to include in the factor analysis.</p>
</div>
""", unsafe_allow_html=True)

selected_cols = st.multiselect("Columns to include", options=numeric_cols, default=[])

if len(selected_cols) < 3:
    st.warning("⚠️ Please select at least 3 columns to run factor analysis.")
    st.stop()

df_work = df_raw[selected_cols].dropna()
st.caption(f"{len(selected_cols)} variables · {len(df_work)} complete responses")

# ── STEP 2: Drop columns ──────────────────────────────────────────────────────
st.markdown("""
<div class="step-card">
    <div class="step-label">Step 2</div>
    <div class="step-title">Drop Columns</div>
    <p class="step-desc">Remove any columns you do not want to include in the analysis.</p>
</div>
""", unsafe_allow_html=True)

drop_cols = st.multiselect("Columns to drop", options=selected_cols, default=[])
final_cols = [c for c in selected_cols if c not in drop_cols]

if len(final_cols) < 3:
    st.warning("⚠️ You need at least 3 columns after dropping. Please adjust your selection.")
    st.stop()

# Re-dropna after column changes, then replace any remaining inf values
df_final = df_raw[final_cols].dropna()
df_final = df_final.replace([np.inf, -np.inf], np.nan).dropna()

if len(df_final) < 10:
    st.error("Not enough complete rows to run factor analysis after cleaning. Check your data.")
    st.stop()

# ── STEP 3: Scree Plot ────────────────────────────────────────────────────────
st.markdown("""
<div class="step-card">
    <div class="step-label">Step 3</div>
    <div class="step-title">Scree Plot</div>
    <p class="step-desc">Eigenvalues for each principal component. The red dashed line marks eigenvalue = 1.</p>
</div>
""", unsafe_allow_html=True)

scaler = StandardScaler()
df_scaled = scaler.fit_transform(df_final)

pca = PCA()
pca.fit(df_scaled)
eigenvalues = pca.explained_variance_

fig_scree, ax = plt.subplots(figsize=(8, 4))
fig_scree.patch.set_facecolor('#f7f5f0')
ax.set_facecolor('#f7f5f0')
ax.bar(range(1, len(eigenvalues) + 1), eigenvalues, color='#1a1a2e', alpha=0.85, width=0.6, zorder=3)
ax.axhline(y=1, color='#e9c46a', linestyle='--', linewidth=2, label="Eigenvalue = 1", zorder=4)
ax.set_xlabel('Principal Component', fontsize=11, color='#333')
ax.set_ylabel('Eigenvalue', fontsize=11, color='#333')
ax.set_title('Scree Plot', fontsize=14, fontweight='bold', color='#1a1a2e', pad=12)
ax.legend(fontsize=9)
ax.grid(axis='y', alpha=0.3, zorder=0)
ax.spines[['top', 'right']].set_visible(False)
ax.spines[['left', 'bottom']].set_color('#ccc')
ax.set_xticks(range(1, len(eigenvalues) + 1))
ax.tick_params(colors='#555')
plt.tight_layout()
st.pyplot(fig_scree)

with st.expander("View eigenvalue table"):
    eigen_df = pd.DataFrame({
        "Component": range(1, len(eigenvalues) + 1),
        "Eigenvalue": eigenvalues.round(3),
        "Variance Explained (%)": (eigenvalues / len(final_cols) * 100).round(2),
        "Cumulative Variance (%)": np.cumsum(eigenvalues / len(final_cols) * 100).round(2),
    })
    st.dataframe(eigen_df.set_index("Component"), use_container_width=True)

# ── STEP 4: Choose number of factors ─────────────────────────────────────────
st.markdown("""
<div class="step-card">
    <div class="step-label">Step 4</div>
    <div class="step-title">Choose Number of Factors</div>
    <p class="step-desc">Based on the scree plot above, select how many factors to extract.</p>
</div>
""", unsafe_allow_html=True)

n_factors = st.number_input(
    "Number of factors",
    min_value=1,
    max_value=len(final_cols) - 1,
    value=2,
    step=1,
)

# ── STEP 5: Loadings Heatmap ──────────────────────────────────────────────────
st.markdown("""
<div class="step-card">
    <div class="step-label">Step 5</div>
    <div class="step-title">Factor Loadings (Varimax Rotation)</div>
    <p class="step-desc">How strongly each variable loads onto each factor.</p>
</div>
""", unsafe_allow_html=True)

try:
    fa = FactorAnalyzer(n_factors=n_factors, rotation='varimax')
    fa.fit(df_scaled)
except Exception as e:
    st.error(f"Factor analysis failed: {e}\n\nTry reducing the number of factors or checking your data for constant columns.")
    st.stop()
loadings = fa.loadings_
loadings_df = pd.DataFrame(
    loadings,
    index=final_cols,
    columns=[f"Factor {i+1}" for i in range(n_factors)]
)

fig_heat, ax2 = plt.subplots(figsize=(max(6, n_factors * 2.5), max(5, len(final_cols) * 0.55)))
fig_heat.patch.set_facecolor('white')
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
ax2.set_title("Varimax Factor Loadings", fontsize=13, fontweight='bold', color='#1a1a2e', pad=14)
ax2.set_xlabel("Factors", fontsize=10, color='#555')
ax2.set_ylabel("Variables", fontsize=10, color='#555')
ax2.tick_params(axis='x', rotation=0, labelsize=9)
ax2.tick_params(axis='y', rotation=0, labelsize=8)
plt.tight_layout()
st.pyplot(fig_heat)

with st.expander("View loadings table"):
    st.dataframe(loadings_df.style.format("{:.3f}"), use_container_width=True)

# ── STEP 6: Name factors ──────────────────────────────────────────────────────
st.markdown("""
<div class="step-card">
    <div class="step-label">Step 6</div>
    <div class="step-title">Name Your Factors</div>
    <p class="step-desc">Give each factor a name based on the variables that load onto it.</p>
</div>
""", unsafe_allow_html=True)

factor_names = []
cols_name = st.columns(n_factors)
for i, col in enumerate(cols_name):
    with col:
        name = st.text_input(f"Factor {i+1} name", value=f"Factor_{i+1}")
        factor_names.append(name)

# ── STEP 7: Scores & Download ─────────────────────────────────────────────────
st.markdown("""
<div class="step-card">
    <div class="step-label">Step 7</div>
    <div class="step-title">Factor Scores</div>
    <p class="step-desc">Each respondent's score on every factor.</p>
</div>
""", unsafe_allow_html=True)

factor_scores = fa.transform(df_scaled)
scores_df = pd.DataFrame(factor_scores, columns=factor_names)

id_cols = [c for c in ['uuid', 'record', 'id', 'ID', 'ResponseId'] if c in df_raw.columns]
if id_cols:
    id_series = df_raw.loc[df_work.index, id_cols[0]].reset_index(drop=True)
    scores_df.insert(0, id_cols[0], id_series.values)

st.dataframe(scores_df.head(10), use_container_width=True)
st.caption(f"Showing first 10 of {len(scores_df)} rows")

st.markdown("---")
st.markdown("### 📥 Download Results")

col_dl1, col_dl2, col_dl3 = st.columns(3)

with col_dl1:
    buf1 = io.BytesIO()
    scores_df.to_excel(buf1, index=False)
    st.download_button(
        label="⬇️ Factor Scores (.xlsx)",
        data=buf1.getvalue(),
        file_name="factor_scores.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )
    st.caption("Use as input for cluster analysis")

with col_dl2:
    buf2 = io.BytesIO()
    loadings_df.to_excel(buf2)
    st.download_button(
        label="⬇️ Factor Loadings (.xlsx)",
        data=buf2.getvalue(),
        file_name="factor_loadings.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )
    st.caption("Variable weights on each factor")

with col_dl3:
    try:
        merged = df_raw.copy()
        merged = merged.drop(columns=[c for c in final_cols if c in merged.columns], errors='ignore')
        scores_reset = scores_df.reset_index(drop=True)
        merged = pd.concat([merged.reset_index(drop=True), scores_reset], axis=1)
        buf3 = io.BytesIO()
        merged.to_excel(buf3, index=False)
        st.download_button(
            label="⬇️ Full Dataset (.xlsx)",
            data=buf3.getvalue(),
            file_name="full_data_with_factors.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        )
        st.caption("Original data with factor scores appended")
    except Exception as e:
        st.warning(f"Could not generate merged file: {e}")

st.markdown("---")
st.markdown(
    "<small style='color:#aaa'>Factor Analysis App · Goizueta Business School · "
    "Powered by scikit-learn & factor_analyzer</small>",
    unsafe_allow_html=True
)
