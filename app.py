import pickle
from pathlib import Path

import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="ClusterIQ | K-Means",
    page_icon="✦",
    layout="wide",
    initial_sidebar_state="collapsed",
)

MODEL_PATH = Path("kmeans_model.pkl")
SCALER_PATH = Path("scaler.pkl")

# ---------------- PREMIUM LIGHT THEME ----------------
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700;800&family=Manrope:wght@500;600;700;800&display=swap');

    :root {
        --ink: #172033;
        --muted: #68758b;
        --purple: #6941e8;
        --purple-dark: #5130c5;
        --line: #e8eaf2;
        --surface: #ffffff;
        --canvas: #f5f6fb;
    }
    html, body, [class*="css"] {
        font-family: 'DM Sans', sans-serif;
    }
    .stApp {
        background:
          radial-gradient(circle at 90% 3%, rgba(121, 93, 239, .10), transparent 25%),
          radial-gradient(circle at 5% 35%, rgba(75, 163, 255, .07), transparent 25%),
          var(--canvas);
        color: var(--ink);
    }
    [data-testid="stHeader"] { background: rgba(245,246,251,.88); }
    [data-testid="stToolbar"] { right: 1rem; }
    .block-container { max-width: 1240px; padding: 2rem 2rem 3rem; }
    #MainMenu, footer { visibility: hidden; }
    h1, h2, h3, p, label, span { color: var(--ink); }
    .brand-row { display:flex; align-items:center; gap:12px; margin-bottom:18px; }
    .brand-mark {
        width:42px; height:42px; border-radius:14px;
        display:flex; align-items:center; justify-content:center;
        color:#fff; font-size:21px; font-weight:800;
        background:linear-gradient(135deg,#805cf4,#4d35c9);
        box-shadow:0 8px 22px rgba(105,65,232,.25);
    }
    .brand-name { font:800 18px 'Manrope',sans-serif; letter-spacing:-.4px; color:#182238; }
    .brand-caption { font-size:11px; color:#8290a5; letter-spacing:1.5px; text-transform:uppercase; }
    .hero {
        position:relative; overflow:hidden; padding:30px 34px; border-radius:24px;
        background:linear-gradient(115deg,#ffffff 0%,#f0edff 58%,#edf5ff 100%);
        border:1px solid #e4defd; box-shadow:0 14px 40px rgba(38,35,75,.06);
        margin-bottom:22px;
    }
    .hero:after {
        content:""; position:absolute; width:240px; height:240px; right:-65px; top:-105px;
        border-radius:50%; border:1px solid rgba(105,65,232,.15);
        box-shadow:0 0 0 24px rgba(105,65,232,.035),0 0 0 50px rgba(105,65,232,.025);
    }
    .eyebrow { color:#6941e8; font-size:11px; font-weight:800; letter-spacing:1.8px; text-transform:uppercase; }
    .hero h1 { font:800 clamp(27px,4vw,39px) 'Manrope',sans-serif; letter-spacing:-1.5px; margin:8px 0 7px; color:#182238; }
    .hero p { color:#68758b; font-size:14px; max-width:620px; margin:0; line-height:1.65; }
    .hero-pill {
        display:inline-flex; align-items:center; gap:7px; margin-top:17px; padding:7px 11px;
        border:1px solid #ddd5ff; border-radius:30px; background:#fff;
        color:#5b3bd1; font-size:11px; font-weight:700;
    }
    .dot { width:7px; height:7px; border-radius:50%; background:#20b486; box-shadow:0 0 0 4px #dff8ef; }
    .metric-card {
        background:#fff; border:1px solid var(--line); border-radius:17px;
        padding:17px 19px; box-shadow:0 5px 18px rgba(31,42,68,.035); min-height:105px;
    }
    .metric-label { color:#8490a3; font-size:11px; font-weight:700; letter-spacing:1.1px; text-transform:uppercase; }
    .metric-value { color:#202b42; font:800 25px 'Manrope',sans-serif; margin-top:7px; }
    .metric-note { color:#8a95a8; font-size:11px; margin-top:2px; }
    .section-heading { font:800 18px 'Manrope',sans-serif; color:#202b42; margin:5px 0 4px; }
    .section-sub { color:#8490a3; font-size:12px; margin-bottom:18px; }
    .panel {
        background:#fff; border:1px solid var(--line); border-radius:20px;
        padding:23px; box-shadow:0 7px 24px rgba(31,42,68,.04); height:100%;
    }
    .panel-title { font:800 16px 'Manrope',sans-serif; color:#202b42; margin-bottom:4px; }
    .panel-desc { color:#8490a3; font-size:12px; margin-bottom:18px; }
    [data-testid="stNumberInput"] label { color:#46536a !important; font-size:12px !important; font-weight:700 !important; }
    [data-testid="stNumberInput"] input {
        border:1px solid #e2e5ee !important; border-radius:10px !important;
        background:#fbfcff !important; color:#202b42 !important; min-height:43px;
    }
    [data-testid="stNumberInput"] input:focus { border-color:#9278f2 !important; box-shadow:0 0 0 3px #eee9ff !important; }
    div.stButton > button, div[data-testid="stFormSubmitButton"] button {
        border:0 !important; border-radius:11px !important; min-height:47px;
        background:linear-gradient(100deg,#7955ed,#5836d2) !important;
        color:white !important; font-weight:800 !important; letter-spacing:.2px;
        box-shadow:0 8px 18px rgba(105,65,232,.20);
        transition:transform .18s ease, box-shadow .18s ease;
    }
    div.stButton > button:hover, div[data-testid="stFormSubmitButton"] button:hover {
        transform:translateY(-1px); box-shadow:0 11px 23px rgba(105,65,232,.27);
    }
    .result-empty {
        border:1px dashed #d9ddec; background:#fafbff; border-radius:16px;
        padding:34px 20px; text-align:center; color:#8792a6;
    }
    .result-card {
        background:linear-gradient(135deg,#f4f0ff,#eef5ff);
        border:1px solid #e2dafe; border-radius:17px; padding:22px;
        text-align:center; margin:12px 0 17px;
    }
    .result-kicker { color:#7958dc; font-size:11px; font-weight:800; letter-spacing:1.5px; }
    .result-number { color:#5130c5; font:800 39px 'Manrope',sans-serif; margin:5px 0; }
    .result-caption { color:#77839a; font-size:12px; }
    .feature-row {
        display:flex; justify-content:space-between; gap:12px; padding:10px 0;
        border-bottom:1px solid #f0f1f6; font-size:12px;
    }
    .feature-row:last-child { border-bottom:0; }
    .feature-name { color:#7a879b; }
    .feature-value { color:#27334a; font-weight:700; }
    .footer { text-align:center; color:#9aa4b5; font-size:11px; padding-top:24px; }
    div[data-testid="stAlert"] { border-radius:12px; }
    @media (max-width: 700px) {
        .block-container { padding:1rem .8rem 2rem; }
        .hero { padding:23px 20px; }
        .panel { padding:17px; }
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Small decorative canvas animation. It is visual only; prediction remains in Python.
components.html(
    """
    <div style="height:1px;overflow:hidden">
      <canvas id="ambient"></canvas>
    </div>
    <script>
      // Lightweight decorative script; no data is collected or sent anywhere.
      const canvas = document.getElementById("ambient");
      if (canvas) {
        const ctx = canvas.getContext("2d");
        let t = 0;
        function draw() {
          t += 0.01;
          ctx.clearRect(0, 0, canvas.width, canvas.height);
          requestAnimationFrame(draw);
        }
        canvas.width = 1; canvas.height = 1; draw();
      }
    </script>
    """,
    height=1,
)

# ---------------- MODEL LOADING ----------------
@st.cache_resource
def load_models():
    if not MODEL_PATH.exists():
        raise FileNotFoundError("kmeans_model.pkl is missing from the project root.")
    if not SCALER_PATH.exists():
        raise FileNotFoundError("scaler.pkl is missing from the project root.")

    # Only load pickle files that you created yourself or trust.
    with MODEL_PATH.open("rb") as f:
        model = pickle.load(f)
    with SCALER_PATH.open("rb") as f:
        scaler = pickle.load(f)
    return model, scaler

try:
    model, scaler = load_models()
except FileNotFoundError as exc:
    st.error(str(exc))
    st.info("Upload both trusted .pkl files beside app.py in your GitHub repository.")
    st.stop()
except Exception as exc:
    st.error("The model files could not be loaded. Check that the deployment scikit-learn version matches the version used to save them.")
    st.caption(f"Technical detail: {exc}")
    st.stop()

# Use original training feature names where available.
default_features = ["sepal_length", "sepal_width", "petal_length", "petal_width"]
feature_names = list(getattr(scaler, "feature_names_in_", default_features))
if len(feature_names) != 4:
    st.error(f"Expected a scaler trained on 4 features, but found {len(feature_names)}.")
    st.stop()

# ---------------- BRAND + HERO ----------------
st.markdown(
    """
    <div class="brand-row">
      <div class="brand-mark">✦</div>
      <div>
        <div class="brand-name">ClusterIQ</div>
        <div class="brand-caption">Machine learning studio</div>
      </div>
    </div>
    <div class="hero">
      <div class="eyebrow">UNSUPERVISED MACHINE LEARNING</div>
      <h1>Cluster Identification</h1>
      <p>Discover the group behind your flower measurements. Enter four features and let your trained K-Means model identify the closest cluster.</p>
      <div class="hero-pill"><span class="dot"></span> Model ready for prediction</div>
    </div>
    """,
    unsafe_allow_html=True,
)

# ---------------- METRICS ----------------
n_clusters = getattr(model, "n_clusters", "—")
m1, m2, m3 = st.columns(3)
with m1:
    st.markdown(f'<div class="metric-card"><div class="metric-label">Algorithm</div><div class="metric-value">K-Means</div><div class="metric-note">Unsupervised clustering</div></div>', unsafe_allow_html=True)
with m2:
    st.markdown(f'<div class="metric-card"><div class="metric-label">Input features</div><div class="metric-value">04</div><div class="metric-note">Measurements per sample</div></div>', unsafe_allow_html=True)
with m3:
    st.markdown(f'<div class="metric-card"><div class="metric-label">Configured clusters</div><div class="metric-value">{n_clusters}</div><div class="metric-note">Groups learned by the model</div></div>', unsafe_allow_html=True)

st.write("")
left, right = st.columns([1.03, 0.97], gap="large")

# ---------------- INPUT PANEL ----------------
with left:
    st.markdown('<div class="panel-title">Enter flower measurements</div><div class="panel-desc">All measurements are in centimetres.</div>', unsafe_allow_html=True)
    with st.form("cluster_prediction_form"):
        c1, c2 = st.columns(2)
        with c1:
            values = []
            values.append(st.number_input("Sepal length (cm)", min_value=0.0, max_value=10.0, value=5.8, step=0.1))
            values.append(st.number_input("Sepal width (cm)", min_value=0.0, max_value=10.0, value=2.7, step=0.1))
        with c2:
            values.append(st.number_input("Petal length (cm)", min_value=0.0, max_value=10.0, value=4.1, step=0.1))
            values.append(st.number_input("Petal width (cm)", min_value=0.0, max_value=10.0, value=1.0, step=0.1))

        submitted = st.form_submit_button("✦  IDENTIFY CLUSTER", use_container_width=True)

    st.caption("Your measurements are transformed using the same fitted StandardScaler used during model training.")

# ---------------- PREDICTION ----------------
if submitted:
    try:
        sample = pd.DataFrame([values], columns=feature_names)
        transformed = scaler.transform(sample)
        predicted_cluster = int(model.predict(transformed)[0])
        st.session_state["cluster_prediction"] = predicted_cluster
        st.session_state["cluster_sample"] = dict(zip(feature_names, values))
    except Exception as exc:
        st.error("Prediction failed. Confirm that the four input features and their order match the model training data.")
        st.caption(f"Technical detail: {exc}")

with right:
    st.markdown('<div class="panel-title">Prediction result</div><div class="panel-desc">Your model output will appear here.</div>', unsafe_allow_html=True)

    if "cluster_prediction" in st.session_state:
        cluster = st.session_state["cluster_prediction"]
        st.markdown(
            f"""
            <div class="result-card">
              <div class="result-kicker">IDENTIFIED CLUSTER</div>
              <div class="result-number">Cluster {cluster}</div>
              <div class="result-caption">K-Means prediction completed successfully</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.markdown("**Submitted measurements**")
        sample = st.session_state["cluster_sample"]
        for name, value in sample.items():
            pretty_name = name.replace("_", " ").title()
            st.markdown(
                f'<div class="feature-row"><span class="feature-name">{pretty_name}</span><span class="feature-value">{value:.1f} cm</span></div>',
                unsafe_allow_html=True,
            )
        st.write("")
        if st.button("↻  Clear result", use_container_width=True):
            st.session_state.pop("cluster_prediction", None)
            st.session_state.pop("cluster_sample", None)
            st.rerun()
    else:
        st.markdown(
            """
            <div class="result-empty">
              <div style="font-size:32px;margin-bottom:10px">✧</div>
              <div style="font-weight:700;color:#526078;margin-bottom:5px">Awaiting measurements</div>
              <div style="font-size:12px;line-height:1.6">Complete the form and select<br>Identify Cluster to see your result.</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

st.markdown(
    '<div class="footer">ClusterIQ · Powered by Python, Streamlit and Scikit-learn</div>',
    unsafe_allow_html=True,
)
