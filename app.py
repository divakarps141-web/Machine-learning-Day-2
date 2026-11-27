
import streamlit as st
import pickle
import pandas as pd
import plotly.express as px

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="K-Means Cluster Dashboard",
    page_icon="🧠",
    layout="wide"
)

# ---------------- CUSTOM CSS ----------------
st.markdown("""
<style>
.stApp {
    background: #0B1120;
    color: #FFFFFF;
}

[data-testid="stSidebar"] {
    display: none;
}

.block-container {
    padding: 2rem 3rem;
    max-width: 1400px;
}

h1, h2, h3, p, label {
    color: #FFFFFF !important;
}

.header {
    background: linear-gradient(110deg, #1E1B4B, #172554);
    padding: 28px;
    border-radius: 18px;
    border: 1px solid #3730A3;
    margin-bottom: 25px;
}

.header h1 {
    font-size: 32px;
    margin: 0;
    color: #C4B5FD !important;
}

.header p {
    color: #CBD5E1 !important;
    margin-top: 8px;
}

.metric {
    background: #17243A;
    border: 1px solid #334155;
    border-radius: 15px;
    padding: 20px;
    text-align: center;
}

.metric-label {
    color: #94A3B8;
    font-size: 14px;
}

.metric-value {
    color: #67E8F9;
    font-size: 28px;
    font-weight: bold;
    margin-top: 8px;
}

.panel {
    background: #17243A;
    border: 1px solid #334155;
    border-radius: 16px;
    padding: 24px;
}

div.stButton > button,
div.stFormSubmitButton > button {
    background: linear-gradient(90deg, #7C3AED, #4F46E5);
    color: white;
    border: none;
    border-radius: 10px;
    height: 48px;
    font-weight: bold;
    width: 100%;
}

div.stButton > button:hover,
div.stFormSubmitButton > button:hover {
    background: #6D28D9;
    color: white;
    border: none;
}

div[data-testid="stNumberInput"] input {
    background: #0F1B30;
    color: white;
    border: 1px solid #475569;
    border-radius: 8px;
}

.result {
    background: linear-gradient(120deg, #312E81, #164E63);
    border: 1px solid #818CF8;
    border-radius: 16px;
    padding: 25px;
    text-align: center;
    margin-top: 20px;
}

.result-label {
    color: #C4B5FD;
    font-size: 15px;
}

.result-value {
    color: #67E8F9;
    font-size: 36px;
    font-weight: 800;
    margin-top: 8px;
}

</style>
""", unsafe_allow_html=True)


# ---------------- LOAD MODEL ----------------
@st.cache_resource
def load_models():

    with open("kmeans_model.pkl", "rb") as file:
        model = pickle.load(file)

    with open("scaler.pkl", "rb") as file:
        scaler = pickle.load(file)

    return model, scaler


try:
    model, scaler = load_models()

except FileNotFoundError:
    st.error(
        "Model files not found. Upload kmeans_model.pkl "
        "and scaler.pkl to your GitHub repository."
    )
    st.stop()

except Exception as error:
    st.error(f"Unable to load model: {error}")
    st.stop()


# ---------------- HEADER ----------------
st.markdown("""
<div class="header">
    <h1>🧠 K-Means Cluster Identification</h1>
    <p>Machine Learning Dashboard | Unsupervised Learning</p>
</div>
""", unsafe_allow_html=True)


# ---------------- METRICS ----------------
c1, c2, c3 = st.columns(3)

with c1:
    st.markdown("""
    <div class="metric">
        <div class="metric-label">ALGORITHM</div>
        <div class="metric-value">K-Means</div>
    </div>
    """, unsafe_allow_html=True)

with c2:
    st.markdown("""
    <div class="metric">
        <div class="metric-label">INPUT FEATURES</div>
        <div class="metric-value">04</div>
    </div>
    """, unsafe_allow_html=True)

with c3:
    st.markdown("""
    <div class="metric">
        <div class="metric-label">MODEL STATUS</div>
        <div class="metric-value">ACTIVE</div>
    </div>
    """, unsafe_allow_html=True)


st.write("")
left, right = st.columns([1, 1], gap="large")


# ---------------- INPUT SECTION ----------------
with left:

    st.markdown('<div class="panel">', unsafe_allow_html=True)

    st.subheader("🌸 Flower Measurements")
    st.caption("Enter the four measurements to identify its cluster.")

    with st.form("cluster_form"):

        col1, col2 = st.columns(2)

        with col1:
            sepal_length = st.number_input(
                "Sepal Length (cm)",
                min_value=0.0,
                max_value=10.0,
                value=5.8,
                step=0.1
            )

            sepal_width = st.number_input(
                "Sepal Width (cm)",
                min_value=0.0,
                max_value=10.0,
                value=2.7,
                step=0.1
            )

        with col2:
            petal_length = st.number_input(
                "Petal Length (cm)",
                min_value=0.0,
                max_value=10.0,
                value=4.1,
                step=0.1
            )

            petal_width = st.number_input(
                "Petal Width (cm)",
                min_value=0.0,
                max_value=10.0,
                value=1.0,
                step=0.1
            )

        submitted = st.form_submit_button(
            "🔍 IDENTIFY CLUSTER",
            use_container_width=True
        )

    st.markdown('</div>', unsafe_allow_html=True)


# ---------------- RESULT SECTION ----------------
with right:

    st.markdown('<div class="panel">', unsafe_allow_html=True)

    st.subheader("📊 Cluster Identification")

    if submitted:

        input_data = pd.DataFrame(
            [[
                sepal_length,
                sepal_width,
                petal_length,
                petal_width
            ]],
            columns=[
                "sepal_length",
                "sepal_width",
                "petal_length",
                "petal_width"
            ]
        )

        try:
            scaled_data = scaler.transform(input_data)
            prediction = model.predict(scaled_data)[0]

            st.session_state["prediction"] = int(prediction)
            st.session_state["input_values"] = input_data.iloc[0].to_dict()

        except Exception as error:
            st.error(f"Prediction failed: {error}")

    if "prediction" in st.session_state:

        prediction = st.session_state["prediction"]

        st.markdown(f"""
        <div class="result">
            <div class="result-label">PREDICTED CLUSTER</div>
            <div class="result-value">Cluster {prediction}</div>
        </div>
        """, unsafe_allow_html=True)

        st.success("Cluster identified successfully!")

        st.write("**Submitted measurements**")
        st.dataframe(
            pd.DataFrame(
                [st.session_state["input_values"]]
            ),
            use_container_width=True,
            hide_index=True
        )

    else:
        st.info(
            "Enter flower measurements and click "
            "'IDENTIFY CLUSTER' to view the result."
        )

    st.markdown('</div>', unsafe_allow_html=True)


# ---------------- FOOTER ----------------
st.write("")
st.divider()

st.markdown(
    "<p style='text-align:center;color:#94A3B8 !important;'>"
    "Powered by Python • Scikit-learn • Streamlit"
    "</p>",
    unsafe_allow_html=True
)
