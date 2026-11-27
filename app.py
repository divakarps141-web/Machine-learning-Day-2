
import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="Iris AI | ML Dashboard",
    page_icon="🌸",
    layout="wide"
)

# ---------------- CUSTOM CSS ----------------
st.markdown("""
<style>
.stApp {
    background: linear-gradient(135deg, #0B1220, #111C32);
    color: #F8FAFC;
}
[data-testid="stSidebar"] {
    background: #111B2E;
    border-right: 1px solid #293B59;
}
h1, h2, h3, p, label {
    color: #F8FAFC !important;
}
.main-title {
    font-size: 38px;
    font-weight: 800;
    color: #A78BFA;
    margin-bottom: 5px;
}
.subtitle {
    color: #B8C5D9;
    font-size: 15px;
    margin-bottom: 25px;
}
.metric-card {
    background: #192943;
    border: 1px solid #344967;
    padding: 22px;
    border-radius: 15px;
    min-height: 115px;
}
.metric-label {
    color: #AFC1D9;
    font-size: 14px;
}
.metric-value {
    color: #FFFFFF;
    font-size: 29px;
    font-weight: 800;
    margin-top: 8px;
}
.section-card {
    background: #192943;
    padding: 20px;
    border-radius: 15px;
    border: 1px solid #344967;
}
div.stButton > button {
    background: linear-gradient(90deg, #7C3AED, #4F46E5);
    color: white;
    border: none;
    border-radius: 10px;
    font-weight: 700;
    padding: 10px 20px;
    width: 100%;
}
div.stButton > button:hover {
    background: #6D5CE7;
    color: white;
    border: none;
}
</style>
""", unsafe_allow_html=True)

# ---------------- LOAD DATA ----------------
@st.cache_resource
def load_model():
    iris = load_iris()

    df = pd.DataFrame(
        iris.data,
        columns=iris.feature_names
    )

    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(df)

    model = KMeans(
        n_clusters=3,
        random_state=42,
        n_init=10
    )

    clusters = model.fit_predict(scaled_data)
    score = silhouette_score(scaled_data, clusters)

    df["Cluster"] = clusters
    df["Species"] = [
        iris.target_names[i] for i in iris.target
    ]

    return df, scaler, model, score, iris.feature_names

df, scaler, model, score, features = load_model()

# ---------------- SIDEBAR ----------------
st.sidebar.markdown("## 🌸 Iris AI")
st.sidebar.caption("Machine Learning Dashboard")
st.sidebar.divider()

page = st.sidebar.radio(
    "NAVIGATION",
    ["Dashboard", "Explore Data", "Predict Cluster", "About Project"]
)

st.sidebar.divider()
st.sidebar.markdown("**Algorithm**")
st.sidebar.info("K-Means Clustering")
st.sidebar.markdown("**Dataset**")
st.sidebar.info("Iris Flower Dataset")

# ---------------- HEADER ----------------
st.markdown(
    '<div class="main-title">🌸 Iris Clustering Dashboard</div>',
    unsafe_allow_html=True
)
st.markdown(
    '<div class="subtitle">Explore flower measurements and discover patterns using unsupervised machine learning.</div>',
    unsafe_allow_html=True
)

# ---------------- DASHBOARD ----------------
if page == "Dashboard":

    c1, c2, c3, c4 = st.columns(4)

    metrics = [
        ("TOTAL FLOWERS", len(df)),
        ("FEATURES", len(features)),
        ("CLUSTERS", 3),
        ("SILHOUETTE SCORE", f"{score:.3f}")
    ]

    for col, (label, value) in zip([c1, c2, c3, c4], metrics):
        with col:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">{label}</div>
                <div class="metric-value">{value}</div>
            </div>
            """, unsafe_allow_html=True)

    st.write("")
    st.subheader("📊 Cluster Visualization")

    fig = px.scatter(
        df,
        x=features[2],
        y=features[3],
        color=df["Cluster"].astype(str),
        hover_data=["Species"],
        color_discrete_sequence=[
            "#A78BFA", "#22D3EE", "#FBBF24"
        ],
        labels={"color": "Cluster"},
        title="Petal Length vs Petal Width"
    )

    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="#192943",
        plot_bgcolor="#192943",
        font_color="#FFFFFF",
        margin=dict(l=20, r=20, t=55, b=20)
    )

    st.plotly_chart(fig, use_container_width=True)

    st.subheader("📋 Dataset Preview")
    st.dataframe(df.head(10), use_container_width=True)

# ---------------- EXPLORE DATA ----------------
elif page == "Explore Data":

    st.subheader("🔍 Explore Iris Dataset")

    st.write("Understand the measurements and cluster distribution.")

    col1, col2 = st.columns(2)

    with col1:
        selected_feature = st.selectbox(
            "Choose a feature",
            features
        )

        fig = px.histogram(
            df,
            x=selected_feature,
            color=df["Cluster"].astype(str),
            color_discrete_sequence=[
                "#A78BFA", "#22D3EE", "#FBBF24"
            ],
            title="Feature Distribution",
            template="plotly_dark"
        )
        fig.update_layout(
            paper_bgcolor="#192943",
            plot_bgcolor="#192943"
        )
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        counts = df["Cluster"].value_counts().sort_index()

        fig2 = px.pie(
            values=counts.values,
            names=[f"Cluster {i}" for i in counts.index],
            hole=0.55,
            title="Cluster Distribution",
            color_discrete_sequence=[
                "#A78BFA", "#22D3EE", "#FBBF24"
            ]
        )
        fig2.update_layout(
            template="plotly_dark",
            paper_bgcolor="#192943"
        )
        st.plotly_chart(fig2, use_container_width=True)

    st.subheader("Complete Dataset")
    st.dataframe(df, use_container_width=True)

# ---------------- PREDICTION ----------------
elif page == "Predict Cluster":

    st.subheader("🧪 Predict a Flower Cluster")
    st.write("Enter the four flower measurements below.")

    with st.form("prediction_form"):

        col1, col2 = st.columns(2)

        with col1:
            sepal_length = st.number_input(
                "Sepal Length (cm)", 0.0, 10.0, 5.8, 0.1
            )
            sepal_width = st.number_input(
                "Sepal Width (cm)", 0.0, 10.0, 2.7, 0.1
            )

        with col2:
            petal_length = st.number_input(
                "Petal Length (cm)", 0.0, 10.0, 4.1, 0.1
            )
            petal_width = st.number_input(
                "Petal Width (cm)", 0.0, 10.0, 1.0, 0.1
            )

        submitted = st.form_submit_button(
            "🔮 Predict Cluster"
        )

    if submitted:
        input_data = [[
            sepal_length,
            sepal_width,
            petal_length,
            petal_width
        ]]

        scaled_input = scaler.transform(input_data)
        prediction = model.predict(scaled_input)[0]

        st.success("Prediction completed successfully!")

        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">PREDICTED RESULT</div>
            <div class="metric-value">Cluster {prediction}</div>
        </div>
        """, unsafe_allow_html=True)

        st.caption(
            "The cluster number represents a group discovered by K-Means. "
            "It is not a guaranteed flower species prediction."
        )

# ---------------- ABOUT ----------------
elif page == "About Project":

    st.subheader("🤖 About This Project")

    st.markdown("""
    ### Iris Flower Clustering

    This project uses **K-Means Clustering**, an unsupervised
    machine learning algorithm, to group flowers based on
    their physical measurements.

    ### Machine Learning Workflow

    1. Load the Iris dataset.
    2. Standardize the four numerical features.
    3. Apply the K-Means clustering algorithm.
    4. Divide the flowers into three clusters.
    5. Evaluate clustering using the Silhouette Score.
    6. Predict the cluster for new flower measurements.

    ### Technologies Used

    - Python
    - Streamlit
    - Pandas
    - Scikit-learn
    - Plotly
    """)

    st.info(
        "This is an educational unsupervised learning project."
    )

st.sidebar.markdown("---")
st.sidebar.caption("Built with Python + Streamlit")
