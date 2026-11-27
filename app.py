
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

st.set_page_config(page_title="Iris Clustering", page_icon="🌸")

st.title("🌸 Iris Flower Clustering Prediction")
st.write("K-Means Machine Learning Project")

# Load dataset
iris = pd.read_csv("iris.csv")

features = [
    "sepal_length",
    "sepal_width",
    "petal_length",
    "petal_width"
]

iris_features = iris[features].copy()
iris_features.drop_duplicates(inplace=True)

# Scaling
scaler = StandardScaler()
iris_scaled = scaler.fit_transform(iris_features)

# K-Means
model = KMeans(
    n_clusters=3,
    random_state=42,
    n_init=10
)

clusters = model.fit_predict(iris_scaled)

# Evaluation
score = silhouette_score(iris_scaled, clusters)

st.metric("Silhouette Score", round(score, 3))

# Prediction
st.header("Predict New Flower Cluster")

col1, col2 = st.columns(2)

with col1:
    sl = st.number_input("Sepal Length", value=5.8)
    sw = st.number_input("Sepal Width", value=2.7)

with col2:
    pl = st.number_input("Petal Length", value=4.1)
    pw = st.number_input("Petal Width", value=1.0)

if st.button("Predict Cluster"):

    new_flower = pd.DataFrame({
        "sepal_length": [sl],
        "sepal_width": [sw],
        "petal_length": [pl],
        "petal_width": [pw]
    })

    new_scaled = scaler.transform(new_flower)

    prediction = model.predict(new_scaled)[0]

    st.success(f"Predicted Cluster: {prediction}")

# Visualization
st.header("Cluster Visualization")

clustered = iris_features.copy()
clustered["cluster"] = clusters

fig, ax = plt.subplots()

ax.scatter(
    clustered["petal_length"],
    clustered["petal_width"],
    c=clustered["cluster"]
)

ax.set_xlabel("Petal Length")
ax.set_ylabel("Petal Width")
ax.set_title("K-Means Iris Clustering")

st.pyplot(fig)

st.subheader("Dataset Preview")
st.dataframe(clustered.head(10))
