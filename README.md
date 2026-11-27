# Iris Clustering App

A machine learning web application that uses **K-Means clustering** to group Iris flowers based on their measurements. The app is built with Python and Streamlit and provides an interactive interface for entering flower measurements and viewing the predicted cluster.

## 🚀 Live Demo

[Open the Iris Clustering App](https://machine-learning-day-2.streamlit.app/)

## ✨ Features

- Interactive web interface built with Streamlit
- K-Means clustering with 3 clusters
- Uses all four Iris flower measurements:
  - Sepal length
  - Sepal width
  - Petal length
  - Petal width
- Standardizes input data using `StandardScaler`
- Loads the trained model and scaler from saved pickle files
- Displays the predicted cluster for the entered measurements

## 🧰 Tech Stack

- Python
- Streamlit
- Pandas
- Scikit-learn
- K-Means clustering
- StandardScaler

## 📁 Project Structure

```text
machine-learning-day-2/
├── app.py
├── requirements.txt
├── kmeans_model.pkl
├── scaler.pkl
└── README.md
```

> Make sure `kmeans_model.pkl` and `scaler.pkl` are included in the repository if the app loads them from these files.

## ⚙️ Run Locally

1. Clone this repository:

   ```bash
   git clone https://github.com/YOUR-USERNAME/machine-learning-day-2.git
   ```

2. Move into the project folder:

   ```bash
   cd machine-learning-day-2
   ```

3. Install the dependencies:

   ```bash
   pip install -r requirements.txt
   ```

4. Start the Streamlit app:

   ```bash
   streamlit run app.py
   ```

5. Open the local URL shown in your terminal (usually `http://localhost:8501`).

## 📦 Requirements

Your `requirements.txt` should contain one package per line:

```text
streamlit==1.67.0
pandas==2.2.3
scikit-learn==1.6.1
```

## 🧠 Machine Learning Workflow

1. Load and inspect the Iris dataset.
2. Select the four numerical flower measurements.
3. Remove duplicate records and check for missing values.
4. Standardize the features using `StandardScaler`.
5. Train a K-Means model with `n_clusters=3`, `random_state=42`, and `n_init=10`.
6. Evaluate the clustering result using the silhouette score.
7. Save the trained model and scaler as pickle files.
8. Load those files in the Streamlit app to predict the cluster for new measurements.

## 📝 Example Input

| Feature | Example value |
|---|---:|
| Sepal length | 5.8 |
| Sepal width | 2.7 |
| Petal length | 4.1 |
| Petal width | 1.0 |

The app uses these measurements to predict a cluster.

## ⚠️ Note

K-Means is an **unsupervised learning** algorithm. The cluster numbers (0, 1, and 2) are group identifiers; they are not guaranteed to correspond directly to Iris species labels.

Pickle files should only be loaded from trusted sources. For reliable deployment, use compatible versions of Python and scikit-learn when creating and loading the model files.

## 👨‍💻 Author

**Divakar P**  
First-year B.E. Artificial Intelligence and Machine Learning student

---

If you find this project useful, feel free to ⭐ the repository!
