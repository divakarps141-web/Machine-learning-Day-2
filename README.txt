CLUSTERIQ — K-MEANS CLUSTER IDENTIFICATION

FILES
- app.py: Streamlit dashboard and Python prediction backend
- requirements.txt: deployment dependencies
- frontend/style.css and frontend/interaction.js: optional frontend assets

REQUIRED MODEL FILES
Place your own trusted files in the repository root:
- kmeans_model.pkl
- scaler.pkl

The app expects a scaler fitted on exactly four features in this order:
sepal_length, sepal_width, petal_length, petal_width
If your scaler was fitted with different column names, the app uses
scaler.feature_names_in_ when available.

DEPLOY
1. Upload app.py, requirements.txt, and your two .pkl files to the
   root of your GitHub repository.
2. In Streamlit Community Cloud, set the main file path to app.py.
3. Deploy or reboot the app.

IMPORTANT
Only load pickle files that you created yourself or trust.
The scikit-learn version in the cloud should match the version used
when you saved the model and scaler. If you know that version,
pin it in requirements.txt, for example:
scikit-learn==YOUR_VERSION
