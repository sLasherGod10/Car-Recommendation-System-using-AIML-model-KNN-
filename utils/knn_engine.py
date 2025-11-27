import os
import joblib
import pandas as pd
from sklearn.neighbors import NearestNeighbors

# Files saved per-dataset
KNN_FILENAME = "knn.pkl"
DF_PROC_FILENAME = "df_proc.csv"
DF_ORIG_FILENAME = "df_original.csv"

def train_knn(df_proc, df_original, model_dir, n_neighbors=5):
    """
    df_proc     : processed features DataFrame (only FEATURE_COLUMNS)
    df_original : original dataframe (for display)
    model_dir   : directory to save model/files
    """
    os.makedirs(model_dir, exist_ok=True)

    df_proc_path = os.path.join(model_dir, DF_PROC_FILENAME)
    df_original_path = os.path.join(model_dir, DF_ORIG_FILENAME)
    knn_path = os.path.join(model_dir, KNN_FILENAME)

    # Save CSVs
    df_proc.to_csv(df_proc_path, index=False)
    df_original.to_csv(df_original_path, index=False)

    # Ensure n_neighbors is valid
    if len(df_proc) <= n_neighbors:
        n_neighbors = max(1, len(df_proc) - 1)

    knn = NearestNeighbors(n_neighbors=n_neighbors, metric="euclidean")
    knn.fit(df_proc.values)   # fit on numeric matrix

    joblib.dump(knn, knn_path)
    return "KNN model trained successfully!"

def load_knn(model_dir):
    df_proc_path = os.path.join(model_dir, DF_PROC_FILENAME)
    df_original_path = os.path.join(model_dir, DF_ORIG_FILENAME)
    knn_path = os.path.join(model_dir, KNN_FILENAME)

    if not os.path.exists(knn_path):
        return None, None, None

    knn = joblib.load(knn_path)
    df_proc = pd.read_csv(df_proc_path)
    df_original = pd.read_csv(df_original_path)
    return knn, df_proc, df_original

def knn_recommend(knn, user_vector):
    """
    knn: trained NearestNeighbors
    user_vector: DataFrame or 1D array / Series representing one sample
    Returns: distances, indices
    """
    import numpy as np
    # Convert input to correct 2D numpy array
    if isinstance(user_vector, pd.DataFrame):
        arr = user_vector.values
    elif isinstance(user_vector, pd.Series):
        arr = user_vector.values.reshape(1, -1)
    else:
        arr = np.asarray(user_vector)
        if arr.ndim == 1:
            arr = arr.reshape(1, -1)

    distances, indices = knn.kneighbors(arr)
    return distances, indices

def restart_knn_model():
    # kept simple — per-dataset restart is handled by deleting model_dir from app.py if needed
    return "KNN reset is now managed per-dataset from app.py."
