import os, json, joblib, pandas as pd
from sklearn.preprocessing import OrdinalEncoder

FEATURE_COLUMNS = [
    "Budget_Lakh", "Mileage_kmpl", "Seating_Capacity",
    "Car_Type", "Fuel_Type", "Transmission", "Condition", "User_Needs", "Car_Color"
]

NUMERICAL_COLUMNS = ["Budget_Lakh", "Mileage_kmpl", "Seating_Capacity"]
CATEGORICAL_COLUMNS = ["Car_Type", "Fuel_Type", "Transmission", "Condition", "User_Needs", "Car_Color"]

ALL_FEATURE_CATEGORIES = {
    "Car_Type": ["SUV", "Hatchback", "Sedan"],
    "Fuel_Type": ["Petrol", "Diesel", "Electric"],
    "Transmission": ["Manual", "Automatic"],
    "Condition": ["New", "Used"],
    "User_Needs": ["Family", "Sporty", "Luxury"],
    "Car_Color": ["Red", "Blue", "Black", "White", "Silver"]
}

def load_dataset(path):
    df = pd.read_csv(path, na_values=['N/A','NA',''])
    df = df.loc[:, ~df.columns.duplicated()]
    return df

def _keep_allowed_columns(df):
    present = [c for c in FEATURE_COLUMNS if c in df.columns]
    return df[present].copy()

def preprocess_fit_save(df, model_dir):
    os.makedirs(model_dir, exist_ok=True)
    df_original = df.copy().reset_index(drop=True)
    df_kept = _keep_allowed_columns(df_original)
    df_proc = df_kept.copy()
    imputation_values = {}

    # Numeric
    for col in NUMERICAL_COLUMNS:
        if col in df_proc.columns:
            df_proc[col] = pd.to_numeric(df_proc[col], errors='coerce')
            mean_val = df_proc[col].mean()
            imputation_values[col] = float(mean_val) if pd.notna(mean_val) else 0.0
            df_proc[col].fillna(imputation_values[col], inplace=True)
        else:
            imputation_values[col] = 0.0
            df_proc[col] = imputation_values[col]

    # Categorical
    for col in CATEGORICAL_COLUMNS:
        if col not in ALL_FEATURE_CATEGORIES:
            ALL_FEATURE_CATEGORIES[col] = []
        if 'missing' not in ALL_FEATURE_CATEGORIES[col]:
            ALL_FEATURE_CATEGORIES[col].append('missing')
        if col in df_proc.columns:
            df_proc[col] = df_proc[col].fillna('missing').astype(str)
        else:
            df_proc[col] = 'missing'
        imputation_values[col] = 'missing'

    joblib.dump(imputation_values, os.path.join(model_dir, "imputer.pkl"))

    # Encode categorical
    for col in CATEGORICAL_COLUMNS:
        df_proc[col] = pd.Categorical(df_proc[col], categories=ALL_FEATURE_CATEGORIES[col])
    encoder = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
    df_proc[CATEGORICAL_COLUMNS] = encoder.fit_transform(df_proc[CATEGORICAL_COLUMNS])
    joblib.dump(encoder, os.path.join(model_dir, "encoder.pkl"))

    df_proc = df_proc[FEATURE_COLUMNS].copy()
    with open(os.path.join(model_dir, "columns.json"), "w") as f:
        json.dump(list(df_proc.columns), f)

    df_proc.to_csv(os.path.join(model_dir, "df_proc.csv"), index=False)
    df_original.to_csv(os.path.join(model_dir, "df_original.csv"), index=False)

    return df_proc

def preprocess_transform(df, model_dir):
    df_proc = df.copy().reset_index(drop=True)
    imputer_path = os.path.join(model_dir, "imputer.pkl")
    encoder_path = os.path.join(model_dir, "encoder.pkl")
    columns_path = os.path.join(model_dir, "columns.json")
    if not os.path.exists(imputer_path) or not os.path.exists(encoder_path) or not os.path.exists(columns_path):
        raise FileNotFoundError("Preprocessing artifacts missing. Train/upload dataset first.")
    imputation_values = joblib.load(imputer_path)
    encoder = joblib.load(encoder_path)
    with open(columns_path, "r") as f:
        cols = json.load(f)

    # Numeric
    for col in NUMERICAL_COLUMNS:
        df_proc[col] = pd.to_numeric(df_proc.get(col, 0), errors='coerce').fillna(imputation_values.get(col,0))

    # Categorical
    for col in CATEGORICAL_COLUMNS:
        df_proc[col] = df_proc.get(col, 'missing').fillna(imputation_values.get(col,'missing')).astype(str)
        if 'missing' not in ALL_FEATURE_CATEGORIES.get(col, []):
            ALL_FEATURE_CATEGORIES[col].append('missing')
        df_proc[col] = pd.Categorical(df_proc[col], categories=ALL_FEATURE_CATEGORIES[col])
    df_proc[CATEGORICAL_COLUMNS] = encoder.transform(df_proc[CATEGORICAL_COLUMNS])

    df_proc = df_proc.reindex(columns=cols, fill_value=0)
    return df_proc[FEATURE_COLUMNS].copy()
