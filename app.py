from flask import Flask, render_template, request, flash, redirect, url_for
import os, json, io, base64
import pandas as pd
import matplotlib.pyplot as plt
from utils.preprocess import preprocess_fit_save, preprocess_transform, load_dataset, FEATURE_COLUMNS, NUMERICAL_COLUMNS, CATEGORICAL_COLUMNS
from utils.knn_engine import train_knn, load_knn, knn_recommend

app = Flask(__name__, static_folder="static", template_folder="templates")
app.secret_key = os.getenv("SECRET_KEY", "IGNITRON_SECRET_KEY")

DATASETS_DIR = "datasets"
MODELS_DIR = "models"
DATASETS_FILE = "datasets.json"
ALLOWED_UPLOAD_USERS = ["NIPUNNAIK", "OMKARKASAR", "ANISHGOSAVI", "ATHARVAKHAIRE"]
TOP_N = 5

os.makedirs(DATASETS_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)

def load_datasets():
    if os.path.exists(DATASETS_FILE):
        with open(DATASETS_FILE, "r") as f:
            return json.load(f)
    return {}

def save_datasets(datasets):
    with open(DATASETS_FILE, "w") as f:
        json.dump(datasets, f, indent=4)

def save_query_local(model_dir, query_record):
    qpath = os.path.join(model_dir, "queries.json")
    lst = []
    if os.path.exists(qpath):
        try:
            with open(qpath, "r") as f:
                lst = json.load(f)
        except:
            lst = []
    lst.append(query_record)
    with open(qpath, "w") as f:
        json.dump(lst, f, indent=2)

def initialize_default_dataset():
    datasets = load_datasets()
    default_name = "car_dataset.csv"
    model_dir = os.path.join(MODELS_DIR, default_name)
    default_dataset_path = os.path.join(DATASETS_DIR, default_name)

    if not os.path.exists(default_dataset_path):
        data = {
            "Car_Name": [f"Car{i}" for i in range(1,21)],
            "Budget_Lakh": [5,7,10,8,12,9,20,25,15,18,6,11,13,14,16,19,21,22,24,23],
            "Mileage_kmpl": [12,15,18,20,10,14,16,11,19,13,17,12,20,18,15,16,14,13,12,19],
            "Seating_Capacity": [5,7,5,5,7,5,5,7,5,5,5,5,7,5,5,7,5,5,7,5],
            "Car_Type": ["SUV","MUV","Sedan","Hatchback"]*5,
            "Fuel_Type": ["Petrol","Diesel","Petrol","Electric","Diesel"]*4,
            "Transmission": ["Manual","Automatic"]*10,
            "Condition": ["New","Used"]*10,
            "User_Needs": ["Family","Sporty","Luxury"]*7,
            "Car_Color": ["Red","Blue","Black","White","Silver"]*4
        }
        pd.DataFrame(data).to_csv(default_dataset_path, index=False)

    # Train default KNN
    try:
        df = load_dataset(default_dataset_path)
        df_proc = preprocess_fit_save(df, model_dir)
        train_knn(df_proc, df, model_dir)
        datasets[default_name] = default_dataset_path
        save_datasets(datasets)
    except Exception as e:
        print(f"Init default dataset/train error: {e}")

initialize_default_dataset()

@app.route("/", methods=["GET","POST"])
def index():
    datasets = load_datasets()
    result, top5, graph_path = (False, None, None)

    if request.method == "POST":
        try:
            selected_dataset = request.form.get("selected_dataset")
            if not selected_dataset:
                flash("Please select dataset.", "error")
                return redirect(url_for("index"))

            dataset_path = datasets.get(selected_dataset)
            model_dir = os.path.join(MODELS_DIR, selected_dataset)

            req = ["budget","mileage","seating","car_type","fuel","transmission","condition","features","car_color"]
            if not all(request.form.get(x) for x in req):
                flash("Fill all fields.", "error")
                return redirect(url_for("index"))

            user_input = {
                "Budget_Lakh": float(request.form.get("budget")),
                "Mileage_kmpl": float(request.form.get("mileage")),
                "Seating_Capacity": int(request.form.get("seating")),
                "Car_Type": request.form.get("car_type"),
                "Fuel_Type": request.form.get("fuel"),
                "Transmission": request.form.get("transmission"),
                "Condition": request.form.get("condition"),
                "User_Needs": request.form.get("features"),
                "Car_Color": request.form.get("car_color")
            }

            user_df = pd.DataFrame([user_input])
            X_user_proc = preprocess_transform(user_df, model_dir)
            knn, df_proc, df_original = load_knn(model_dir)
            if knn is None:
                flash("KNN model not found.", "error")
                return redirect(url_for("index"))

            distances, indices = knn_recommend(knn, X_user_proc)
            top_idx = indices[0][:TOP_N]
            top_dist = distances[0][:TOP_N]
            top_raw = df_original.iloc[top_idx].reset_index(drop=True)

            top5 = []
            for i in range(len(top_raw)):
                rec = top_raw.iloc[i].to_dict()
                rec["Distance"] = float(top_dist[i])
                top5.append(rec)

            save_query_local(model_dir, {"user_input": user_input, "recommendations": top5})

            # Graph
            graph_type = request.form.get("graph_type","distance")
            plt.figure(figsize=(7,4))
            if graph_type == "distance":
                plt.bar(range(1,len(top_dist)+1), top_dist)
                plt.title("Distance to Top 5 Cars")
                plt.xlabel("Rank")
                plt.ylabel("Euclidean Distance")
            else:
                from sklearn.decomposition import PCA
                pca = PCA(n_components=2)
                coords = pca.fit_transform(df_proc.values)
                user_coord = pca.transform(X_user_proc.values)
                plt.scatter(coords[:,0], coords[:,1], alpha=0.7,label="Dataset")
                plt.scatter(user_coord[:,0],user_coord[:,1],color='red',s=80,label="User")
                plt.title("PCA: Dataset vs User")
                plt.legend()
            buf = io.BytesIO()
            plt.tight_layout()
            plt.savefig(buf, format="png")
            plt.close()
            graph_path = base64.b64encode(buf.getvalue()).decode('utf-8')

            result = True
        except Exception as e:
            flash(f"Error: {e}", "error")

    return render_template("index.html", datasets=sorted(datasets.keys()), result=result, top5=top5, graph_path=graph_path)

@app.route("/admin", methods=["GET","POST"])
def admin():
    if request.method == "POST":
        upload_user = request.form.get("upload_user","").upper()
        if upload_user not in ALLOWED_UPLOAD_USERS:
            flash("Unauthorized upload user.", "error")
            return redirect(url_for("admin"))

        if "dataset" not in request.files or request.files["dataset"].filename == "":
            flash("No file selected.", "error")
            return redirect(url_for("admin"))

        f = request.files["dataset"]
        dataset_name = f.filename
        dataset_path = os.path.join(DATASETS_DIR, dataset_name)
        model_dir = os.path.join(MODELS_DIR, dataset_name)
        try:
            f.save(dataset_path)
            df = load_dataset(dataset_path)

            # CLEANING
            for col in NUMERICAL_COLUMNS:
                df[col] = pd.to_numeric(df[col], errors='coerce')
                df[col].fillna(df[col].mean(), inplace=True)
            for col in CATEGORICAL_COLUMNS:
                df[col] = df[col].fillna('missing')

            df_proc = preprocess_fit_save(df, model_dir)
            train_knn(df_proc, df, model_dir)
            datasets = load_datasets()
            datasets[dataset_name] = dataset_path
            save_datasets(datasets)
            flash(f"Dataset '{dataset_name}' uploaded and KNN trained.", "success")
        except Exception as e:
            flash(f"Admin error: {e}", "error")
        return redirect(url_for("admin"))

    datasets = load_datasets()
    return render_template("admin.html", datasets=sorted(datasets.keys()), allowed_users=ALLOWED_UPLOAD_USERS)

@app.route("/delete_model/<dataset>")
def delete_model(dataset):
    model_dir = os.path.join(MODELS_DIR, dataset)
    try:
        import shutil
        if os.path.exists(model_dir):
            shutil.rmtree(model_dir)
        datasets = load_datasets()
        if dataset in datasets:
            datasets.pop(dataset)
            save_datasets(datasets)
        flash("Model deleted.", "success")
    except Exception as e:
        flash(f"Delete error: {e}", "error")
    return redirect(url_for("admin"))

if __name__ == "__main__":
    app.run(debug=True)
