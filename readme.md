

# Ignitron – Smart Car Selection Using AI (KNN)

## Project Overview

**Ignitron** is a **smart car selection web application** powered by **unsupervised machine learning (KNN)**.
It allows users to input their car preferences (budget, mileage, seating, car type, fuel type, etc.) and receive the **top 5 recommended cars** closest to their requirements.

The project also provides **interactive visualizations**, including:

* **Distance ranking bar chart** – showing how close each recommendation is to the user input.
* **PCA scatter plot** – showing user input against the dataset in 2D space for better understanding.

The **admin portal** allows authorized users to upload **clean datasets** that the system uses to train/update the KNN model.

---

## AIML Concepts Used

* **Unsupervised Learning**: KNN is used to find cars closest to the user’s requirements using **Euclidean distance**.
* **Feature Encoding**: Converts categorical variables into numeric form using **OrdinalEncoder** for KNN processing.
* **Dimensionality Reduction (PCA)**: Reduces high-dimensional data to 2D for visualization alongside user input.
* **Recommendation System**: Provides **content-based recommendations** without requiring labeled output.

---

## Project Files and Explanation

### 1. `app.py`

* Main Flask backend application.
* Handles routes:

  * `/` – user interface for input, recommendations, and graph selection.
  * `/admin` – admin interface to upload CSV datasets and retrain KNN model.
  * `/delete_model/<dataset>` – delete old dataset/model.
* Saves queries locally for auditing purposes.
* Generates graphs as **base64 images** for easy web display.

---

### 2. `datasets/`

* Folder containing uploaded CSV datasets.
* Must include **clean data** (no missing values or NaNs).

---

### 3. `models/`

* Stores **trained KNN models** and preprocessing artifacts per dataset:

  * `imputer.pkl` → stores numeric/categorical imputation values.
  * `encoder.pkl` → ordinal encoder for categorical columns.
  * `df_proc.csv` → processed dataset for KNN.
  * `df_original.csv` → original dataset for display.
  * `columns.json` → column order of features.
  * `knn.pkl` → trained KNN model.

---

### 4. `templates/`

* HTML templates for front-end UI using **TailwindCSS** and animated effects.

**Files**:

1. `base.html` – main layout, navigation, header, footer.
2. `index.html` – user interface for car selection, graph selection, and displaying **top 5 cars**.
3. `admin.html` – admin portal for uploading datasets with instructions to **upload clean data only**.

---

### 5. `utils/preprocess.py`

* Handles **preprocessing of datasets and user input**:

  * Keeps only allowed feature columns.
  * Imputes missing values (though datasets should ideally be clean).
  * Encodes categorical variables into numeric using **OrdinalEncoder**.
  * Saves preprocessing artifacts for future transformations.
  * Ensures user input aligns with trained model features.

---

### 6. `utils/knn_engine.py`

* Handles **KNN training and recommendations**:

  * Trains `NearestNeighbors` model.
  * Saves model in `models/`.
  * Provides functions to load KNN model and return **distances & indices**.
  * Supports multiple datasets for recommendations.

---

### 7. `requirements.txt`

* Python libraries required:

```
flask
pandas
matplotlib
scikit-learn
joblib
numpy
```

* Install via:

```bash
pip install -r requirements.txt
```

---

## Dataset Guidelines

* **Upload clean CSV datasets only** via admin portal.
* Required columns:

```
Car_Name, Budget_Lakh, Mileage_kmpl, Seating_Capacity, Car_Type, Fuel_Type, Transmission, Condition, User_Needs, Car_Color
```

* Example: synthetic dataset of up to **1000 cars**.
* Admin uploads dataset → preprocessing → KNN model is trained automatically.

---

## How to Use

### Admin Portal

1. Go to `/admin`.
2. Enter authorized username (e.g., `NIPUNNAIK`).
3. Upload **clean CSV dataset**.
4. System preprocesses dataset and trains KNN model automatically.
5. Only authorized users can upload or delete datasets.

### User Portal

1. Go to `/`.
2. Select **dataset**.
3. Input car requirements:

   * Budget (Lakh)
   * Mileage (kmpl)
   * Seating Capacity
   * Car Type (SUV, MUV, Sedan, etc.)
   * Fuel Type
   * Transmission
   * Condition
   * Features (Family, Sporty, Luxury)
   * Car Color
4. Choose **graph type** to visualize output.
5. See **top 5 recommended cars** with distance score.

---

## Notes

* **NaN or missing values** in datasets will cause KNN to fail.
* Users can choose graphs for visualization:

  * **Distance ranking** – bar chart.
  * **PCA 2D plot** – scatter plot.
* The system is **unsupervised KNN-based**; no supervised ML labels are required.

---

## Summary

Ignitron provides a **smart, interactive, AI-driven car recommendation system**:

* **Unsupervised KNN** for recommendation.
* **TailwindCSS** front-end with animated UI.
* **Admin-controlled dataset updates**.
* **Top 5 recommendations** with graph visualizations.
* **Clean dataset requirement** ensures accuracy and prevents errors.

