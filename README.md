

# 🚗 **IGNITRON – AI Car Recommendation & Price Prediction System**

### *An Intelligent Car Assistant Powered by Machine Learning & Flask*

Ignitron is an AI-powered car recommendation and price prediction system built using **Python**, **Flask**, and machine learning models like **KNN** and **Random Forest**.
The application allows **admins to upload a clean CSV dataset**, automatically preprocesses the data, trains the ML model, and updates the system with new predictions.

The **user home page** displays meaningful insights such as **PCA visualization**, **Bar charts**, and dataset summaries that help users understand car trends and pricing patterns.

---

# 👨‍💻 **Admin Team**

The admin panel is accessible only to the authorized project maintainers:

* **Atharva Khaire**
* **Anish Gosavi**
* **Nipun Naik**
* **Omkar Kasar**

Admins can upload a CSV, view analytics, and retrain the model.

---

# 📁 **Project Structure**

```
IGNITRON_CAPSTONE_
│── datasets/
│   ├── car_dataset.csv
│   ├── clean_indian_car_dataset_5000.csv
│   ├── family.csv
│   ├── mix.csv
│   ├── sports.csv
│   └── ...
│
│── models/                # (Optional: saved models folder)
│
│── static/                # CSS, JS, images
│
│── templates/
│   ├── admin.html         # Admin CSV upload & training page
│   ├── base.html
│   └── index.html         # Homepage with charts (PCA, bar graphs)
│
│── utils/
│   ├── knn_engine.py      # ML model logic (KNN)
│   ├── preprocess.py      # Data cleaning & preprocessing
│   └── random_forest...   # Optional model file
│
│── app.py                 # Main Flask application
│── requirements.txt       # Python dependencies
│── README.md
│── datasets.json          # Dataset selection metadata
```

---

# ✨ **Key Features**

### 🔹 1. **Admin CSV Upload**

* Admins can upload *any clean car dataset* from the admin panel.
* The system automatically:

  * Validates the CSV
  * Preprocesses the dataset
  * Trains the ML model (KNN / Random Forest)
  * Saves updated model files

### 🔹 2. **AI-Powered Car Recommendation**

* Based on input features, the system predicts:

  * Car price
  * Car category suitability
  * Comparison between similar cars

### 🔹 3. **Interactive Data Visualizations**

Home page shows:

* PCA Scatter Plot (Feature reduction visualization)
* Bar Charts (Car type counts, brand distribution, etc.)
* Insights extracted from uploaded datasets

### 🔹 4. **Modular Architecture**

* Clean separation using `utils/` for ML engines & preprocessing
* `templates/` for UI
* `static/` for styling & JS

---

# 🚀 **How to Run IGNITRON (Step-by-step)**

### **📌 Step 1: Install Requirements**

Open terminal in the project folder and run:

```bash
pip install -r requirements.txt
```

---

### **📌 Step 2: Run the Flask App**

```bash
python app.py
```

If using Windows PowerShell, you may need:

```bash
python .\app.py
```

---

### **📌 Step 3: Open in Browser**

Flask will start at:

```
http://127.0.0.1:5000/
```

---

# 🛠 **Project Usage Instructions**

### 🔐 **Admin Panel**

Go to:

```
http://127.0.0.1:5000/admin
```

Here admins can:

* Upload a CSV dataset
* Start preprocessing
* Retrain ML model
* View dataset summary

---

### 🏠 **User Homepage**

Visit:

```
http://127.0.0.1:5000/
```

What you will see:

* PCA Visualization
* Bar Graphs
* Dataset summary cards
* Car recommendation insights

---

# 📈 **Machine Learning Models Used**

* **KNN (K-Nearest Neighbors)** – for recommendation & similarity matching
* **Random Forest Regressor** – for price prediction
* **PCA (Principal Component Analysis)** – for visualization

---

# 🔮 **Future Enhancements**

* Deploy Ignitron on Render / Railway
* Add user login system
* Add car comparison dashboard
* Include real-time market price scraping
* Add fuel efficiency & reseller recommendation module

---

# 🧡 Credits

Project developed by:
**Atharva Khaire, Anish Gosavi, Nipun Naik, and Omkar Kasar**


