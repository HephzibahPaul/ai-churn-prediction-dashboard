# 📊 AI-Powered Customer Churn Prediction Dashboard  
### Banking & Financial Services · Machine Learning · Business Analytics

This project delivers an end-to-end **Customer Churn Prediction System** combining:

- **Machine Learning (Random Forest Model)**
- **Business Intelligence (Segmentation & Churn Drivers)**
- **Prescriptive Retention Insights (LTV-Based Recommendations)**
- **Interactive Dashboard (Streamlit + Altair)**

Designed from the perspective of a **Business Analyst** working with:
Customer Retention · CRM · Product · Risk · Data Science teams.

---

## 🚀 1. Key Features

### 🔮 **Single Customer Prediction**
- Predicts *churn probability* (0%–100%)
- Assigns risk category: **Low · Medium · High**
- Provides **actionable retention recommendations**
- Computes **Customer LTV (Lifetime Value)**
- Suggests **maximum retention budget**

---

### 📊 **Portfolio Insights (Churn Dashboard)**
- Churn by **Geography**
- Churn by **Age Group**
- Rule-Based **Customer Segments**
  - Loyal  
  - At-Risk  
  - High-Value At-Risk  
  - New / Neutral
- **Top Churn Drivers** (Feature Importance)
- High-Value At-Risk Customer Table

---

## 🏗 2. Project Structure

churn_prediction_project/
│── dashboard/
│ └── app.py
│── data/
│ ├── raw_customers.csv
│ └── generate_synthetic_data.py
│── models/
│ ├── churn_model.pkl
│ ├── feature_cols.pkl
│ └── feature_importances.pkl
│── screenshots/
│ ├── dashboard_home.png
│ ├── prediction_result.png
│ ├── churn_insights_geo.png
│ ├── churn_insights_age.png
│ ├── segments_chart.png
│ ├── drivers_chart.png
│ └── high_value_at_risk_table.png
│── src/
│ ├── train_model.py
│ ├── data_preprocessing.py
│ └── init.py
│── requirements.txt
│── README.md
│── LICENSE

yaml
Copy code

---

## ⚙️ 3. Installation & Setup

### **Create virtual environment**
```bash
python -m venv venv
Activate environment
bash
Copy code
venv\Scripts\activate
Install dependencies
bash
Copy code
pip install -r requirements.txt
Run the Dashboard
bash
Copy code
cd dashboard
streamlit run app.py
🧠 4. Machine Learning Overview
Model: Random Forest Classifier

Training approach includes:

Train-test split

One-hot encoding

Feature alignment

Feature importance extraction

Evaluated on:

Accuracy

Precision / Recall

F1-score

ROC-AUC

💼 5. Business Value Delivered
This project helps business teams:

Reduce churn via early identification

Discover at-risk & high-value customers

Allocate retention budget using LTV

Understand drivers behind customer churn

Improve customer engagement strategies

Accelerate data-driven decision-making

📸 6. Dashboard Screenshots
(Add your screenshots here)

📜 7. License
This project uses the MIT License.
See the full license in the LICENSE file.

👩‍💻 8. Author
Hephzibah Paul
AI & Business Analytics
GitHub: https://github.com/HephzibahPaul