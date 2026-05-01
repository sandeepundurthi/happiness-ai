# HappiScope AI 🌍  
### Global Happiness Intelligence Dashboard (2005–2025)

HappiScope AI is an end-to-end data analytics and machine learning project built using the World Happiness Report dataset. The platform analyzes 20 years of happiness trends across 150+ countries, predicts happiness scores using machine learning, and provides an interactive Dash dashboard for global rankings, country comparisons, and scenario-based forecasting.

---

# 🚀 Project Highlights

✅ Cleaned and prepared panel data covering 2005–2025 across 150+ countries  
✅ Performed trend analysis on country happiness scores over time  
✅ Built an XGBoost regression model achieving **R² = 0.91**  
✅ Identified top happiness drivers using explainable AI (SHAP)  
✅ Developed an interactive Dash dashboard for insights and forecasting  

---

# 📊 Key Insights

- Social Support was the strongest predictor of happiness  
- GDP per Capita and Healthy Life Expectancy were major contributors  
- Finland remained among the happiest countries globally  
- Some countries showed major gains, while others declined sharply over time  

---

# 🛠️ Tech Stack

- Python  
- Pandas  
- NumPy  
- Scikit-learn  
- XGBoost  
- SHAP  
- Dash  
- Plotly  
- Matplotlib  
- Seaborn  

---

# 📁 Project Structure

```text
happiness-ai/
│
├── app/
│   └── dash_app.py
│
├── data/
│   ├── raw/
│   └── processed/
│
├── models/
│   └── happiness_xgb.pkl
│
├── reports/
│   ├── global_happiness_trend.png
│   ├── top15_latest.png
│   └── shap_summary.png
│
├── src/
│   ├── load_data.py
│   ├── clean_data.py
│   ├── eda.py
│   ├── model.py
│   └── shap_analysis.py
│
├── requirements.txt
└── README.md
```

⚙️ Installation & Run

1️⃣ Clone Repository

git clone https://github.com/yourusername/happiness-ai.git
cd happiness-ai

2️⃣ Create Virtual Environment

python3 -m venv .venv
source .venv/bin/activate

3️⃣ Install Dependencies

pip install -r requirements.txt

▶️ Run Data Pipeline

Load Dataset
python src/load_data.py

Clean Data
python src/clean_data.py

Exploratory Analysis
python src/eda.py

Train Model
python src/model.py

SHAP Explainability
python src/shap_analysis.py

🌐 Launch Dashboard
python app/dash_app.py

Then open:

http://127.0.0.1:8050/

📈 Model Performance
Metric	Score
MAE	0.228
RMSE	0.324
R²	0.912

🧠 Top Predictive Drivers
Social Support
GDP per Capita
Healthy Life Expectancy
Freedom
Corruption

💼 Resume Description

Built an end-to-end analytics platform using 2005–2025 World Happiness data across 150+ countries; cleaned panel data, analyzed country trends, and developed an XGBoost model (R² = 0.91) to predict happiness scores. Deployed an interactive Dash dashboard for rankings, comparisons, and forecasting.

🔮 Future Enhancements

Global choropleth happiness map
Forecast scores to 2030
Country clustering analysis
AI-generated country insights
Cloud deployment (Render / Azure / AWS)
📬 Contact

Sandeep Undurthi
