# 🌳 Employee Attrition Prediction AI: End-to-End ML System
An end-to-end Machine Learning web application designed to predict whether an employee is likely to leave the company — classified as **High Risk** or **Low Risk** with real-time probability scoring. This project leverages an optimized **Random Forest Classifier** to analyze 31 HR parameters, delivering instant attrition insights through a sophisticated dark-mode professional dashboard.

## 🚀 Live Demo
[🔗 View Live App](https://gqx7y9xvvcp2qv78jsraub.streamlit.app/)

## 🛠️ Tech Stack
- **Engine:** Python 3.10+
- **Machine Learning:** Scikit-Learn (RandomForestClassifier + DecisionTreeClassifier)
- **Web Framework:** Streamlit
- **Data Handling:** Pandas & NumPy
- **Visuals:** Plotly Express & Graph Objects

## 📊 Model Performance
The model was trained on the **IBM HR Analytics Employee Attrition Dataset** (1,470 employees × 31 features), with class balancing applied to handle the imbalanced target:

- **Optimization:** Bootstrap Aggregation (Bagging) + Random Feature Subsets
- **Class Balancing:** Minority class upsampling for unbiased predictions
- **Reliability:** Real-time attrition risk score (0–100%) for every employee profile
- **Key Features:** OverTime, MonthlyIncome, Age, JobSatisfaction, YearsAtCompany (31 total parameters)

## 💡 Features
- **31-Parameter Analysis:** Full HR profile input covering personal, job, and satisfaction dimensions
- **Instant Inference:** Pre-trained model loaded at startup — zero waiting time for end users
- **Dynamic Risk Score:** Real-time gauge showing attrition probability with color-coded risk levels (Low / Moderate / High / Very High)
- **Interactive EDA:** Explore dataset distributions, correlations, and attrition breakdown by department, role, and more
- **Dual Model Support:** Train and compare both Decision Tree and Random Forest with custom hyperparameters
- **Model Comparison Dashboard:** Radar chart + bar chart + overfitting analysis across multiple model configurations
- **Feature Importance:** Ranked visualization of the top drivers behind employee attrition
- **Model Export:** Download any trained model as a `.pkl` file for external use
- **Professional Dark UI:** Gradient design built with Syne + Space Mono typography

## 📂 Project Structure
```text
├── app.py                      # Main Streamlit application code
├── requirements.txt            # Required Python libraries (pandas, sklearn, etc.)
└── README.md                   # Project documentation & setup guide
```

## ⚙️ Local Setup
```bash
# 1. Clone the repository
git clone https://github.com/HafsaIbrahim5/Employee_Attrition.git
cd Employee_Attrition

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run the app
streamlit run app.py
```

## 🧠 ML Pipeline
```
Raw Data → Preprocessing (Label Encoding) → Class Balancing (Upsampling)
→ Train/Test Split → Model Training → Evaluation (Accuracy, F1, AUC)
→ Hyperparameter Tuning → Pre-loaded Inference → Streamlit Deployment
```

## 📈 App Pages
| Page | Description |
|------|-------------|
| 🏠 Home & Theory | Decision Tree & Random Forest explained with interactive diagrams |
| 📊 Data Explorer | EDA — distributions, correlations, attrition analysis |
| 🤖 Train Models | Train DT or RF with custom hyperparameters + full evaluation suite |
| 🔮 Predict Attrition | Instant risk prediction with dynamic gauge |
| 📈 Model Comparison | Radar chart, bar chart, overfitting analysis |
| 👤 About | Project details, tech stack, references |

## 📚 References
- [IBM HR Analytics Employee Attrition Dataset — Kaggle](https://www.kaggle.com/datasets/pavansubhasht/ibm-hr-analytics-attrition-dataset)
- [Decision Trees & Random Forest for Beginners — Kaggle Notebook](https://www.kaggle.com/code/faressayah/decision-trees-random-forest-for-beginners)
- [scikit-learn Documentation](https://scikit-learn.org)
- [Hyperparameter Tuning the Random Forest — Towards Data Science](https://towardsdatascience.com/hyperparameter-tuning-the-random-forest-in-python-using-scikit-learn-28d2aa77dd74)

---

Built to demonstrate a complete ML pipeline — from exploratory analysis to a production-ready interactive dashboard.

**👩‍💻 Developed by [Hafsa Ibrahim](https://www.linkedin.com/in/hafsa-ibrahim-ai-mi/) — AI & Machine Intelligence Engineer**
[![LinkedIn](https://img.shields.io/badge/LinkedIn-0077B5?style=flat&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/hafsa-ibrahim-ai-mi/)
[![GitHub](https://img.shields.io/badge/GitHub-100000?style=flat&logo=github&logoColor=white)](https://github.com/HafsaIbrahim5)
