import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import pickle
import base64
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
    f1_score,
)
from sklearn.utils import resample
import warnings

warnings.filterwarnings("ignore")

# ──────────────────────────────────────────────────────
# PAGE CONFIG
# ──────────────────────────────────────────────────────
st.set_page_config(
    page_title="Employee Attrition AI | Hafsa Ibrahim",
    page_icon="🌳",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ──────────────────────────────────────────────────────
# COLORS
# ──────────────────────────────────────────────────────
TEAL = "#00d4aa"
PURPLE = "#7c3aed"
PINK = "#f472b6"
AMBER = "#fbbf24"
BLUE = "#60a5fa"
GREEN = "#34d399"
COLORS = [TEAL, PURPLE, PINK, AMBER, BLUE, GREEN, "#fb923c", "#a78bfa"]

# ──────────────────────────────────────────────────────
# CSS
# ──────────────────────────────────────────────────────
CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=Syne:wght@400;700;800&family=DM+Sans:wght@300;400;500;600&display=swap');
:root{
  --bg0:#080d1a; --bg1:#0d1426; --bg2:#111b33;
  --card:#131c30; --card2:#1a2540;
  --teal:#00d4aa; --purple:#7c3aed; --pink:#f472b6; --amber:#fbbf24;
  --blue:#60a5fa; --green:#34d399;
  --txt:#e2e8f7; --muted:#7d8db5;
  --border:rgba(0,212,170,.13);
}
html,body,[data-testid="stAppViewContainer"]{
  background:var(--bg0)!important;
  color:var(--txt)!important;
  font-family:"DM Sans",sans-serif!important;
}
[data-testid="stSidebar"]{
  background:var(--bg1)!important;
  border-right:1px solid var(--border)!important;
}
[data-testid="stSidebar"] *{color:var(--txt)!important;}
h1,h2,h3,h4{font-family:"Syne",sans-serif!important;}

.hero{
  background:linear-gradient(135deg,#0d1426 0%,#160e2e 50%,#080d1a 100%);
  border:1px solid var(--border);border-radius:22px;
  padding:44px 48px;margin-bottom:32px;
  position:relative;overflow:hidden;
}
.hero::before{
  content:"";position:absolute;inset:0;
  background:radial-gradient(ellipse 60% 60% at 20% 50%,rgba(0,212,170,.07),transparent);
  pointer-events:none;
}
.hero-title{
  font-family:"Syne",sans-serif;font-size:3rem;font-weight:800;line-height:1.05;
  background:linear-gradient(130deg,var(--teal) 0%,var(--purple) 55%,var(--pink) 100%);
  -webkit-background-clip:text;-webkit-text-fill-color:transparent;background-clip:text;
  margin-bottom:12px;
}
.hero-sub{color:var(--muted);font-size:1.1rem;font-weight:300;letter-spacing:.3px;}
.badge{
  display:inline-block;background:rgba(0,212,170,.1);
  border:1px solid rgba(0,212,170,.3);color:var(--teal);
  padding:3px 13px;border-radius:20px;
  font-family:"Space Mono",monospace;font-size:.72rem;
  margin-right:8px;margin-bottom:14px;
}
.mgrid{display:grid;grid-template-columns:repeat(4,1fr);gap:14px;margin:18px 0;}
.mcard{
  background:var(--card);border:1px solid var(--border);
  border-radius:16px;padding:22px 18px;text-align:center;
  position:relative;overflow:hidden;
}
.mcard::after{
  content:"";position:absolute;bottom:0;left:0;right:0;height:3px;
  background:linear-gradient(90deg,var(--teal),var(--purple));
}
.mval{
  font-family:"Syne",sans-serif;font-size:2rem;font-weight:800;
  background:linear-gradient(135deg,var(--teal),var(--purple));
  -webkit-background-clip:text;-webkit-text-fill-color:transparent;background-clip:text;
}
.mlabel{
  color:var(--muted);font-size:.72rem;
  font-family:"Space Mono",monospace;
  text-transform:uppercase;letter-spacing:1.2px;margin-top:5px;
}
.sec{display:flex;align-items:center;gap:12px;margin:30px 0 18px;}
.sec-num{
  font-family:"Space Mono",monospace;font-size:.68rem;color:var(--teal);
  background:rgba(0,212,170,.1);border:1px solid rgba(0,212,170,.25);
  padding:3px 10px;border-radius:6px;
}
.sec-title{font-family:"Syne",sans-serif;font-size:1.45rem;font-weight:700;color:var(--txt);}
.icard{
  background:var(--card);border:1px solid var(--border);
  border-radius:14px;padding:22px 24px;margin:14px 0;
}
.icard-t {border-left:4px solid var(--teal);}
.icard-p {border-left:4px solid var(--purple);}
.icard-pk{border-left:4px solid var(--pink);}
.icard-a {border-left:4px solid var(--amber);}

/* instant-badge shown on predict page */
.instant-badge{
  display:inline-flex;align-items:center;gap:6px;
  background:rgba(0,212,170,.1);border:1px solid rgba(0,212,170,.3);
  color:var(--teal);padding:5px 14px;border-radius:30px;
  font-family:"Space Mono",monospace;font-size:.72rem;
  margin-bottom:16px;
}

.pred-yes{
  background:linear-gradient(135deg,rgba(239,68,68,.1),rgba(239,68,68,.05));
  border:2px solid rgba(239,68,68,.4);border-radius:20px;padding:32px;text-align:center;
}
.pred-no{
  background:linear-gradient(135deg,rgba(0,212,170,.1),rgba(0,212,170,.05));
  border:2px solid rgba(0,212,170,.4);border-radius:20px;padding:32px;text-align:center;
}
.pred-title{font-family:"Syne",sans-serif;font-size:2.2rem;font-weight:800;}
.pred-pct  {font-family:"Syne",sans-serif;font-size:3.5rem;font-weight:800;margin:10px 0;}
.pred-desc {color:var(--muted);font-size:.95rem;line-height:1.7;}

.footer{
  background:var(--bg1);border:1px solid var(--border);border-radius:16px;
  padding:26px 30px;margin-top:48px;
  display:flex;justify-content:space-between;align-items:center;flex-wrap:wrap;gap:14px;
}
.footer-name{
  font-family:"Syne",sans-serif;font-size:1.15rem;font-weight:700;
  background:linear-gradient(135deg,var(--teal),var(--purple));
  -webkit-background-clip:text;-webkit-text-fill-color:transparent;background-clip:text;
}
.flink{
  display:inline-flex;align-items:center;gap:7px;
  background:rgba(0,212,170,.07);border:1px solid rgba(0,212,170,.22);
  color:var(--teal)!important;padding:8px 18px;border-radius:40px;
  font-family:"Space Mono",monospace;font-size:.75rem;text-decoration:none!important;
}

/* Streamlit overrides */
.stButton>button{
  background:linear-gradient(135deg,var(--teal),#00a884)!important;
  color:#080d1a!important;border:none!important;border-radius:10px!important;
  font-family:"DM Sans",sans-serif!important;font-weight:600!important;
  padding:10px 26px!important;
  box-shadow:0 4px 18px rgba(0,212,170,.3)!important;
}
.stButton>button:hover{transform:translateY(-2px)!important;}
[data-testid="stMetricValue"]{
  color:var(--teal)!important;font-family:"Syne",sans-serif!important;
}
.stTabs [data-baseweb="tab-list"]{
  background:var(--card)!important;border-radius:12px!important;padding:4px!important;
}
.stTabs [aria-selected="true"]{
  background:rgba(0,212,170,.15)!important;
  color:var(--teal)!important;border-radius:8px!important;
}
.stDataFrame{border:1px solid var(--border)!important;border-radius:12px!important;}
[data-testid="stExpander"]{
  background:var(--card)!important;border:1px solid var(--border)!important;
  border-radius:12px!important;
}
.stSelectbox>div>div,.stNumberInput>div>div{
  background:var(--card)!important;border-color:var(--border)!important;
}
</style>
"""


# ──────────────────────────────────────────────────────
# HELPERS
# ──────────────────────────────────────────────────────
def plotly_dark(fig, title="", height=400):
    fig.update_layout(
        title=dict(text=title, font=dict(family="Syne", size=14, color="#e2e8f7")),
        height=height,
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#7d8db5", family="DM Sans"),
        xaxis=dict(
            gridcolor="rgba(255,255,255,0.04)", zerolinecolor="rgba(255,255,255,0.04)"
        ),
        yaxis=dict(
            gridcolor="rgba(255,255,255,0.04)", zerolinecolor="rgba(255,255,255,0.04)"
        ),
        margin=dict(l=40, r=20, t=50, b=40),
        colorway=COLORS,
    )
    return fig


def render_footer():
    st.markdown(
        """
    <div class="footer">
      <div>
        <div class="footer-name">Hafsa Ibrahim</div>
        <div style="color:var(--muted);font-size:.8rem;margin-top:3px;">
          AI &amp; Machine Intelligence Engineer
        </div>
      </div>
      <div style="display:flex;gap:14px;flex-wrap:wrap;">
        <a href="https://www.linkedin.com/in/hafsa-ibrahim-ai-mi/"
           target="_blank" class="flink">&#128279; LinkedIn</a>
        <a href="https://github.com/HafsaIbrahim5"
           target="_blank" class="flink">&#128025; GitHub</a>
      </div>
    </div>
    """,
        unsafe_allow_html=True,
    )


# ──────────────────────────────────────────────────────
# DATA
# ──────────────────────────────────────────────────────
@st.cache_data
def sample_data():
    np.random.seed(42)
    n = 1470
    dept_m = {0: "Human Resources", 1: "Research & Development", 2: "Sales"}
    edu_m = {
        0: "Human Resources",
        1: "Life Sciences",
        2: "Marketing",
        3: "Medical",
        4: "Other",
        5: "Technical Degree",
    }
    role_m = {
        0: "Healthcare Representative",
        1: "Human Resources",
        2: "Laboratory Technician",
        3: "Manager",
        4: "Manufacturing Director",
        5: "Research Director",
        6: "Research Scientist",
        7: "Sales Executive",
        8: "Sales Representative",
    }
    ms_m = {0: "Divorced", 1: "Married", 2: "Single"}
    bt_m = {0: "Non-Travel", 1: "Travel_Frequently", 2: "Travel_Rarely"}

    age = np.random.randint(18, 61, n)
    income = np.random.randint(1009, 19999, n)
    overtime = np.where(np.random.rand(n) > 0.72, "Yes", "No")
    distance = np.random.randint(1, 30, n)
    job_sat = np.random.randint(1, 5, n)
    wlb = np.random.randint(1, 5, n)
    yrs_co = np.random.randint(0, 41, n)

    # Realistic multi-factor attrition probability
    att_p = (
        np.where(age < 28, 0.12, 0.0)
        + np.where(income < 3000, 0.20, 0.0)
        + np.where(income < 5000, 0.10, 0.0)
        + np.where(overtime == "Yes", 0.18, 0.0)
        + np.where(distance > 20, 0.08, 0.0)
        + np.where(job_sat == 1, 0.15, 0.0)
        + np.where(wlb == 1, 0.10, 0.0)
        + np.where(yrs_co <= 1, 0.12, 0.0)
        + np.where(yrs_co >= 10, -0.10, 0.0)
        + np.where(income > 10000, -0.12, 0.0)
        + np.where(job_sat >= 3, -0.08, 0.0)
        + 0.04  # base rate
    )
    att_p = np.clip(att_p, 0.02, 0.92)
    attrition = np.where(np.random.rand(n) < att_p, "Yes", "No")
    vm = np.vectorize(lambda d, m: m[d])

    df = pd.DataFrame(
        {
            "Age": age,
            "Attrition": attrition,
            "BusinessTravel": vm(np.random.randint(0, 3, n), bt_m),
            "DailyRate": np.random.randint(102, 1500, n),
            "Department": vm(
                np.random.choice([0, 1, 2], n, p=[0.07, 0.65, 0.28]), dept_m
            ),
            "DistanceFromHome": np.random.randint(1, 30, n),
            "Education": np.random.randint(1, 6, n),
            "EducationField": vm(np.random.randint(0, 6, n), edu_m),
            "EnvironmentSatisfaction": np.random.randint(1, 5, n),
            "Gender": np.where(np.random.rand(n) > 0.4, "Male", "Female"),
            "HourlyRate": np.random.randint(30, 101, n),
            "JobInvolvement": np.random.randint(1, 5, n),
            "JobLevel": np.random.randint(1, 6, n),
            "JobRole": vm(np.random.randint(0, 9, n), role_m),
            "JobSatisfaction": np.random.randint(1, 5, n),
            "MaritalStatus": vm(
                np.random.choice([0, 1, 2], n, p=[0.23, 0.46, 0.31]), ms_m
            ),
            "MonthlyIncome": income,
            "MonthlyRate": np.random.randint(2094, 26999, n),
            "NumCompaniesWorked": np.random.randint(0, 10, n),
            "OverTime": overtime,
            "PercentSalaryHike": np.random.randint(11, 26, n),
            "PerformanceRating": np.random.choice([3, 4], n, p=[0.85, 0.15]),
            "RelationshipSatisfaction": np.random.randint(1, 5, n),
            "StockOptionLevel": np.random.randint(0, 4, n),
            "TotalWorkingYears": np.random.randint(0, 41, n),
            "TrainingTimesLastYear": np.random.randint(0, 7, n),
            "WorkLifeBalance": np.random.randint(1, 5, n),
            "YearsAtCompany": np.random.randint(0, 41, n),
            "YearsInCurrentRole": np.random.randint(0, 19, n),
            "YearsSinceLastPromotion": np.random.randint(0, 16, n),
            "YearsWithCurrManager": np.random.randint(0, 18, n),
        }
    )
    return df


@st.cache_data
def preprocess(df):
    df2 = df.copy()
    le_dict = {}
    for col in df2.select_dtypes("object").columns:
        le = LabelEncoder()
        df2[col] = le.fit_transform(df2[col])
        le_dict[col] = le
    return df2, le_dict


def do_train(df_enc, model_type, params, balance=False, test_size=0.2):
    X = df_enc.drop("Attrition", axis=1)
    y = df_enc["Attrition"]
    if balance:
        tmp = pd.concat([X, y], axis=1)
        maj = tmp[tmp.Attrition == 0]
        minn = tmp[tmp.Attrition == 1]
        minn_up = resample(minn, replace=True, n_samples=len(maj), random_state=42)
        tmp = pd.concat([maj, minn_up])
        X, y = tmp.drop("Attrition", axis=1), tmp["Attrition"]
    Xtr, Xte, ytr, yte = train_test_split(
        X, y, test_size=test_size, random_state=42, stratify=y
    )
    clf = (
        DecisionTreeClassifier(**params, random_state=42)
        if model_type == "Decision Tree"
        else RandomForestClassifier(**params, random_state=42, n_jobs=-1)
    )
    clf.fit(Xtr, ytr)
    return clf, Xtr, Xte, ytr, yte, list(X.columns)


def eval_model(clf, Xtr, ytr, Xte, yte):
    yp = clf.predict(Xte)
    ypp = clf.predict_proba(Xte)[:, 1]
    ypTr = clf.predict(Xtr)
    return dict(
        train_acc=accuracy_score(ytr, ypTr),
        test_acc=accuracy_score(yte, yp),
        f1=f1_score(yte, yp, zero_division=0),
        roc_auc=roc_auc_score(yte, ypp),
        cm=confusion_matrix(yte, yp),
        report=classification_report(yte, yp, output_dict=True),
        y_pred=yp,
        y_proba=ypp,
        y_test=yte,
    )


# ──────────────────────────────────────────────────────
# ★  PRE-TRAINED DEFAULT MODEL  ★
#    @st.cache_resource  →  trained ONCE, lives for the
#    entire server lifetime.  Inference is instant (<1ms).
# ──────────────────────────────────────────────────────
@st.cache_resource
def load_default_model():
    """
    Pre-trains a tuned Random Forest at app startup.
    The user never waits — Predict page is instant from the first click.
    """
    df = sample_data()
    df_enc, le_d = preprocess(df)
    params = dict(
        n_estimators=200,
        max_depth=10,
        min_samples_split=5,
        min_samples_leaf=2,
        max_features="sqrt",
    )
    clf, Xtr, Xte, ytr, yte, feats = do_train(
        df_enc, "Random Forest", params, balance=True, test_size=0.2
    )
    return clf, feats, le_d


# load at import time (cached — runs once only)
_default_clf, _default_feats, _default_le = load_default_model()


# ──────────────────────────────────────────────────────
# SIDEBAR
# ──────────────────────────────────────────────────────
st.markdown(CSS, unsafe_allow_html=True)

with st.sidebar:
    st.markdown(
        """
    <div style="text-align:center;padding:18px 0 8px;">
      <div style="font-family:Syne,sans-serif;font-size:1.4rem;font-weight:800;
                  background:linear-gradient(135deg,#00d4aa,#7c3aed);
                  -webkit-background-clip:text;-webkit-text-fill-color:transparent;">
        &#127795; Attrition AI
      </div>
      <div style="font-family:Space Mono,monospace;font-size:.62rem;
                  color:#7d8db5;letter-spacing:2px;margin-top:3px;">
        ML DASHBOARD v2.0
      </div>
    </div>
    <hr style="border-color:rgba(0,212,170,.12);margin:10px 0 18px;">
    """,
        unsafe_allow_html=True,
    )

    page = st.radio(
        "Navigation",
        [
            "&#127968;  Home & Theory",
            "&#128202;  Data Explorer",
            "&#129302;  Train Models",
            "&#128302;  Predict Attrition",
            "&#128200;  Model Comparison",
            "&#128100;  About",
        ],
        label_visibility="collapsed",
    )

    st.markdown(
        "<hr style='border-color:rgba(0,212,170,.1);margin:18px 0;'>",
        unsafe_allow_html=True,
    )
    st.markdown(
        "<p style='font-family:Space Mono,monospace;font-size:.68rem;"
        "color:#7d8db5;'>DATA SOURCE</p>",
        unsafe_allow_html=True,
    )
    data_src = st.radio(
        "", ["Sample IBM HR Dataset", "Upload CSV"], label_visibility="collapsed"
    )
    uploaded = None
    if data_src == "Upload CSV":
        uploaded = st.file_uploader(
            "Upload CSV", type=["csv"], label_visibility="collapsed"
        )

# ── load data ─────────────────────────────────────────
if uploaded is not None:
    df_raw = pd.read_csv(uploaded)
    if "Attrition" not in df_raw.columns:
        st.error("Dataset must have an Attrition column.")
        st.stop()
else:
    df_raw = sample_data()

df_enc, le_dict = preprocess(df_raw)


# ╔═══════════════════════════════════════════════╗
# ║  HOME & THEORY                                ║
# ╚═══════════════════════════════════════════════╝
if "Home" in page:
    att_yes = (df_raw["Attrition"] == "Yes").sum()
    att_rate = att_yes / len(df_raw) * 100

    st.markdown(
        f"""
    <div class="hero">
      <div>
        <span class="badge">IBM HR Analytics</span>
        <span class="badge">Classification</span>
        <span class="badge">scikit-learn</span>
        <span class="badge">Plotly</span>
      </div>
      <div class="hero-title">Employee Attrition<br>Prediction System</div>
      <div class="hero-sub">
        Predicting which employees are likely to leave using<br>
        Decision Trees &amp; Random Forest ensemble methods
      </div>
    </div>
    <div class="mgrid">
      <div class="mcard">
        <div class="mval">{len(df_raw):,}</div>
        <div class="mlabel">Total Employees</div>
      </div>
      <div class="mcard">
        <div class="mval">{df_raw.shape[1]}</div>
        <div class="mlabel">Features</div>
      </div>
      <div class="mcard">
        <div class="mval">{att_yes}</div>
        <div class="mlabel">Attrited</div>
      </div>
      <div class="mcard">
        <div class="mval">{att_rate:.1f}%</div>
        <div class="mlabel">Attrition Rate</div>
      </div>
    </div>
    """,
        unsafe_allow_html=True,
    )

    c1, c2 = st.columns(2)
    with c1:
        st.markdown(
            """
        <div class="icard icard-t">
          <h3 style="font-family:Syne,sans-serif;color:#00d4aa;margin-top:0;">
            &#127795; Decision Tree
          </h3>
          <p style="color:#7d8db5;line-height:1.85;">
            A supervised ML model that builds a
            <strong style="color:#e2e8f7;">binary tree</strong> of decisions using the
            <strong style="color:#e2e8f7;">CART</strong> algorithm.
            Each node tests a feature, branches represent outcomes,
            and leaves give predictions.
            Splits minimise <em>Gini impurity</em> or <em>entropy</em>.
          </p>
          <p style="color:#7d8db5;margin-bottom:0;">
            <strong style="color:#00d4aa;">Pros:</strong>
            Interpretable, no scaling, handles mixed types<br>
            <strong style="color:#f472b6;">Cons:</strong>
            Prone to overfitting, unstable
          </p>
        </div>
        """,
            unsafe_allow_html=True,
        )
    with c2:
        st.markdown(
            """
        <div class="icard icard-p">
          <h3 style="font-family:Syne,sans-serif;color:#7c3aed;margin-top:0;">
            &#127794; Random Forest
          </h3>
          <p style="color:#7d8db5;line-height:1.85;">
            An ensemble of trees built on
            <strong style="color:#e2e8f7;">bootstrap samples</strong>
            with <strong style="color:#e2e8f7;">random feature subsets</strong>
            at each split. The final class is a majority vote, dramatically
            reducing variance compared to a single tree.
          </p>
          <p style="color:#7d8db5;margin-bottom:0;">
            <strong style="color:#00d4aa;">Pros:</strong>
            High accuracy, robust, feature importance<br>
            <strong style="color:#f472b6;">Cons:</strong>
            Slower training, less interpretable
          </p>
        </div>
        """,
            unsafe_allow_html=True,
        )

    st.markdown(
        '<div class="sec">'
        '<span class="sec-num">02</span>'
        '<span class="sec-title">Algorithm Visualisation</span>'
        "</div>",
        unsafe_allow_html=True,
    )

    t1, t2 = st.tabs(["  Decision Tree Flow  ", "  Random Forest Flow  "])

    with t1:
        fig = go.Figure()
        nodes = [
            (0.50, 0.91, "Root<br><b>Age &lt; 35?</b>", TEAL),
            (0.25, 0.68, "OverTime<br><b>= Yes?</b>", PURPLE),
            (0.75, 0.68, "Income<br><b>&lt; $3000?</b>", PURPLE),
            (0.10, 0.44, "<b>STAY</b>", GREEN),
            (0.40, 0.44, "<b>LEAVE</b>", PINK),
            (0.62, 0.44, "Dept<br><b>= Sales?</b>", AMBER),
            (0.88, 0.44, "<b>STAY</b>", GREEN),
            (0.55, 0.18, "<b>LEAVE</b>", PINK),
            (0.70, 0.18, "<b>STAY</b>", GREEN),
        ]
        edges = [
            (0, 1, "Yes"),
            (0, 2, "No"),
            (1, 3, "No"),
            (1, 4, "Yes"),
            (2, 5, "Yes"),
            (2, 6, "No"),
            (5, 7, "Yes"),
            (5, 8, "No"),
        ]
        for s, e, lbl in edges:
            x0, y0 = nodes[s][0], nodes[s][1] - 0.035
            x1, y1 = nodes[e][0], nodes[e][1] + 0.035
            fig.add_shape(
                type="line",
                x0=x0,
                y0=y0,
                x1=x1,
                y1=y1,
                line=dict(color="rgba(136,146,176,.35)", width=1.5),
            )
            fig.add_annotation(
                x=(x0 + x1) / 2,
                y=(y0 + y1) / 2,
                text=f"<b>{lbl}</b>",
                font=dict(size=10, color=TEAL),
                showarrow=False,
                bgcolor="rgba(8,13,26,.85)",
                borderpad=3,
            )
        for x, y, txt, col in nodes:
            r, g, b = int(col[1:3], 16), int(col[3:5], 16), int(col[5:7], 16)
            fig.add_shape(
                type="rect",
                x0=x - 0.1,
                y0=y - 0.038,
                x1=x + 0.1,
                y1=y + 0.038,
                fillcolor=f"rgba({r},{g},{b},.12)",
                line=dict(color=col, width=1.8),
            )
            fig.add_annotation(
                x=x, y=y, text=txt, font=dict(size=9, color="#e2e8f7"), showarrow=False
            )
        fig.update_layout(
            xaxis=dict(visible=False, range=[0, 1]),
            yaxis=dict(visible=False, range=[0.05, 0.98]),
            height=400,
            margin=dict(l=10, r=10, t=10, b=10),
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
        )
        st.plotly_chart(fig, use_container_width=True)

    with t2:
        fig2 = go.Figure()
        for i, (bx, col) in enumerate(zip([0.14, 0.38, 0.62, 0.86], COLORS[:4])):
            r, g, b = int(col[1:3], 16), int(col[3:5], 16), int(col[5:7], 16)
            fig2.add_shape(
                type="rect",
                x0=bx - 0.1,
                y0=0.72,
                x1=bx + 0.1,
                y1=0.93,
                fillcolor=f"rgba({r},{g},{b},.12)",
                line=dict(color=col, width=1.6),
            )
            fig2.add_annotation(
                x=bx,
                y=0.825,
                text=f"<b>Tree {i+1}</b>",
                font=dict(size=10, color="#e2e8f7"),
                showarrow=False,
            )
            fig2.add_shape(
                type="line",
                x0=bx,
                y0=0.72,
                x1=0.5,
                y1=0.55,
                line=dict(color="rgba(136,146,176,.25)", width=1, dash="dot"),
            )
        fig2.add_shape(
            type="rect",
            x0=0.36,
            y0=0.43,
            x1=0.64,
            y1=0.55,
            fillcolor="rgba(251,191,36,.12)",
            line=dict(color=AMBER, width=2),
        )
        fig2.add_annotation(
            x=0.5,
            y=0.49,
            text="<b>Majority Vote</b>",
            font=dict(size=12, color="#e2e8f7"),
            showarrow=False,
        )
        fig2.add_shape(
            type="line",
            x0=0.5,
            y0=0.43,
            x1=0.5,
            y1=0.28,
            line=dict(color=TEAL, width=2.5),
        )
        fig2.add_shape(
            type="rect",
            x0=0.36,
            y0=0.14,
            x1=0.64,
            y1=0.28,
            fillcolor="rgba(0,212,170,.12)",
            line=dict(color=TEAL, width=2.5),
        )
        fig2.add_annotation(
            x=0.5,
            y=0.21,
            text="<b>Final Prediction</b>",
            font=dict(size=12, color=TEAL),
            showarrow=False,
        )
        fig2.update_layout(
            xaxis=dict(visible=False, range=[0, 1]),
            yaxis=dict(visible=False, range=[0.08, 0.98]),
            height=400,
            margin=dict(l=10, r=10, t=10, b=10),
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
        )
        st.plotly_chart(fig2, use_container_width=True)

    st.markdown(
        '<div class="sec">'
        '<span class="sec-num">03</span>'
        '<span class="sec-title">Key Hyperparameters</span>'
        "</div>",
        unsafe_allow_html=True,
    )
    hp_list = [
        (
            "max_depth",
            TEAL,
            "Controls tree depth. Deeper → complex → overfitting risk.",
        ),
        (
            "n_estimators",
            PURPLE,
            "Number of trees in Random Forest. More = stable but slower.",
        ),
        (
            "min_samples_split",
            PINK,
            "Min samples to split a node. Higher = simpler tree.",
        ),
        (
            "max_features",
            AMBER,
            "Features per split. Injects randomness into the forest.",
        ),
        (
            "min_samples_leaf",
            BLUE,
            "Min samples at leaf node — acts as regularisation.",
        ),
        ("criterion", GREEN, "Gini impurity or Information Gain (entropy) for splits."),
    ]
    cc = st.columns(3)
    for i, (name, col, desc) in enumerate(hp_list):
        r, g, b = int(col[1:3], 16), int(col[3:5], 16), int(col[5:7], 16)
        with cc[i % 3]:
            st.markdown(
                f"""
            <div style="background:var(--card);
                        border:1px solid rgba({r},{g},{b},.28);
                        border-radius:12px;padding:18px;margin-bottom:12px;">
              <div style="font-family:Space Mono,monospace;font-size:.78rem;
                          color:{col};margin-bottom:7px;">{name}</div>
              <div style="color:#7d8db5;font-size:.83rem;line-height:1.6;">{desc}</div>
            </div>
            """,
                unsafe_allow_html=True,
            )

    render_footer()


# ╔═══════════════════════════════════════════════╗
# ║  DATA EXPLORER                                ║
# ╚═══════════════════════════════════════════════╝
elif "Data" in page:
    st.markdown(
        '<div class="sec"><span class="sec-num">01</span>'
        '<span class="sec-title">Data Explorer</span></div>',
        unsafe_allow_html=True,
    )

    tabs = st.tabs(
        [
            "  Overview  ",
            "  Distributions  ",
            "  Correlations  ",
            "  Attrition Analysis  ",
        ]
    )

    # ── Overview ──────────────────────────────────
    with tabs[0]:
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Rows", f"{df_raw.shape[0]:,}")
        c2.metric("Columns", df_raw.shape[1])
        c3.metric("Numerical", len(df_raw.select_dtypes("number").columns))
        c4.metric("Categorical", len(df_raw.select_dtypes("object").columns))
        st.dataframe(df_raw.head(10), use_container_width=True, height=290)

        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**Missing Values**")
            mv = df_raw.isnull().sum()
            mv = mv[mv > 0]
            if mv.empty:
                st.success("No missing values — clean dataset!")
            else:
                st.dataframe(mv.rename("Missing"))
        with col2:
            st.markdown("**Data Types**")
            dt = df_raw.dtypes.astype(str).value_counts().reset_index()
            dt.columns = ["Type", "Count"]
            fig = px.pie(
                dt,
                values="Count",
                names="Type",
                color_discrete_sequence=COLORS,
                hole=0.55,
            )
            plotly_dark(fig, height=230)
            fig.update_traces(textfont_color="#e2e8f7")
            st.plotly_chart(fig, use_container_width=True)

        with st.expander("Statistical Summary"):
            st.dataframe(df_raw.describe().round(2), use_container_width=True)

    # ── Distributions ─────────────────────────────
    with tabs[1]:
        num_cols = df_raw.select_dtypes("number").columns.tolist()
        cs, cc2 = st.columns([1, 3])
        with cs:
            sel = st.selectbox("Feature", num_cols)
            ptype = st.radio("Chart", ["Histogram", "Box", "Violin"])
            by_att = st.checkbox("Split by Attrition", True)
        with cc2:
            kw = dict(
                color="Attrition" if by_att else None,
                color_discrete_sequence=[TEAL, PINK],
            )
            if ptype == "Histogram":
                fig = px.histogram(
                    df_raw, x=sel, nbins=30, barmode="overlay", opacity=0.75, **kw
                )
            elif ptype == "Box":
                fig = px.box(df_raw, x="Attrition" if by_att else None, y=sel, **kw)
            else:
                fig = px.violin(
                    df_raw, x="Attrition" if by_att else None, y=sel, box=True, **kw
                )
            plotly_dark(fig, f"{sel} Distribution", 380)
            st.plotly_chart(fig, use_container_width=True)

        st.markdown("**Categorical Features**")
        cat_f = [c for c in df_raw.select_dtypes("object").columns if c != "Attrition"]
        gcols = st.columns(3)
        for i, cf in enumerate(cat_f[:6]):
            vc = df_raw[cf].value_counts()
            fig = px.bar(
                x=vc.index,
                y=vc.values,
                color=vc.values,
                color_continuous_scale=[[0, "#111b33"], [1, TEAL]],
            )
            plotly_dark(fig, cf, 230)
            fig.update_layout(showlegend=False, coloraxis_showscale=False)
            fig.update_traces(marker_line_width=0)
            with gcols[i % 3]:
                st.plotly_chart(fig, use_container_width=True)

    # ── Correlations ──────────────────────────────
    with tabs[2]:
        num_enc = df_enc.select_dtypes("number")
        corr = num_enc.corr()
        cl, cr = st.columns([3, 1])
        with cr:
            nf = st.slider("Num features", 8, min(30, len(corr.columns)), 14)
            sort_c = st.selectbox(
                "Sort by", ["Attrition"] + [c for c in corr.columns if c != "Attrition"]
            )
        with cl:
            top = corr[sort_c].abs().nlargest(nf).index
            sub = corr.loc[top, top]
            fig = px.imshow(
                sub,
                color_continuous_scale=[[0, PURPLE], [0.5, "#0a0e1a"], [1, TEAL]],
                zmin=-1,
                zmax=1,
                aspect="auto",
                text_auto=".2f",
            )
            plotly_dark(fig, "Correlation Heatmap", 560)
            fig.update_traces(textfont=dict(size=8))
            st.plotly_chart(fig, use_container_width=True)

        att_c = corr["Attrition"].drop("Attrition").sort_values()
        fig2 = go.Figure(
            go.Bar(
                x=att_c.values,
                y=att_c.index,
                orientation="h",
                marker=dict(
                    color=att_c.values,
                    colorscale=[[0, PINK], [0.5, "#1a2540"], [1, TEAL]],
                    cmin=-1,
                    cmax=1,
                    line=dict(width=0),
                ),
            )
        )
        plotly_dark(fig2, "Correlation with Attrition", 500)
        st.plotly_chart(fig2, use_container_width=True)

    # ── Attrition Analysis ────────────────────────
    with tabs[3]:
        c1, c2 = st.columns(2)
        with c1:
            vc = df_raw["Attrition"].value_counts()
            fig = px.pie(
                values=vc.values,
                names=vc.index,
                color_discrete_sequence=[TEAL, PINK],
                hole=0.62,
            )
            fig.add_annotation(
                text=f"<b>{vc['Yes']/vc.sum()*100:.1f}%</b><br>Attrition",
                x=0.5,
                y=0.5,
                showarrow=False,
                font=dict(size=14, color="#e2e8f7", family="Syne"),
            )
            plotly_dark(fig, "Attrition Distribution", 320)
            st.plotly_chart(fig, use_container_width=True)
        with c2:
            da = (
                df_raw.groupby(["Department", "Attrition"]).size().reset_index(name="N")
            )
            fig = px.bar(
                da,
                x="Department",
                y="N",
                color="Attrition",
                color_discrete_sequence=[TEAL, PINK],
                barmode="group",
            )
            plotly_dark(fig, "Attrition by Department", 320)
            st.plotly_chart(fig, use_container_width=True)

        factors = [
            f
            for f in [
                "BusinessTravel",
                "OverTime",
                "MaritalStatus",
                "Gender",
                "JobLevel",
                "EducationField",
            ]
            if f in df_raw.columns
        ]
        gcols = st.columns(3)
        for i, f in enumerate(factors[:6]):
            ar = (
                df_raw.groupby(f)["Attrition"]
                .apply(lambda x: (x == "Yes").mean() * 100)
                .reset_index()
            )
            ar.columns = [f, "Rate"]
            ar = ar.sort_values("Rate")
            fig = go.Figure(
                go.Bar(
                    x=ar["Rate"],
                    y=ar[f].astype(str),
                    orientation="h",
                    marker=dict(
                        color=ar["Rate"],
                        colorscale=[[0, TEAL], [1, PINK]],
                        line=dict(width=0),
                    ),
                    text=ar["Rate"].round(1).astype(str) + "%",
                    textposition="outside",
                    textfont=dict(color="#7d8db5", size=10),
                )
            )
            plotly_dark(fig, f"Attrition Rate by {f}", 270)
            fig.update_layout(xaxis_range=[0, ar["Rate"].max() * 1.35])
            with gcols[i % 3]:
                st.plotly_chart(fig, use_container_width=True)

        fig_s = px.scatter(
            df_raw,
            x="Age",
            y="MonthlyIncome",
            color="Attrition",
            color_discrete_sequence=[TEAL, PINK],
            opacity=0.65,
            hover_data=["JobRole", "Department"],
        )
        plotly_dark(fig_s, "Age vs Monthly Income — coloured by Attrition", 420)
        st.plotly_chart(fig_s, use_container_width=True)

    render_footer()


# ╔═══════════════════════════════════════════════╗
# ║  TRAIN MODELS                                 ║
# ╚═══════════════════════════════════════════════╝
elif "Train" in page:
    st.markdown(
        '<div class="sec"><span class="sec-num">01</span>'
        '<span class="sec-title">Train &amp; Evaluate Models</span></div>',
        unsafe_allow_html=True,
    )

    left, right = st.columns([1, 2])

    with left:
        model_type = st.selectbox("Model", ["Decision Tree", "Random Forest"])
        st.markdown("**Hyperparameters**")

        if model_type == "Decision Tree":
            crit = st.selectbox("Criterion", ["gini", "entropy"])
            md = st.slider("max_depth", 1, 30, 5)
            mss = st.slider("min_samples_split", 2, 50, 10)
            msl = st.slider("min_samples_leaf", 1, 30, 5)
            mf = st.selectbox("max_features", ["None", "sqrt", "log2"])
            params = dict(
                criterion=crit,
                max_depth=md,
                min_samples_split=mss,
                min_samples_leaf=msl,
                max_features=None if mf == "None" else mf,
            )
        else:
            ne = st.slider("n_estimators", 10, 500, 100, 10)
            crit = st.selectbox("Criterion", ["gini", "entropy"])
            md = st.slider("max_depth", 1, 30, 10)
            mss = st.slider("min_samples_split", 2, 50, 5)
            msl = st.slider("min_samples_leaf", 1, 30, 2)
            mf = st.selectbox("max_features", ["sqrt", "log2", "None"])
            params = dict(
                n_estimators=ne,
                criterion=crit,
                max_depth=md,
                min_samples_split=mss,
                min_samples_leaf=msl,
                max_features=None if mf == "None" else mf,
            )

        balance = st.checkbox("Balance Classes (upsample minority)", True)
        test_sz = st.slider("Test size %", 10, 40, 20) / 100
        train_btn = st.button("Train Model", use_container_width=True)

    with right:
        if train_btn:
            with st.spinner(f"Training {model_type}..."):
                clf, Xtr, Xte, ytr, yte, feats = do_train(
                    df_enc, model_type, params, balance, test_sz
                )
                m = eval_model(clf, Xtr, ytr, Xte, yte)
            st.session_state.update(
                dict(
                    clf=clf,
                    metrics=m,
                    feat_names=feats,
                    model_type=model_type,
                    trained=True,
                )
            )
            st.success(
                "Model trained! Head to **Predict Attrition** — inference is instant."
            )

        if st.session_state.get("trained"):
            clf = st.session_state["clf"]
            m = st.session_state["metrics"]
            feats = st.session_state["feat_names"]

            st.markdown(
                f"""
            <div class="mgrid">
              <div class="mcard">
                <div class="mval">{m["test_acc"]*100:.1f}%</div>
                <div class="mlabel">Test Accuracy</div>
              </div>
              <div class="mcard">
                <div class="mval">{m["train_acc"]*100:.1f}%</div>
                <div class="mlabel">Train Accuracy</div>
              </div>
              <div class="mcard">
                <div class="mval">{m["f1"]:.3f}</div>
                <div class="mlabel">F1 Score</div>
              </div>
              <div class="mcard">
                <div class="mval">{m["roc_auc"]:.3f}</div>
                <div class="mlabel">ROC AUC</div>
              </div>
            </div>
            """,
                unsafe_allow_html=True,
            )

            t1, t2, t3, t4 = st.tabs(
                [
                    "  Confusion Matrix  ",
                    "  ROC Curve  ",
                    "  Feature Importance  ",
                    "  Report  ",
                ]
            )

            with t1:
                fig = px.imshow(
                    m["cm"],
                    text_auto=True,
                    aspect="auto",
                    x=["Stay", "Leave"],
                    y=["Stay", "Leave"],
                    color_continuous_scale=[[0, "#080d1a"], [0.5, PURPLE], [1, TEAL]],
                )
                plotly_dark(fig, "Confusion Matrix", 350)
                fig.update_traces(textfont_size=18)
                st.plotly_chart(fig, use_container_width=True)

            with t2:
                fpr, tpr, _ = roc_curve(m["y_test"], m["y_proba"])
                fig = go.Figure()
                fig.add_trace(
                    go.Scatter(
                        x=[0, 1],
                        y=[0, 1],
                        mode="lines",
                        line=dict(dash="dash", color="rgba(136,146,176,.4)", width=1),
                        name="Baseline",
                    )
                )
                fig.add_trace(
                    go.Scatter(
                        x=fpr,
                        y=tpr,
                        mode="lines",
                        line=dict(color=TEAL, width=2.5),
                        fill="tozeroy",
                        fillcolor="rgba(0,212,170,.07)",
                        name=f"AUC = {m['roc_auc']:.3f}",
                    )
                )
                plotly_dark(fig, "ROC Curve", 400)
                fig.update_layout(xaxis_title="FPR", yaxis_title="TPR")
                st.plotly_chart(fig, use_container_width=True)

            with t3:
                if hasattr(clf, "feature_importances_"):
                    fi = (
                        pd.DataFrame(
                            {"Feature": feats, "Importance": clf.feature_importances_}
                        )
                        .sort_values("Importance")
                        .tail(20)
                    )
                    fig = go.Figure(
                        go.Bar(
                            x=fi["Importance"],
                            y=fi["Feature"],
                            orientation="h",
                            marker=dict(
                                color=fi["Importance"],
                                colorscale=[[0, PURPLE], [1, TEAL]],
                                line=dict(width=0),
                            ),
                        )
                    )
                    plotly_dark(fig, "Top 20 Feature Importances", 520)
                    st.plotly_chart(fig, use_container_width=True)

            with t4:
                rpt = m["report"]
                rep_df = pd.DataFrame(
                    {
                        "Class": ["No Attrition", "Attrition"],
                        "Precision": [rpt["0"]["precision"], rpt["1"]["precision"]],
                        "Recall": [rpt["0"]["recall"], rpt["1"]["recall"]],
                        "F1": [rpt["0"]["f1-score"], rpt["1"]["f1-score"]],
                        "Support": [int(rpt["0"]["support"]), int(rpt["1"]["support"])],
                    }
                )
                st.dataframe(
                    rep_df.style.format(
                        {"Precision": "{:.3f}", "Recall": "{:.3f}", "F1": "{:.3f}"}
                    ),
                    use_container_width=True,
                )

                model_bytes = pickle.dumps(clf)
                b64 = base64.b64encode(model_bytes).decode()
                st.markdown(
                    f'<a href="data:application/octet-stream;base64,{b64}" '
                    f'download="model.pkl" '
                    f'style="display:inline-flex;align-items:center;gap:8px;'
                    f"background:rgba(0,212,170,.08);"
                    f"border:1px solid rgba(0,212,170,.28);"
                    f"color:#00d4aa;padding:10px 22px;border-radius:10px;"
                    f"font-family:Space Mono,monospace;font-size:.78rem;"
                    f'text-decoration:none;margin-top:14px;">'
                    f"&#11015;&#65039; Download Trained Model (.pkl)</a>",
                    unsafe_allow_html=True,
                )
        else:
            st.markdown(
                """
            <div style="background:var(--card);border:1px solid var(--border);
                        border-radius:16px;padding:60px;text-align:center;
                        color:#7d8db5;">
              <div style="font-size:3rem;margin-bottom:14px;">&#129302;</div>
              <div style="font-family:Syne,sans-serif;font-size:1.2rem;
                          color:#e2e8f7;margin-bottom:8px;">
                Configure &amp; Train Your Model
              </div>
              <div>Set hyperparameters on the left, then click
                <strong style="color:#00d4aa;">Train Model</strong>
              </div>
            </div>
            """,
                unsafe_allow_html=True,
            )

    render_footer()


# ╔═══════════════════════════════════════════════╗
# ║  PREDICT ATTRITION  — always instant          ║
# ╚═══════════════════════════════════════════════╝
elif "Predict" in page:
    st.markdown(
        '<div class="sec"><span class="sec-num">01</span>'
        '<span class="sec-title">Predict Employee Attrition</span></div>',
        unsafe_allow_html=True,
    )

    # ── always use pre-trained model; override if user trained one ──
    clf = st.session_state.get("clf", _default_clf)
    feats = st.session_state.get("feat_names", _default_feats)
    le_active = st.session_state.get("le_dict", _default_le)

    model_source = (
        "your custom-trained model"
        if st.session_state.get("trained")
        else "default pre-trained Random Forest"
    )

    st.markdown(
        f"""
    <div class="icard icard-t">
      <div style="display:flex;align-items:center;gap:12px;flex-wrap:wrap;">
        <span class="instant-badge">&#9889; Instant Inference</span>
        <span style="color:var(--muted);font-size:.85rem;">
          Using <strong style="color:#e2e8f7;">{model_source}</strong>
          — no waiting, predictions are ready immediately.
        </span>
      </div>
    </div>
    """,
        unsafe_allow_html=True,
    )

    c1, c2, c3 = st.columns(3)
    inp = {}

    with c1:
        st.markdown("**&#128100; Personal Info**")
        inp["Age"] = st.number_input("Age", 18, 60, 35)
        inp["Gender"] = st.selectbox("Gender", ["Male", "Female"])
        inp["MaritalStatus"] = st.selectbox(
            "Marital Status", ["Single", "Married", "Divorced"]
        )
        inp["Education"] = st.selectbox(
            "Education",
            [1, 2, 3, 4, 5],
            format_func=lambda x: {
                1: "Below College",
                2: "College",
                3: "Bachelor",
                4: "Master",
                5: "Doctor",
            }[x],
        )
        inp["EducationField"] = st.selectbox(
            "Education Field",
            [
                "Life Sciences",
                "Medical",
                "Marketing",
                "Technical Degree",
                "Human Resources",
                "Other",
            ],
        )
        inp["DistanceFromHome"] = st.slider("Distance From Home (km)", 1, 29, 5)

    with c2:
        st.markdown("**&#128188; Job Info**")
        inp["Department"] = st.selectbox(
            "Department", ["Research & Development", "Sales", "Human Resources"]
        )
        inp["JobRole"] = st.selectbox(
            "Job Role",
            [
                "Sales Executive",
                "Research Scientist",
                "Laboratory Technician",
                "Manufacturing Director",
                "Healthcare Representative",
                "Manager",
                "Sales Representative",
                "Research Director",
                "Human Resources",
            ],
        )
        inp["JobLevel"] = st.selectbox("Job Level", [1, 2, 3, 4, 5])
        inp["BusinessTravel"] = st.selectbox(
            "Business Travel", ["Travel_Rarely", "Travel_Frequently", "Non-Travel"]
        )
        inp["OverTime"] = st.selectbox("Overtime", ["No", "Yes"])
        inp["MonthlyIncome"] = st.number_input("Monthly Income ($)", 1009, 19999, 5000)
        inp["PercentSalaryHike"] = st.slider("% Salary Hike", 11, 25, 14)
        inp["StockOptionLevel"] = st.selectbox("Stock Option Level", [0, 1, 2, 3])

    with c3:
        st.markdown("**&#128202; Satisfaction &amp; Tenure**")
        inp["JobSatisfaction"] = st.selectbox(
            "Job Satisfaction",
            [1, 2, 3, 4],
            format_func=lambda x: {1: "Low", 2: "Medium", 3: "High", 4: "Very High"}[x],
        )
        inp["EnvironmentSatisfaction"] = st.selectbox(
            "Environment Satisfaction",
            [1, 2, 3, 4],
            format_func=lambda x: {1: "Low", 2: "Medium", 3: "High", 4: "Very High"}[x],
        )
        inp["WorkLifeBalance"] = st.selectbox(
            "Work-Life Balance",
            [1, 2, 3, 4],
            format_func=lambda x: {1: "Bad", 2: "Good", 3: "Better", 4: "Best"}[x],
        )
        inp["JobInvolvement"] = st.selectbox("Job Involvement", [1, 2, 3, 4])
        inp["RelationshipSatisfaction"] = st.selectbox(
            "Relationship Satisfaction", [1, 2, 3, 4]
        )
        inp["TotalWorkingYears"] = st.number_input("Total Working Years", 0, 40, 10)
        inp["YearsAtCompany"] = st.number_input("Years at Company", 0, 40, 5)
        inp["YearsInCurrentRole"] = st.number_input("Years in Current Role", 0, 18, 3)
        inp["YearsSinceLastPromotion"] = st.number_input(
            "Years Since Promotion", 0, 15, 1
        )
        inp["YearsWithCurrManager"] = st.number_input("Years With Manager", 0, 17, 3)

    # defaults for less-critical fields
    for k, v in dict(
        DailyRate=800,
        HourlyRate=65,
        MonthlyRate=14000,
        NumCompaniesWorked=3,
        TrainingTimesLastYear=3,
        PerformanceRating=3,
    ).items():
        inp[k] = v

    if st.button("&#128302; Predict Attrition Risk"):
        row = {}
        for f in feats:
            v = inp.get(f, 0)
            if isinstance(v, str) and f in le_active:
                try:
                    v = int(le_active[f].transform([v])[0])
                except Exception:
                    v = 0
            row[f] = [v]

        Xp = pd.DataFrame(row)[feats]
        pred = clf.predict(Xp)[0]
        prob = clf.predict_proba(Xp)[0]
        risk = prob[1] * 100

        # ── risk label & color — based purely on risk %, not pred ──
        if risk >= 70:
            risk_label = "VERY HIGH"
            risk_color = "#ef4444"
            risk_icon = "🔴"
            card_bg = "linear-gradient(135deg,rgba(239,68,68,.13),rgba(239,68,68,.06))"
            card_border = "rgba(239,68,68,.45)"
            rec = (
                "Action needed: review compensation, workload, and "
                "career path immediately."
            )
        elif risk >= 50:
            risk_label = "HIGH"
            risk_color = "#f97316"
            risk_icon = "🟠"
            card_bg = (
                "linear-gradient(135deg,rgba(249,115,22,.13),rgba(249,115,22,.06))"
            )
            card_border = "rgba(249,115,22,.45)"
            rec = (
                "Consider: career development conversation, "
                "compensation review, or role rotation."
            )
        elif risk >= 30:
            risk_label = "MODERATE"
            risk_color = "#fbbf24"
            risk_icon = "🟡"
            card_bg = (
                "linear-gradient(135deg,rgba(251,191,36,.13),rgba(251,191,36,.06))"
            )
            card_border = "rgba(251,191,36,.45)"
            rec = (
                "Monitor closely. Keep engagement high with "
                "recognition and growth opportunities."
            )
        else:
            risk_label = "LOW"
            risk_color = "#00d4aa"
            risk_icon = "🟢"
            card_bg = "linear-gradient(135deg,rgba(0,212,170,.13),rgba(0,212,170,.06))"
            card_border = "rgba(0,212,170,.45)"
            rec = (
                "Employee is stable. Maintain current engagement "
                "and recognition programs."
            )

        stay_pct = 100 - risk

        st.markdown(
            f"""
        <div style="background:{card_bg};
                    border:2px solid {card_border};
                    border-radius:20px 20px 0 0;
                    padding:32px;text-align:center;
                    margin-bottom:0;">
          <div style="font-family:Syne,sans-serif;font-size:2rem;font-weight:800;
                      color:{risk_color};">
            {risk_icon} {risk_label} ATTRITION RISK
          </div>
          <div style="font-family:Syne,sans-serif;font-size:3.8rem;font-weight:800;
                      color:{risk_color};margin:10px 0;line-height:1;">
            {risk:.1f}%
          </div>
          <div style="color:#7d8db5;font-size:.95rem;line-height:1.7;margin-top:6px;">
            Attrition risk score &nbsp;|&nbsp;
            Stay probability: <strong style="color:#e2e8f7;">{stay_pct:.1f}%</strong><br>
            {rec}
          </div>
        </div>
        """,
            unsafe_allow_html=True,
        )

        # ── Gauge — seamlessly attached below the card ──
        gauge_col = PINK if risk >= 60 else AMBER if risk >= 30 else TEAL

        fig = go.Figure(
            go.Indicator(
                mode="gauge+number",
                value=risk,
                domain={"x": [0, 1], "y": [0, 1]},
                title={
                    "text": "Attrition Risk Score (%)",
                    "font": {"color": "#7d8db5", "family": "Syne", "size": 13},
                },
                number={
                    "suffix": "%",
                    "font": {"color": "#e2e8f7", "size": 28, "family": "Syne"},
                    "valueformat": ".1f",
                },
                gauge={
                    "axis": {
                        "range": [0, 100],
                        "tickcolor": "#7d8db5",
                        "tickwidth": 1,
                        "tickfont": {"size": 11, "color": "#7d8db5"},
                    },
                    "bar": {"color": gauge_col, "thickness": 0.28},
                    "bgcolor": "#111b33",
                    "borderwidth": 0,
                    "steps": [
                        {"range": [0, 30], "color": "rgba(0,212,170,.10)"},
                        {"range": [30, 60], "color": "rgba(251,191,36,.10)"},
                        {"range": [60, 100], "color": "rgba(244,114,182,.10)"},
                    ],
                    "threshold": {
                        "line": {"color": AMBER, "width": 2},
                        "value": 50,
                    },
                },
            )
        )
        fig.update_layout(
            height=300,
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            margin=dict(l=40, r=40, t=60, b=10),
            font=dict(color="#7d8db5"),
        )

        # wrap in a div that connects visually to the card above
        st.markdown(
            """
        <div style="background:var(--card);
                    border:2px solid var(--border);
                    border-top:none;
                    border-bottom-left-radius:20px;
                    border-bottom-right-radius:20px;
                    padding:8px 0 16px;
                    margin-bottom:24px;">
        """,
            unsafe_allow_html=True,
        )
        st.plotly_chart(fig, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

        # ── quick interpretation legend ─────────────────
        st.markdown(
            """
        <div style="display:flex;gap:10px;flex-wrap:wrap;margin-top:4px;">
          <div style="display:flex;align-items:center;gap:6px;font-size:.8rem;color:#7d8db5;">
            <span style="width:12px;height:12px;border-radius:50%;
                         background:#00d4aa;display:inline-block;"></span>Low  0–30%
          </div>
          <div style="display:flex;align-items:center;gap:6px;font-size:.8rem;color:#7d8db5;">
            <span style="width:12px;height:12px;border-radius:50%;
                         background:#fbbf24;display:inline-block;"></span>Moderate  30–60%
          </div>
          <div style="display:flex;align-items:center;gap:6px;font-size:.8rem;color:#7d8db5;">
            <span style="width:12px;height:12px;border-radius:50%;
                         background:#f472b6;display:inline-block;"></span>High  60–100%
          </div>
        </div>
        """,
            unsafe_allow_html=True,
        )

    render_footer()


# ╔═══════════════════════════════════════════════╗
# ║  MODEL COMPARISON                             ║
# ╚═══════════════════════════════════════════════╝
elif "Comparison" in page:
    st.markdown(
        '<div class="sec"><span class="sec-num">01</span>'
        '<span class="sec-title">Model Comparison</span></div>',
        unsafe_allow_html=True,
    )
    st.info(
        "Benchmark results from the IBM HR notebook. "
        "Train your own model in **Train Models** to add it here."
    )

    res = {
        "Model": ["DT Default", "DT Tuned", "RF Default", "RF Tuned"],
        "Train Acc": [1.00, 0.94, 1.00, 0.98],
        "Test Acc": [0.82, 0.87, 0.86, 0.89],
        "F1 Score": [0.41, 0.55, 0.48, 0.60],
        "ROC AUC": [0.63, 0.74, 0.72, 0.83],
    }
    if st.session_state.get("trained"):
        m = st.session_state["metrics"]
        mt = st.session_state["model_type"]
        res["Model"].append(f"{mt} (Yours)")
        res["Train Acc"].append(round(m["train_acc"], 3))
        res["Test Acc"].append(round(m["test_acc"], 3))
        res["F1 Score"].append(round(m["f1"], 3))
        res["ROC AUC"].append(round(m["roc_auc"], 3))

    df_res = pd.DataFrame(res)
    metrics_cols = ["Train Acc", "Test Acc", "F1 Score", "ROC AUC"]

    st.dataframe(
        df_res.style.highlight_max(
            subset=["Test Acc", "F1 Score", "ROC AUC"], color="rgba(0,212,170,.25)"
        ),
        use_container_width=True,
    )

    # Radar
    fig_r = go.Figure()
    for i, row in df_res.iterrows():
        vals = [row[c] for c in metrics_cols] + [row[metrics_cols[0]]]
        col = COLORS[i % len(COLORS)]
        r2, g2, b2 = int(col[1:3], 16), int(col[3:5], 16), int(col[5:7], 16)
        fig_r.add_trace(
            go.Scatterpolar(
                r=vals,
                theta=metrics_cols + [metrics_cols[0]],
                fill="toself",
                name=row["Model"],
                line=dict(color=col, width=2),
                fillcolor=f"rgba({r2},{g2},{b2},.08)",
            )
        )
    fig_r.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 1],
                gridcolor="rgba(255,255,255,0.05)",
                tickfont=dict(color="#7d8db5"),
            ),
            angularaxis=dict(
                gridcolor="rgba(255,255,255,0.05)", tickfont=dict(color="#7d8db5")
            ),
            bgcolor="rgba(0,0,0,0)",
        ),
        showlegend=True,
        height=480,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#7d8db5", family="DM Sans"),
        legend=dict(font=dict(color="#7d8db5")),
        title=dict(
            text="Performance Radar", font=dict(family="Syne", color="#e2e8f7", size=14)
        ),
    )
    st.plotly_chart(fig_r, use_container_width=True)

    # Grouped bar
    df_m = df_res.melt(
        id_vars="Model", value_vars=metrics_cols, var_name="Metric", value_name="Score"
    )
    fig_b = px.bar(
        df_m,
        x="Metric",
        y="Score",
        color="Model",
        barmode="group",
        color_discrete_sequence=COLORS,
    )
    plotly_dark(fig_b, "Metrics Comparison", 420)
    st.plotly_chart(fig_b, use_container_width=True)

    # Overfitting
    st.markdown(
        '<div class="sec"><span class="sec-num">02</span>'
        '<span class="sec-title">Overfitting Analysis</span></div>',
        unsafe_allow_html=True,
    )
    df_res["Gap"] = df_res["Train Acc"] - df_res["Test Acc"]
    fig_g = px.bar(
        df_res,
        x="Model",
        y="Gap",
        color="Gap",
        color_continuous_scale=[[0, TEAL], [0.5, AMBER], [1, PINK]],
    )
    plotly_dark(fig_g, "Train–Test Accuracy Gap (lower = less overfitting)", 360)
    fig_g.add_hline(
        y=0.05,
        line_dash="dash",
        line_color=AMBER,
        annotation_text="5% threshold",
        annotation_font_color=AMBER,
    )
    st.plotly_chart(fig_g, use_container_width=True)

    render_footer()


# ╔═══════════════════════════════════════════════╗
# ║  ABOUT                                        ║
# ╚═══════════════════════════════════════════════╝
elif "About" in page:
    st.markdown(
        """
    <div class="hero" style="text-align:center;">
      <div style="font-size:4rem;margin-bottom:14px;">&#128105;&#8205;&#128187;</div>
      <div class="hero-title" style="font-size:2.4rem;">Hafsa Ibrahim</div>
      <div class="hero-sub">AI &amp; Machine Intelligence Engineer</div>
      <div style="margin-top:22px;display:flex;justify-content:center;
                  gap:16px;flex-wrap:wrap;">
        <a href="https://www.linkedin.com/in/hafsa-ibrahim-ai-mi/"
           target="_blank" class="flink">&#128279; LinkedIn Profile</a>
        <a href="https://github.com/HafsaIbrahim5"
           target="_blank" class="flink">&#128025; GitHub Profile</a>
      </div>
    </div>
    """,
        unsafe_allow_html=True,
    )

    c1, c2 = st.columns(2)
    with c1:
        st.markdown(
            """
        <div class="icard icard-t">
          <h3 style="font-family:Syne,sans-serif;color:#00d4aa;margin-top:0;">
            About This Project
          </h3>
          <p style="color:#7d8db5;line-height:1.9;">
            End-to-end <strong style="color:#e2e8f7;">Employee Attrition Prediction</strong>
            pipeline using the IBM HR Analytics dataset.
            Applies Decision Trees &amp; Random Forests to predict which employees
            are most likely to resign, covering EDA, preprocessing, model training,
            evaluation, hyperparameter tuning, and deployment.
          </p>
          <p style="color:#7d8db5;line-height:1.9;margin-bottom:0;">
            The Predict page uses a
            <strong style="color:#00d4aa;">pre-loaded model</strong> so inference
            is always instant — no waiting for clients or end users.
          </p>
        </div>
        """,
            unsafe_allow_html=True,
        )
    with c2:
        stack = [
            ("Python 3.10+", TEAL),
            ("Streamlit", PURPLE),
            ("scikit-learn", PINK),
            ("Plotly", AMBER),
            ("Pandas", BLUE),
            ("NumPy", GREEN),
        ]
        badges = (
            "<div class='icard icard-p'>"
            "<h3 style='font-family:Syne,sans-serif;color:#7c3aed;"
            "margin-top:0;'>Tech Stack</h3>"
            "<div style='display:grid;grid-template-columns:1fr 1fr;gap:9px;'>"
        )
        for name, col in stack:
            r, g, b = int(col[1:3], 16), int(col[3:5], 16), int(col[5:7], 16)
            badges += (
                f"<div style='background:rgba({r},{g},{b},.1);"
                f"border:1px solid rgba({r},{g},{b},.28);"
                f"border-radius:8px;padding:8px 12px;"
                f"font-family:Space Mono,monospace;font-size:.72rem;"
                f"color:{col};'>{name}</div>"
            )
        badges += "</div></div>"
        st.markdown(badges, unsafe_allow_html=True)

    st.markdown(
        '<div class="sec"><span class="sec-num">02</span>'
        '<span class="sec-title">Project Features</span></div>',
        unsafe_allow_html=True,
    )

    feats_list = [
        ("Interactive EDA", "Rich visualisations of the IBM HR dataset"),
        ("Correlation Analysis", "Heatmaps & target correlation bar charts"),
        (
            "Dual Model Training",
            "Decision Tree & Random Forest with custom hyperparams",
        ),
        ("Class Balancing", "Upsample minority class to handle class imbalance"),
        ("Full Evaluation Suite", "Accuracy, F1, AUC, confusion matrix, ROC curve"),
        ("Feature Importance", "Ranked bar chart of top attrition predictors"),
        ("Instant Predictions", "Pre-trained model — zero wait time for the end user"),
        ("Model Comparison", "Radar + bar chart + overfitting analysis"),
        ("Model Export", "Download trained model as .pkl file"),
        ("Professional Dark UI", "Gradient design with Syne + Space Mono typography"),
    ]
    icons = ["📊", "🔥", "🤖", "⚖️", "📈", "🌟", "⚡", "📉", "⬇️", "🎨"]
    fcols = st.columns(2)
    for i, (title, desc) in enumerate(feats_list):
        with fcols[i % 2]:
            st.markdown(
                f"""
            <div style="display:flex;align-items:flex-start;gap:13px;
                        background:var(--card);border:1px solid var(--border);
                        border-radius:12px;padding:16px;margin-bottom:10px;">
              <div style="font-size:1.4rem;flex-shrink:0;">{icons[i]}</div>
              <div>
                <div style="font-family:Syne,sans-serif;font-weight:600;
                             color:#e2e8f7;margin-bottom:3px;">{title}</div>
                <div style="color:#7d8db5;font-size:.83rem;">{desc}</div>
              </div>
            </div>
            """,
                unsafe_allow_html=True,
            )

    st.markdown(
        """
    <div class="icard" style="margin-top:20px;">
      <h4 style="font-family:Syne,sans-serif;color:#e2e8f7;margin-top:0;">
        References
      </h4>
      <ul style="color:#7d8db5;line-height:2.1;">
        <li>Kaggle:
          <a href="https://www.kaggle.com/code/faressayah/decision-trees-random-forest-for-beginners"
             style="color:#00d4aa;" target="_blank">
            Decision Trees &amp; Random Forest for Beginners
          </a>
        </li>
        <li>IBM HR Analytics Employee Attrition Dataset</li>
        <li>scikit-learn —
          <a href="https://scikit-learn.org" style="color:#00d4aa;" target="_blank">
            sklearn.org
          </a>
        </li>
        <li>Towards Data Science — Hyperparameter Tuning the Random Forest</li>
      </ul>
    </div>
    """,
        unsafe_allow_html=True,
    )

    render_footer()
