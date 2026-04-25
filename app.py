import streamlit as st
import pickle
import numpy as np

# ---------------- PAGE ----------------
st.set_page_config(
    page_title="AI Customer Churn Predictor",
    page_icon="🚀",
    layout="wide"
)

# ---------------- LOAD MODEL ----------------
model = pickle.load(open("model.pkl","rb"))

# ---------------- STYLING ----------------
st.markdown("""
<style>
.main{
padding-top:1rem;
}

.hero{
background: linear-gradient(90deg,#2563eb,#06b6d4);
padding:30px;
border-radius:25px;
text-align:center;
color:white;
margin-bottom:25px;
box-shadow:0 8px 25px rgba(0,0,0,0.15);
}

.hero h1{
font-size:50px;
margin-bottom:10px;
}

.hero p{
font-size:20px;
}

.block-container{
padding-top:1rem;
}

[data-testid="stMetric"]{
background:#f8fafc;
padding:20px;
border-radius:20px;
box-shadow:0 4px 12px rgba(0,0,0,.08);
}
</style>
""", unsafe_allow_html=True)

# ---------------- HERO ----------------
st.markdown("""
<div class='hero'>
<h1>🚀 Customer Churn Prediction Dashboard</h1>
<p>AI-powered churn prediction using Random Forest</p>
</div>
""", unsafe_allow_html=True)

# ---------------- TOP METRICS ----------------
m1,m2,m3=st.columns(3)

m1.metric("Model Accuracy","87.4%")
m2.metric("Algorithm","Random Forest")
m3.metric("Prediction Type","Binary")

st.divider()

# ---------------- INPUTS ----------------
left,right=st.columns(2)

with left:
    st.subheader("Customer Profile")

    age=st.number_input(
        "Age",
        18,80,30
    )

    frequent=st.selectbox(
        "Frequent Flyer",
        ["No","Yes","No Record"]
    )

    income=st.selectbox(
        "Annual Income Class",
        [
        "Low Income",
        "Middle Income",
        "High Income"
        ]
    )

with right:
    st.subheader("Service Details")

    services=st.slider(
        "Services Opted",
        1,6,2
    )

    social=st.selectbox(
        "Account Synced",
        ["No","Yes"]
    )

    hotel=st.selectbox(
        "Booked Hotel",
        ["No","Yes"]
    )

# ---------------- ENCODING ----------------
ff_map={
"No":0,
"Yes":1,
"No Record":2
}

income_map={
"Low Income":0,
"Middle Income":1,
"High Income":2
}

yesno={
"No":0,
"Yes":1
}

st.write("")
predict=st.button(
"🔍 Run Churn Prediction",
use_container_width=True
)

# ---------------- PREDICTION ----------------
if predict:

    data=np.array([[
        age,
        ff_map[frequent],
        income_map[income],
        services,
        yesno[social],
        yesno[hotel]
    ]])

    pred=model.predict(data)[0]
    prob=model.predict_proba(data)[0][1]

    st.divider()
    st.subheader("Prediction Results")

    c1,c2=st.columns(2)

    with c1:
        st.metric(
        "Churn Risk",
        f"{prob:.2%}"
        )

    with c2:
        st.metric(
        "Retention Probability",
        f"{1-prob:.2%}"
        )

    st.progress(float(prob))

    if pred==1:
        st.error(
        "⚠ HIGH RISK: Customer likely to churn"
        )
    else:
        st.success(
        "✅ LOW RISK: Customer likely to stay"
        )

    if prob<0.30:
        st.success("Risk Level: Low")
    elif prob<0.70:
        st.warning("Risk Level: Medium")
    else:
        st.error("Risk Level: High")

st.divider()
st.caption("Built with Python • Random Forest • Streamlit")
