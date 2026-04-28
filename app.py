import streamlit as st
import pickle
import numpy as np
import pandas as pd

st.set_page_config(
page_title="AI Churn Analytics Dashboard",
page_icon="📈",
layout="wide"
)

model=pickle.load(open("model.pkl","rb"))

st.markdown("""
<style>
.main{
padding-top:0rem;
background:#0f172a;
}

/* Hero Banner */
.hero{
background:linear-gradient(90deg,#1d4ed8,#06b6d4);
padding:35px;
border-radius:30px;
text-align:center;
color:white;
box-shadow:0px 8px 20px rgba(0,0,0,.15);
margin-bottom:25px;
}

.hero h1{
font-size:52px;
margin-bottom:8px;
}

.hero p{
font-size:20px;
}

/* Metric Cards */
[data-testid="stMetric"]{
background:#ffffff;
padding:20px;
border-radius:20px;
box-shadow:0px 4px 15px rgba(0,0,0,.12);
}

/* Metric labels */
[data-testid="stMetricLabel"]{
color:#111827 !important;
font-size:18px !important;
font-weight:700 !important;
}

/* Metric values */
[data-testid="stMetricValue"]{
color:#111827 !important;
font-size:34px !important;
font-weight:800 !important;
}

/* Input containers */
[data-testid="stSelectbox"],
[data-testid="stNumberInput"],
[data-testid="stSlider"]{
border-radius:15px;
}

/* Button */
.stButton>button{
background:linear-gradient(90deg,#2563eb,#06b6d4);
color:white;
border:none;
padding:14px;
font-size:18px;
font-weight:bold;
border-radius:14px;
}

.stButton>button:hover{
transform:scale(1.02);
}

/* Snapshot box */
.info-box{
background:#1e3a5f;
padding:20px;
border-radius:18px;
color:white;
font-size:18px;
}
</style>
""",unsafe_allow_html=True)

st.markdown("""
<div class='hero'>
<h1>🚀 AI Customer Churn Analytics</h1>
<p>Predict churn risk using Random Forest Machine Learning</p>
</div>
""",unsafe_allow_html=True)

a,b,c,d=st.columns(4)

a.metric("Model Accuracy","87.4%")
b.metric("Algorithm","Random Forest")
c.metric("Features","6")
d.metric("Task","Classification")

st.divider()

left,right=st.columns([2,1])

with left:

    st.subheader("Customer Input")

    col1,col2=st.columns(2)

    with col1:
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

    with col2:
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

with right:
    st.subheader("Feature Snapshot")

    st.markdown(f"""
<div class='info-box'>
<p>Age: {age}</p>
<p>Services: {services}</p>
<p>Income: {income}</p>
</div>
""",unsafe_allow_html=True)

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
"🔍 Run Prediction",
use_container_width=True
)

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
    st.subheader("Prediction Dashboard")

    p1,p2,p3=st.columns(3)

    p1.metric(
    "Churn Risk",
    f"{prob:.2%}"
    )

    p2.metric(
    "Retention",
    f"{1-prob:.2%}"
    )

    if prob<0.30:
        risk="Low"
    elif prob<0.70:
        risk="Medium"
    else:
        risk="High"

    p3.metric(
    "Risk Level",
    risk
    )

    st.progress(float(prob))

    if pred==1:
        st.error(
        "⚠ Customer likely to churn"
        )
    else:
        st.success(
        "✅ Customer likely to stay"
        )

    st.subheader("Feature Influence Graph")

    import pandas as pd

features=[
"Age",
"FrequentFlyer",
"IncomeClass",
"ServicesOpted",
"AccountSynced",
"BookedHotel"
]

importance=model.feature_importances_

chart_data=pd.DataFrame(
{
"Feature":features,
"Importance":importance
}
).set_index("Feature")

st.subheader("Actual Feature Importance")
st.bar_chart(chart_data)

st.divider()
st.caption("Built using Python • Random Forest • Streamlit")
