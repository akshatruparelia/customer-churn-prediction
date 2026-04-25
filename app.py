import streamlit as st
import pickle
import numpy as np

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="Customer Churn Predictor",
    page_icon="📊",
    layout="centered"
)

# ---------------- LOAD MODEL ----------------
model = pickle.load(open("model.pkl","rb"))

# ---------------- CUSTOM CSS ----------------
st.markdown("""
<style>
.main {
    padding-top: 1rem;
}
.big-title {
    text-align:center;
    padding:20px;
    border-radius:20px;
    background: linear-gradient(90deg,#4f46e5,#06b6d4);
    color:white;
    font-size:40px;
    font-weight:bold;
}
.subtitle{
    text-align:center;
    font-size:18px;
    color:#666;
    margin-bottom:20px;
}
.pred-box{
    padding:20px;
    border-radius:16px;
    font-size:24px;
    text-align:center;
    font-weight:bold;
}
</style>
""", unsafe_allow_html=True)

# ---------------- HEADER ----------------
st.markdown(
'<div class="big-title">Customer Churn Prediction Dashboard</div>',
unsafe_allow_html=True
)

st.markdown(
'<div class="subtitle">Predict whether a travel customer is likely to churn</div>',
unsafe_allow_html=True
)

# ---------------- INPUTS ----------------
col1,col2 = st.columns(2)

with col1:
    age = st.number_input("Age",18,80,30)

    frequent = st.selectbox(
        "Frequent Flyer",
        ["No","Yes","No Record"]
    )

    income = st.selectbox(
        "Annual Income Class",
        ["Low Income","Middle Income","High Income"]
    )

with col2:
    services = st.slider(
        "Services Opted",
        1,6,2
    )

    social = st.selectbox(
        "Account Synced",
        ["No","Yes"]
    )

    hotel = st.selectbox(
        "Booked Hotel",
        ["No","Yes"]
    )

# ---------------- ENCODING ----------------
ff_map = {
"No":0,
"Yes":1,
"No Record":2
}

income_map = {
"Low Income":0,
"Middle Income":1,
"High Income":2
}

yesno = {
"No":0,
"Yes":1
}

# ---------------- BUTTON ----------------
if st.button("🔍 Predict Churn",use_container_width=True):

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

    st.subheader("Prediction Result")

    if pred==1:
        st.error(
            f"⚠ Customer Likely To Churn\n\nRisk Probability: {prob:.2%}"
        )
    else:
        st.success(
            f"✅ Customer Likely To Stay\n\nRetention Probability: {(1-prob):.2%}"
        )

    st.progress(float(prob))

    st.metric(
        label="Churn Risk Score",
        value=f"{prob:.2%}"
    )

st.divider()
st.caption("Built with Random Forest + Streamlit")
