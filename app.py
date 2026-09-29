import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# Page Configuration
st.set_page_config(
    page_title="Careem Pulse | AI Churn Engine",
    page_icon="🚖",
    layout="wide"
)

# Custom Styling
st.markdown("""
    <style>
    .main-header {font-size: 2.2rem; font-weight: 700; color: #00EB78;}
    .sub-header {font-size: 1.1rem; color: #6c757d;}
    .metric-card {background-color: #f8f9fa; padding: 15px; border-radius: 10px; border-left: 5px solid #00EB78;}
    </style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-header">🚖 Careem Pulse: AI Churn Predictor & Retention Engine</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Super App Customer Retention & Automated Strategy Generator</div>', unsafe_allow_html=True)
st.write("")

# 1. Generate Synthetic Careem Super-App Data
@st.cache_data
def generate_careem_data():
    np.random.seed(42)
    n = 1000
    data = pd.DataFrame({
        'user_id': [f"CRM_{i:04d}" for i in range(1, n+1)],
        'trips_last_30d': np.random.poisson(lam=5, size=n),
        'food_orders_last_30d': np.random.poisson(lam=3, size=n),
        'days_since_last_activity': np.random.randint(1, 45, size=n),
        'cancellation_rate': np.random.uniform(0.0, 0.45, size=n),
        'avg_rating_given': np.random.uniform(3.0, 5.0, size=n),
        'is_careem_plus': np.random.choice([0, 1], size=n, p=[0.75, 0.25]),
        'support_tickets_30d': np.random.poisson(lam=0.7, size=n)
    })
    
    # Synthetic Churn Logic (Ground Truth Engine)
    churn_score = (
        (data['days_since_last_activity'] > 20).astype(int) * 3.5 +
        (data['cancellation_rate'] > 0.25).astype(int) * 2.5 +
        (data['trips_last_30d'] < 2).astype(int) * 2.0 +
        (data['support_tickets_30d'] > 1).astype(int) * 2.0 -
        (data['is_careem_plus'] == 1).astype(int) * 2.5
    )
    data['churn'] = (churn_score >= 3.5).astype(int)
    return data

df = generate_careem_data()

# 2. Train Prediction Model
@st.cache_resource
def train_model(data):
    X = data.drop(columns=['user_id', 'churn'])
    y = data['churn']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train, y_train)
    acc = clf.score(X_test, y_test)
    return clf, acc, X.columns.tolist()

model, accuracy, feature_names = train_model(df)

# Sidebar - Overview & Navigation
st.sidebar.image("https://upload.wikimedia.org/wikipedia/commons/ thumb/8/87/Careem_logo.svg/512px-Careem_logo.svg.png", width=180)
st.sidebar.markdown("---")
st.sidebar.subheader("🎯 Customer Profiler")
selected_id = st.sidebar.selectbox("Select User ID to Analyze:", df['user_id'])
user_row = df[df['user_id'] == selected_id].iloc[0]

# Display Summary Metrics
st.sidebar.markdown("---")
st.sidebar.metric("Model Classification Accuracy", f"{accuracy:.1%}")

# Main Layout
col_a, col_b = st.columns([1, 1])

with col_a:
    st.subheader("👤 User Activity Profile")
    
    m1, m2 = st.columns(2)
    m1.metric("Rides (30d)", int(user_row['trips_last_30d']))
    m2.metric("Food Orders (30d)", int(user_row['food_orders_last_30d']))
    
    m3, m4 = st.columns(2)
    m3.metric("Days Inactive", int(user_row['days_since_last_activity']))
    m4.metric("Careem Plus Member", "Yes ✅" if user_row['is_careem_plus'] == 1 else "No ❌")
    
    st.write(f"**Cancellation Rate:** {user_row['cancellation_rate']:.1%}")
    st.write(f"**Support Tickets Raised:** {int(user_row['support_tickets_30d'])}")

# Model Prediction Logic
user_features = pd.DataFrame([user_row.drop(['user_id', 'churn'])])
churn_prob = model.predict_proba(user_features)[0][1]

with col_b:
    st.subheader("📊 Risk Assessment")
    st.write("Current Churn Probability Score:")
    st.progress(float(churn_prob))
    
    if churn_prob >= 0.60:
        st.error(f"🚨 **HIGH RISK**: {churn_prob:.1%} Probability of Churn")
        risk_level = "High"
    elif churn_prob >= 0.35:
        st.warning(f"⚠️️ **MODERATE RISK**: {churn_prob:.1%} Probability of Churn")
        risk_level = "Moderate"
    else:
        st.success(f"✅ **LOW RISK**: {churn_prob:.1%} Probability of Churn")
        risk_level = "Low"

# AI Retention Playbook Generator
st.markdown("---")
st.subheader("🤖 AI-Driven Retention Strategy Playbook")

def generate_ai_playbook(user, prob, risk):
    reasons = []
    if user['days_since_last_activity'] > 15:
        reasons.append(f"Inactivity of {int(user['days_since_last_activity'])} days")
    if user['cancellation_rate'] > 0.20:
        reasons.append(f"High cancellation rate ({user['cancellation_rate']:.0%})")
    if user['support_tickets_30d'] > 0:
        reasons.append(f"{int(user['support_tickets_30d'])} unresolved support ticket(s)")
    if user['is_careem_plus'] == 0:
        reasons.append("Non-subscriber to Careem Plus")
        
    primary_cause = ", ".join(reasons) if reasons else "Slight drop in monthly engagement"
    
    if risk == "High":
        incentive = "Offer 1-Month Free Careem Plus + AED 20 voucher for next 2 Food/Ride orders."
        channel = "Push Notification + Direct WhatsApp Message"
        message_copy = f"We miss you, {user['user_id']}! Enjoy 1-Month Free Careem Plus on us + AED 20 off your next trip."
    elif risk == "Moderate":
        incentive = "15% discount on next 3 Ride bookings."
        channel = "In-App Banner + Push Notification"
        message_copy = f"Ready for your next ride? Take 15% off your next 3 trips with code CAREEM15!"
    else:
        incentive = "Loyalty points multiplier (2x points on next order)."
        channel = "In-App Push"
        message_copy = f"Thanks for being a loyal Careem customer! Earn 2x reward points today."
        
    return primary_cause, incentive, channel, message_copy

cause, offer, channel, push_text = generate_ai_playbook(user_row, churn_prob, risk_level)

if st.button("Generate Dynamic Retention Action Plan"):
    with st.spinner("Analyzing Super-App telemetry & behavioral signals..."):
        c1, c2 = st.columns(2)
        with c1:
            st.markdown(f"**1. Root Cause Diagnosis:**\n- {cause}")
            st.markdown(f"**2. Recommended Incentive:**\n- {offer}")
        with c2:
            st.markdown(f"**3. Target Channel:**\n- {channel}")
            st.markdown(f"**4. Tailored Copy:**\n> *\"{push_text}\"*")
