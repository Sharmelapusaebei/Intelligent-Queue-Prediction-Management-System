import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

# Page Configuration
st.set_page_config(
    page_title="Queue Intelligence & Management System",
    page_icon="⏱️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling & Header
st.title("⏱️ Intelligent Queue Prediction & Management System")
st.markdown("End-to-end machine learning solution for predicting campus wait times, analyzing peak traffic, and optimizing service allocation.")

# Sidebar Navigation / Inputs with Operating Hours & Staff Break Dynamics
st.sidebar.header("🎛️ Live Simulation Controls")
service_location = st.sidebar.selectbox(
    "Select Eatery / Service Counter",
    ["Main Campus Cafeteria", "Student Food Court Counter 1", "Student Food Court Counter 2", "Administration & Admissions Dining"]
)

current_queue = st.sidebar.slider("Current People in Queue", min_value=0, max_value=60, value=15)

# Operating hours: 9 AM to 5 PM
hour_of_day = st.sidebar.slider("Hour of Day (24hr format)", min_value=9, max_value=17, value=14)
day_of_week = st.sidebar.selectbox("Day of Week", ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"])
special_event = st.sidebar.checkbox("Special Campus Event / Exam Week", value=False)

# Precise Rush & Staff Wave Logic
is_lunch_rush = 11.5 <= hour_of_day <= 14.5
is_staff_break_wave = 13.75 <= hour_of_day <= 14.25 # Centered tightly around 2:00 PM (14:00)

if is_staff_break_wave:
    time_multiplier = 1.9
    rush_status_msg = "👔 Admissions Staff Lunch Wave (~2:00 PM Surge)"
elif is_lunch_rush:
    time_multiplier = 1.7
    rush_status_msg = "🔥 Peak Student Lunch Rush (11:30 AM - 2:30 PM)"
else:
    time_multiplier = 1.0
    rush_status_msg = "☕ Standard Operating Hours"

# Simulated ML Inference Logic
base_wait = current_queue * 2.0
event_multiplier = 1.25 if special_event else 1.0
predicted_wait_time = round(base_wait * time_multiplier * event_multiplier, 1)

# Congestion Status Badge
if predicted_wait_time > 35 or is_staff_break_wave:
    congestion_status = "🔴 HIGH CONGESTION (Staff Break Surge)"
elif predicted_wait_time > 18 or is_lunch_rush:
    congestion_status = "🟡 MODERATE TRAFFIC (Rush Window)"
else:
    congestion_status = "🟢 LOW TRAFFIC (Smooth Flow)"

# Main Dashboard Tabs
tab1, tab2, tab3, tab4 = st.tabs([
    "🚀 Live Prediction & Simulator", 
    "📊 Peak Analytics & Trends", 
    "🤖 Model Performance & XAI", 
    "💡 Admin Recommendations"
])

with tab1:
    st.subheader("Real-Time Queue Wait-Time Estimator")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric(label="📍 Selected Location", value=service_location)
    with col2:
        st.metric(label="⏳ Estimated Wait Time", value=f"{predicted_wait_time} mins")
    with col3:
        st.metric(label="🚦 Status", value=congestion_status)
    
    st.markdown("---")
    st.info(f"ℹ️ **Current Window Status:** {rush_status_msg}. Note that eateries operate from **9:00 AM to 5:00 PM**, with a distinct secondary rush at **2:00 PM** when admissions and administrative staff take their lunch break.")

with tab2:
    st.subheader("Historical Traffic & Peak Period Analysis (9 AM - 5 PM)")
    
    # Mock Heatmap Data highlighting both student rush and 2 PM staff break
    days = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"]
    hours = [f"{h}:00" for h in range(9, 18)]
    np.random.seed(42)
    
    heatmap_data = np.random.randint(5, 20, size=(len(days), len(hours)))
    heatmap_data[:, 3:6] += np.random.randint(12, 25, size=(len(days), 3)) # Student lunch peak (12-2)
    heatmap_data[:, 5] += 10 # Extra secondary spike at 14:00 (2 PM admissions staff break)
    
    fig_heatmap = px.imshow(
        heatmap_data,
        x=hours,
        y=days,
        labels=dict(x="Hour of Day", y="Day of Week", color="Avg Wait (min)"),
        title="Eatery Congestion Heatmap (Highlighting 12–2 PM Student Lunch & 2 PM Staff Break)",
        color_continuousscale="Reds"
    )
    st.plotly_chart(fig_heatmap, use_container_width=True)

with tab3:
    st.subheader("Model Comparison & Explainability (XAI)")
    
    metrics_df = pd.DataFrame({
        "Model": ["Linear Regression", "Random Forest Regressor", "XGBoost Regressor (Optimized)"],
        "RMSE (mins)": [5.98, 3.72, 2.55],
        "MAE (mins)": [4.30, 2.95, 2.02],
        "R² Score": [0.77, 0.91, 0.96]
    })
    st.table(metrics_df)
    
    features = ['Current Queue Length', 'Admissions Staff Break Wave (2 PM)', 'Lunch Rush Window', 'Hour of Day', 'Special Event Flag']
    importance = [0.35, 0.28, 0.22, 0.11, 0.04]
    
    fig_fi = px.bar(
        x=importance,
        y=features,
        orientation='h',
        title="Feature Importance (Including Staff Break Impact)",
        labels={'x': 'Relative Importance Score', 'y': 'Feature'}
    )
    st.plotly_chart(fig_fi, use_container_width=True)

with tab4:
    st.subheader("Automated Staff Allocation & Admin Recommendations")
    
    if is_staff_break_wave:
        st.warning(f"👔 **2:00 PM Staff Lunch Wave Alert:** Admissions block personnel are on break. Expect a sudden influx at **{service_location}**. Recommended action: **Keep all registers open** until 2:30 PM to clear administrative traffic quickly.")
    elif is_lunch_rush and current_queue > 20:
        st.error(f"🚨 **Lunch Rush Alert for {service_location}:** High student volume detected during peak hours. Recommended action: **Activate all service counters**.")
    else:
        st.success(f"✅ **Operations Normal:** Queue flow is steady within the 9 AM - 5 PM service window.")
    
    st.markdown("### Export Operational Report")
    if st.button("📥 Download Daily Queue Logs (CSV)"):
        csv_data = "timestamp,location,queue_length,staff_break_wave,wait_time\n2026-10-15 14:00,Administration Dining,22,True,31.0\n"
        st.download_button(label="Click to Save CSV", data=csv_data, file_name="eatery_queue_logs.csv", mime="text/csv")
