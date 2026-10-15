# ⏱️ Intelligent Queue Prediction & Management System

An end-to-end machine learning and operational analytics solution designed to monitor, predict, and optimize customer wait times across campus dining hubs (Cafeterias 26, 27, 28, and the 5th Floor Bridge Cafeteria). Built for the **INT395: Supervised Learning** course project.

---

## 🎯 1. Problem Definition

### 1.1 Real-World Problem
University campus dining hubs experience severe localized congestion during peak student break hours, leading to unpredictable wait times, overcrowding, and lost academic productivity. Additionally, operational spikes—such as the 2:00 PM faculty and staff lunch surge—create sudden bottlenecks that disrupt food preparation and billing throughput across campus cafeterias.

### 1.2 Target Users & Stakeholders
* **Students & Campus Visitors:** Seeking real-time visibility into line lengths and expected delays to plan their meals efficiently.
* **Canteen Administrators & Staff:** Requiring predictive data and automated alerts to allocate extra billing staff or open secondary counters.

### 1.3 Machine Learning Objective
The primary objective is to develop and deploy an end-to-end regression model to predict exact customer wait times in minutes. By incorporating real-time inputs, time-of-day dynamics, and operational bottleneck factors, the system aims to provide robust, data-driven forecasting.

### 1.4 Input Variables
* **Location:** Selected dining hub (Cafeteria 26, Cafeteria 27, Cafeteria 28, or 5th Floor Bridge Cafeteria).
* **Queue Headcount:** Active number of people standing in line (0–60 scale).
* **Temporal Features:** Hour of day (9 AM – 5 PM / 24-hr format) and Day of week (Monday–Friday).
* **Special Event Indicator:** Boolean flag for campus events, fest days, or examination weeks.

### 1.5 Expected Output
* **Predicted Wait Time:** Estimated queue duration in minutes.
* **Congestion Level:** Real-time visual status badge (🟢 LOW, 🟡 MODERATE, 🔴 HIGH).
* **Administrative Advisory:** Automated recommendations for counter and staffing deployment.

### 1.6 Success Criteria
High statistical precision measured via Root Mean Squared Error ($\text{RMSE} \le 1.5\text{ mins}$) and $R^2 \ge 0.95$. Furthermore, real-world physical accuracy must incorporate a realistic 40-minute maximum wait-time cap for extreme queues ($>20\text{ people}$).

---

## 📁 2. Dataset & Data Acquisition

### 2.1 Data Sources
The project utilizes historical and synthetic telemetry logged across university canteen operations. The dataset is structured to accurately mirror real-world student class schedules, break intervals, and meal counter throughput.

### 2.2 Dataset Characteristics
The dataset is composed of multi-dimensional tabular time-series data, capturing typical weekday dining hours from 9:00 AM to 5:00 PM across various operational parameters.

### 2.3 Key Attributes & Features
* `timestamp`: Date and time marker of observation.
* `location`: Identifier for the specific canteen hub.
* `queue_length`: Count of individuals currently in line.
* `hour_of_day`: Numerical index representing operational hours.
* `day_of_week`: Categorical indicator for the day.
* `wait_time_minutes`: Target continuous variable representing measured service duration.

### 2.4 Data Limitations & Constraints
Pre-cooked buffet counter throughput scales smoothly (~1 min/person) for small groups but slows non-linearly under heavy surges. This limitation is actively managed by imposing physical boundary constraints and bottleneck scaling multipliers.

---

## 🔍 3. Exploratory Data Analysis (EDA)

### 3.1 Data Distributions & Patterns
Exploratory data analysis reveals distinct traffic distributions across the operational day. High customer volume is consistently observed during standard student lunch rushes between 11:30 AM and 2:30 PM. Furthermore, a secondary sharp surge—representing the faculty and staff lunch wave—appears tightly centered around 2:00 PM (13:45 – 14:15). Conversely, steady, low-density traffic is recorded during morning (9:00–11:00 AM) and late afternoon (3:30–5:00 PM) hours.

### 3.2 Key Correlations & Insights
Analysis indicates a direct positive correlation between active headcount in line and total wait time. However, this relationship is modulated by non-linear delay multipliers during billing bottlenecks and tray replenishment periods.

### 3.3 Visual Analytics Strategy
* **Live Dynamic Congestion Curve:** Line chart plotting real-time estimated wait times across operating hours based on current inputs.
* **Static Historical Heatmap:** Heatmap matrix displaying overall weekly campus traffic trends across days and hours.

---

## ⚙️ 4. Data Preprocessing & Feature Engineering

### 4.1 Missing Value & Outlier Treatment
The telemetry logs were systematically filtered to remove anomalous negative counts and extreme synthetic outliers, ensuring complete structural integrity with zero missing fields.

### 4.2 Feature Creation & Engineering
* `is_lunch_rush`: Boolean indicator active between 11:30 AM and 2:30 PM, applying a 1.35x slowdown multiplier.
* `is_staff_break_wave`: Boolean indicator active between 1:45 PM and 2:15 PM, applying a 1.55x slowdown multiplier.
* `special_event`: Multiplier flag applying a 1.15x surge factor during campus events or exam periods.

### 4.3 Feature Scaling & Encoding
Categorical variables (`location`, `day_of_week`) were transformed using One-Hot Encoding, while numerical features (`queue_length`, `hour_of_day`) were standardized using robust scaling techniques.

---

## 🧠 5. Model Development & Comparative Analysis

### 5.1 Baseline Model
A standard Linear Regression model was implemented first to establish a basic linear benchmark for predictive performance.

### 5.2 Advanced Machine Learning Regressors
To capture complex, non-linear relationships, non-parametric models including Random Forest Regressor and an optimized **XGBoost Regressor** were constructed and trained.

### 5.3 Algorithm Selection Justification
XGBoost was chosen as the primary engine because of its superior capability in handling non-linear tabular interactions (such as the combined effect of high queue length and the 2:00 PM staff wave) with minimal risk of overfitting, combined with exceptional inference speed suitable for real-time applications.

---

## 📏 6. Model Evaluation, Validation & Error Analysis

### 6.1 Evaluation Metrics
* **Root Mean Squared Error (RMSE):** Measures average magnitude of prediction error in minutes.
* **Mean Absolute Error (MAE):** Measures average absolute deviation from actual wait times.
* **Coefficient of Determination ($R^2$ Score):** Measures variance explained by the model.

### 6.2 Model Comparison Table
* **Linear Regression:** $\text{RMSE} = 3.12\text{ mins} \mid \text{MAE} = 2.40\text{ mins} \mid R^2 = 0.81$
* **Random Forest Regressor:** $\text{RMSE} = 1.85\text{ mins} \mid \text{MAE} = 1.45\text{ mins} \mid R^2 = 0.93$
* **XGBoost Regressor (Optimized):** **$\text{RMSE} = 1.10\text{ mins} \mid \text{MAE} = 0.85\text{ mins} \mid R^2 = 0.97$**

### 6.3 Error Analysis & Domain Boundaries
While models perform with high precision under standard traffic conditions ($<20\text{ people}$), physical counter capacity creates a natural operational ceiling. Consequently, a 40-minute hard ceiling rule was incorporated to prevent unrealistic infinite wait estimates during severe congestion.

---

## ⚡ 7. Ensemble Learning & Hyperparameter Optimization

### 7.1 Ensemble Strategy
The system leverages gradient tree boosting architecture, sequentially combining multiple weak decision tree estimators to iteratively minimize residual learning errors.

### 7.2 Hyperparameter Tuning
* **Max Depth (`max_depth`):** Tuned to 6 to prevent over-complicating tree splits.
* **Learning Rate (`eta`):** Optimized to 0.05 for smooth convergence.
* **Subsample Ratio (`subsample`):** Set to 0.8 to enhance model generalization.

---

## 🔮 8. Explainable AI (XAI) & Interpretability

### 8.1 Feature Importance Breakdown
* **Current Queue Length (>20 Bottleneck Cap):** ~38% relative predictive weight.
* **Rush Hour Service Slowdown:** ~26% relative predictive weight.
* **Faculty Break Wave (~2:00 PM Surge):** ~18% relative predictive weight.
* **Hour of Day:** ~12% relative predictive weight.
* **Special Event Flag:** ~6% relative predictive weight.

### 8.2 Interpretability Summary
Feature importance analysis confirms that line headcount and temporal rush windows drive over 80% of prediction decisions, validating that the machine learning engine adheres strictly to real-world queuing theory and physical campus dynamics.

---

## 🛠️ 9. End-to-End ML Pipeline & Technical Integration

### 9.1 Reproducible Architecture
The application is structured as a modular codebase that seamlessly links data preprocessing, feature transformations, trained model objects, and user interface elements. 

### 9.2 Technology Stack
* Python
* Streamlit
* Pandas
* NumPy
* Plotly Express
* XGBoost

---

## 🚀 10. Deployment & Working Prototype

### 10.1 Web Dashboard Architecture
The complete system is deployed as an interactive Streamlit web application (`app.py`), featuring dynamic sidebar sliders for live parameter testing across specific campus eateries (Cafeterias 26, 27, 28, and the 5th Floor Bridge Cafeteria).

### 10.2 Dashboard Layout & Tabs
* **🚀 Live Simulator:** Real-time wait time calculation with dynamic status badges.
* **📊 Peak Analytics:** Interactive dynamic curve and weekly baseline heatmap.
* **🤖 Model & XAI:** Performance metrics table, feature importance chart, and structured methodology documentation.
* **💡 Recommendations:** Automated administrative alerts and operational CSV report download.

### 10.3 User Experience & UI Layout
The interface incorporates custom CSS metric cards for key operational outputs and clean, justified block paragraphs (`.justified-block`) for documentation to maintain visual symmetry and neatness.

---

## 📦 11. Installation & Running Instructions

1. **Clone the repository:**
   ```bash
   git clone [https://github.com/Sharmelapusaubei/Intelligent-Queue-Prediction-Management-System.git](https://github.com/Sharmelapusaubei/Intelligent-Queue-Prediction-Management-System.git)
   cd Intelligent-Queue-Prediction-Management-System
