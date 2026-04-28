import streamlit as st
import joblib
import pandas as pd
import shap
import warnings

warnings.filterwarnings("ignore")

# ---------- CONFIG ----------
st.set_page_config(page_title="CardioRisk AI", page_icon="❤️", layout="wide")

# ---------- LOAD MODEL ----------
data = joblib.load("model.pkl")
model = data["model"]
scaler = data["scaler"]
features = data["features"]

# ---------- UI ----------
st.markdown("""
<style>
.card {
    padding: 20px;
    border-radius: 15px;
    background: white;
    box-shadow: 0px 4px 15px rgba(0,0,0,0.05);
    margin-bottom: 20px;
}
.title {font-size: 28px; font-weight: bold;}
.subtitle {color: gray; margin-bottom: 20px;}
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="title">❤️ CardioRisk AI</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">AI-powered cardiovascular risk assessment</div>', unsafe_allow_html=True)

col1, col2 = st.columns([1, 1.2])

# ---------- INPUT ----------
with col1:
    st.markdown('<div class="card">', unsafe_allow_html=True)

    age = st.slider("Age", 20, 80, 40)
    height = st.number_input("Height (cm)", 100, 220, 170)
    weight = st.number_input("Weight (kg)", 30, 150, 70)

    gender = st.selectbox("Gender", ["Female", "Male"])
    cholesterol = st.selectbox("Cholesterol", [1, 2, 3])
    gluc = st.selectbox("Glucose", [1, 2, 3])

    smoke = st.selectbox("Smoking", [0, 1])
    alco = st.selectbox("Alcohol", [0, 1])
    active = st.selectbox("Physically Active", [0, 1])

    ap_hi = st.number_input("Systolic BP", 80, 200, 120)
    ap_lo = st.number_input("Diastolic BP", 50, 150, 80)

    predict_btn = st.button("🔍 Analyze Risk", use_container_width=True)

    st.markdown('</div>', unsafe_allow_html=True)

# ---------- PREP ----------
def prepare_input():
    gender_val = 1 if gender == "Female" else 2

    df = pd.DataFrame([{
        "age": age,
        "gender": gender_val,
        "height": height,
        "weight": weight,
        "ap_hi": ap_hi,
        "ap_lo": ap_lo,
        "cholesterol": cholesterol,
        "gluc": gluc,
        "smoke": smoke,
        "alco": alco,
        "active": active
    }])

    df["BMI"] = df["weight"] / ((df["height"] / 100) ** 2)
    df["pulse_pressure"] = df["ap_hi"] - df["ap_lo"]

    df = df.reindex(columns=features, fill_value=0)
    return df

# ---------- TEXT HELPERS ----------
def feature_to_text(feature):
    mapping = {
        "ap_hi": "high systolic blood pressure",
        "ap_lo": "high diastolic blood pressure",
        "BMI": "unhealthy body weight",
        "cholesterol": "high cholesterol",
        "gluc": "high blood sugar",
        "smoke": "smoking",
        "alco": "alcohol intake",
        "active": "low physical activity",
        "pulse_pressure": "abnormal blood pressure gap"
    }
    return mapping.get(feature, feature)

def recommendation(feature):
    tips = {
        "ap_hi": "Reduce salt, manage stress, and monitor BP regularly.",
        "ap_lo": "Maintain a healthy lifestyle and monitor BP.",
        "BMI": "Aim for gradual weight loss with diet and exercise.",
        "cholesterol": "Reduce saturated fats and eat more fiber.",
        "gluc": "Reduce sugar intake and monitor glucose levels.",
        "smoke": "Quit smoking — biggest improvement you can make.",
        "alco": "Limit alcohol intake.",
        "active": "Do at least 30 mins of exercise most days.",
        "pulse_pressure": "Monitor BP and consult a doctor if needed."
    }
    return tips.get(feature, "Improve this factor to reduce risk.")

def detailed_explanation(feature, df):
    row = df.iloc[0]
    bmi = row["BMI"]

    explanations = {
        "ap_hi": f"Systolic BP is {row['ap_hi']}, above healthy range.",
        "ap_lo": f"Diastolic BP is {row['ap_lo']}, may strain the heart.",
        "BMI": f"BMI is {bmi:.1f}, outside healthy range (18.5–24.9).",
        "cholesterol": "Elevated cholesterol increases artery blockage risk.",
        "gluc": "High blood sugar increases cardiovascular risk.",
        "smoke": "Smoking damages blood vessels.",
        "alco": "Alcohol can negatively affect heart health.",
        "active": "Low activity weakens cardiovascular fitness.",
        "pulse_pressure": "Large BP gap may indicate vascular stiffness."
    }

    return explanations.get(feature, "This factor increases your risk.")

# ---------- FILTER ----------
def is_actionable(feature, df):
    row = df.iloc[0]
    bmi = row["BMI"]

    if feature in ["height", "age", "gender", "weight"]:
        return False

    if feature == "ap_hi" and 120 <= ap_hi <= 130:
        return False
    if feature == "ap_lo" and 80 <= ap_lo <= 85:
        return False

    if feature == "BMI" and 18.5 <= bmi <= 24.9:
        return False

    if feature == "cholesterol" and cholesterol == 1:
        return False

    if feature == "gluc" and gluc == 1:
        return False

    if feature == "smoke" and smoke == 0:
        return False

    if feature == "alco" and alco == 0:
        return False

    if feature == "active" and active == 1:
        return False

    return True

# ---------- OUTPUT ----------
with col2:
    if predict_btn:
        st.markdown('<div class="card">', unsafe_allow_html=True)

        df = prepare_input()
        scaled = scaler.transform(df)
        prob = model.predict_proba(scaled)[0][1]

        st.subheader("📊 Risk Assessment")

        # 🎉 HEALTHY MESSAGE
        if prob < 0.2:
            st.success("🎉 You are quite healthy! Keep going — stay active and keep pumping happiness ❤️")
        else:
            if prob < 0.3:
                label, color = "Low Risk", "green"
            elif prob < 0.7:
                label, color = "Moderate Risk", "orange"
            else:
                label, color = "High Risk", "red"

            st.markdown(f"<h2 style='color:{color}'>{label}</h2>", unsafe_allow_html=True)
            st.progress(int(prob * 100))
            st.metric("Risk Probability", f"{prob:.2f}")

            st.markdown("---")

            # ---------- SHAP ----------
            try:
                explainer = shap.Explainer(model)
                shap_values = explainer(scaled)

                values = shap_values.values[0]
                contributions = list(zip(features, values))

                contributions = [
                    (f, val) for f, val in contributions
                    if is_actionable(f, df) and val > 0
                ]

                if contributions:
                    contributions.sort(key=lambda x: abs(x[1]), reverse=True)
                    top = contributions[:3]

                    st.subheader("🧠 Key Risk Drivers")

                    for f, val in top:
                        st.markdown(f"""
                        <div style="padding:12px;border-radius:10px;margin-bottom:10px;
                        background:#fff4f4;border-left:5px solid #ff4d4d;">
                        <b>⚠️ {feature_to_text(f)}</b><br>
                        <span style="color:gray;">{detailed_explanation(f, df)}</span><br><br>
                        <b>💡 What you can do:</b> {recommendation(f)}
                        </div>
                        """, unsafe_allow_html=True)

                else:
                    st.success("✅ No major actionable risk factors detected.")

            except Exception as e:
                st.error(f"SHAP Error: {e}")

        st.markdown('</div>', unsafe_allow_html=True)