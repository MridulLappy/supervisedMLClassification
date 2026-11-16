import joblib
import pandas as pd
import streamlit as st
 
st.set_page_config(page_title="Dry Bean Classifier", page_icon="🫘", layout="wide")
 
 
@st.cache_resource
def load_artifacts():
    return joblib.load("bean_model.pkl"), joblib.load("scaler.pkl"), pd.read_csv("Beans_Dataset.csv")
 
 
model, scaler, raw = load_artifacts()
 
# Everything else is worked out here, so only the 3 files above need to be saved.
features = list(scaler.feature_names_in_)        # column order the model was trained on
classes = list(model.classes_)
f_min, f_max, f_median = raw[features].min(), raw[features].max(), raw[features].median()
class_median = raw.groupby("Class")[features].median()   # for the example picker
 
 
def predict(values: dict):
    """Repeat the training preprocessing (scaling) on one bean, then predict."""
    X = pd.DataFrame([values], columns=features)
    X_scaled = pd.DataFrame(scaler.transform(X), columns=features)
    label = model.predict(X_scaled)[0]
    proba = None
    if hasattr(model, "predict_proba"):
        proba = pd.Series(model.predict_proba(X_scaled)[0], index=model.classes_).sort_values(ascending=False)
    return label, proba
 
 
st.title("🫘 Dry Bean Type Classifier")
st.write("Enter the physical measurements of a bean (from the camera system) and get the predicted bean type.")
 
# ---- optional example picker
example = st.selectbox("Start from an example bean (optional)", ["Typical bean (overall median)"] + classes)
defaults = f_median if example.startswith("Typical") else class_median.loc[example]
 
# ---- inputs
st.subheader("Measurements")
values = {}
cols = st.columns(4)
for i, f in enumerate(features):
    lo, hi = float(f_min[f]), float(f_max[f])
    step = (hi - lo) / 1000 or 0.001
    values[f] = cols[i % 4].number_input(
        f, min_value=lo, max_value=hi, value=float(defaults[f]), step=step, format="%.6f",
        key=f"{f}_{example}",            # re-create inputs when the example changes
        help=f"Training data range: {lo:.4g} to {hi:.4g}")
 
# ---- prediction
if st.button("Predict bean type", type="primary"):
    label, proba = predict(values)
    st.success(f"Predicted class: **{label}**")
    if proba is not None:
        st.write("Confidence by class")
        st.bar_chart(proba)