import streamlit as st
import pandas as pd
import joblib

# ==========================
# Page Configuration
# ==========================

st.set_page_config(
    page_title="Email Spam Detector",
    page_icon="📧",
    layout="centered"
)

# ==========================
# Load Model
# ==========================

try:
    model = joblib.load("Spam_detector_complete.pkl")
except Exception as e:
    st.error("❌ Model could not be loaded.")
    st.exception(e)
    st.stop()

# ==========================
# Header
# ==========================

try:
    st.image("Spam_icon.png", width=120)
except:
    pass

st.title("📧 Email Spam Detector")
st.write("Fill in the email details and click **Predict**.")

st.divider()

# ==========================
# User Inputs
# ==========================

email_id = st.number_input(
    "Email ID",
    min_value=0,
    value=1001,
    step=1
)

sender_email = st.text_input(
    "Sender Email",
    placeholder="example@gmail.com"
)

subject = st.text_input(
    "Subject",
    placeholder="Enter Email Subject"
)

email_length = st.number_input(
    "Email Length",
    min_value=0,
    value=100
)

num_links = st.number_input(
    "Number of Links",
    min_value=0,
    value=0
)

num_special_chars = st.number_input(
    "Number of Special Characters",
    min_value=0,
    value=0
)

capital_words = st.number_input(
    "Capital Words",
    min_value=0,
    value=0
)

has_attachment = st.selectbox(
    "Has Attachment?",
    ["No", "Yes"]
)

attachment = 1 if has_attachment == "Yes" else 0

st.divider()

# ==========================
# Prediction Button
# ==========================

if st.button("🔍 Predict", use_container_width=True):

    new_email = pd.DataFrame({
        "Email_ID": [email_id],
        "Sender_Email": [sender_email],
        "Subject": [subject],
        "Email_Length": [email_length],
        "Num_Links": [num_links],
        "Num_Special_Chars": [num_special_chars],
        "Capital_Words": [capital_words],
        "Has_Attachment": [attachment]
    })

    prediction = model.predict(new_email)
    pred = int(prediction[0])

    st.subheader("Prediction Result")

    if pred == 1:
        st.error("🚨 Spam Email Detected")
    else:
        st.success("✅ No Spam Email Detected")

    # Show prediction probability if available
    if hasattr(model, "predict_proba"):
        probability = model.predict_proba(new_email)[0]

        spam_probability = probability[1] * 100
        not_spam_probability = probability[0] * 100

        st.write("### Confidence")

        st.progress(int(max(spam_probability, not_spam_probability)))

        col1, col2 = st.columns(2)

        with col1:
            st.metric(
                "Not Spam",
                f"{not_spam_probability:.2f}%"
            )

        with col2:
            st.metric(
                "Spam",
                f"{spam_probability:.2f}%"
            )

    with st.expander("Entered Email Details"):
        st.dataframe(new_email)