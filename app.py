
# IMPORTS

import streamlit as st
from utils import extract_text_from_pdf, preprocess_text, calculate_similarity

# APP TITLE

st.title("📄 AI Resume Analyzer")
st.write("Upload your Resume and compare it with Job Description")

# FILE UPLOAD

uploaded_file = st.file_uploader("Upload Resume (PDF)", type=["pdf"])

# JOB DESCRIPTION INPUT

jd_text = st.text_area("Paste Job Description Here")

# ANALYZE BUTTON

if st.button("Analyze Resume"):

    if uploaded_file is not None and jd_text != "":

        # Extract resume text
        resume_text = extract_text_from_pdf(uploaded_file)

        # Preprocess both texts
        resume_text = preprocess_text(resume_text)
        jd_text = preprocess_text(jd_text)

        # Calculate similarity
        score = calculate_similarity(resume_text, jd_text)

        st.success(f"Match Score: {score}%")

        # Score Interpretation (Realistic Scale)
        if score > 45:
            st.write("✅ Strong Match!")
        elif score > 25:
            st.write("⚡ Moderate Match")
        else:
            st.write("❌ Low Match - Improve Skills")

    else:
        st.warning("Please upload resume and enter job description.")
