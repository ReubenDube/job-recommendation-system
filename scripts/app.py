# app.py
import streamlit as st
import pandas as pd
import joblib
from sqlalchemy import create_engine
from sklearn.metrics.pairwise import cosine_similarity
from pathlib import Path

st.set_page_config(page_title="Job Recommendation System", layout="wide")
st.title("🔍 Job Recommendation System")
st.markdown("### Find jobs that match your skills and preferences")

# === HARD-CODED ABSOLUTE PATH (Most Reliable for now) ===
DB_PATH = r"C:\Users\User\Documents\job-postings-etl-pipeline\job-recommendation-system\data\job_postings.db"

st.info(f"Database path: {DB_PATH}")

if not Path(DB_PATH).exists():
    st.error("❌ Database file not found!")
    st.info("Please confirm that `job_postings.db` exists in the `data` folder.")
    st.stop()

# Load data and models
@st.cache_resource
def load_assets():
    engine = create_engine(f"sqlite:///{DB_PATH}")
    df = pd.read_sql("SELECT * FROM job_postings", engine)
    
    PROJECT_ROOT = Path(DB_PATH).parent.parent
    tfidf = joblib.load(PROJECT_ROOT / "models" / "tfidf_vectorizer.joblib")
    tfidf_matrix = joblib.load(PROJECT_ROOT / "models" / "tfidf_matrix.joblib")
    
    return df, tfidf, tfidf_matrix

df, tfidf, tfidf_matrix = load_assets()

st.success(f"✅ Successfully loaded {len(df):,} job postings!")

# User Input
user_input = st.text_area(
    "Describe the type of job you're looking for",
    placeholder="Data Scientist with Python, SQL, Machine Learning experience. Looking for roles in Pretoria or remote...",
    height=130
)

if st.button("🔍 Get Personalized Recommendations", type="primary"):
    if user_input.strip():
        with st.spinner("Finding the best matching jobs..."):
            user_vector = tfidf.transform([user_input])
            similarity_scores = cosine_similarity(user_vector, tfidf_matrix).flatten()
            
            top_indices = similarity_scores.argsort()[-10:][::-1]
            recommendations = df.iloc[top_indices].copy()
            recommendations['match_score'] = (similarity_scores[top_indices] * 100).round(1)

            st.subheader(f"Top {len(recommendations)} Recommendations")

            for _, row in recommendations.iterrows():
                with st.expander(f"**{row['title']}** — **{row['match_score']:.1f}%** Match"):
                    st.write(f"**Location**: {row.get('location', 'N/A')}")
                    st.write(f"**Industry**: {row.get('industry', 'N/A')}")
                    st.write(f"**Function**: {row.get('function', 'N/A')}")
                    st.write("**Description**:", str(row.get('description', ''))[:350] + "...")
    else:
        st.warning("Please describe the job you are looking for.")

st.caption("Project 3 • Job Recommendation System")