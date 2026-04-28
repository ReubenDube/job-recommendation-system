# scripts/recommendation_engine.py
import pandas as pd
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import joblib

PROJECT_ROOT = Path(__file__).parent.parent
DB_PATH = PROJECT_ROOT / "data" / "job_postings.db"

def load_jobs():
    from sqlalchemy import create_engine
    engine = create_engine(f"sqlite:///{DB_PATH}")
    df = pd.read_sql("SELECT * FROM job_postings", engine)
    return df

def build_recommendation_system():
    print("🔄 Building Job Recommendation System...")
    df = load_jobs()
    
    # Create a combined text feature for better recommendations
    df['combined_text'] = (
        df['title'].fillna('') + " " + 
        df['description'].fillna('') + " " + 
        df['industry'].fillna('') + " " + 
        df['function'].fillna('')
    )
    
    # Create TF-IDF vectorizer
    tfidf = TfidfVectorizer(stop_words='english', max_features=5000)
    tfidf_matrix = tfidf.fit_transform(df['combined_text'])
    
    # Save the vectorizer and matrix for later use
    joblib.dump(tfidf, PROJECT_ROOT / "models" / "tfidf_vectorizer.joblib")
    joblib.dump(tfidf_matrix, PROJECT_ROOT / "models" / "tfidf_matrix.joblib")
    
    print(f"✅ Recommendation system built successfully!")
    print(f"   Vocabulary size: {len(tfidf.get_feature_names_out())} terms")
    print(f"   Ready to recommend jobs based on text similarity")
    
    return df, tfidf, tfidf_matrix

if __name__ == "__main__":
    build_recommendation_system()