# scripts/data_loader.py
import pandas as pd
from pathlib import Path
from sqlalchemy import create_engine

PROJECT_ROOT = Path(__file__).parent.parent
DB_PATH = PROJECT_ROOT / "data" / "job_postings.db"

def load_jobs():
    """Load job postings from the ETL database"""
    if not DB_PATH.exists():
        raise FileNotFoundError(f"Database not found at {DB_PATH}. Copy it from previous project.")
    
    engine = create_engine(f"sqlite:///{DB_PATH}")
    df = pd.read_sql("SELECT * FROM job_postings", engine)
    
    print(f"✅ Successfully loaded {len(df):,} job postings")
    print(f"Shape: {df.shape}")
    print(f"Columns: {df.columns.tolist()}")
    
    return df

if __name__ == "__main__":
    df = load_jobs()
    
    print("\nSample Job:")
    print("Title:", df['title'].iloc[0])
    print("Location:", df['location'].iloc[0])
    print("Industry:", df['industry'].iloc[0])
    print("\nReady for building the recommendation system!")