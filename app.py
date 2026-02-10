from fastapi import FastAPI
from pydantic import BaseModel
from supabase import create_client
import pandas as pd
import os
import re
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from dotenv import load_dotenv
from typing import Optional


# --------------------
# App setup
# --------------------
app = FastAPI(title="Recipe Recommendation API")

load_dotenv()

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")

supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

# --------------------
# Load & prepare data ONCE at startup
# --------------------
print("Loading recipes from Supabase...")

response = supabase.table("recipe").select("*").execute()
df = pd.DataFrame(response.data)

# Ensure strings
df["name"] = df["name"].astype(str)
df["ingredients"] = df["ingredients"].astype(str)

# --------------------
# Text cleaning
# --------------------
nltk.download("stopwords")
stop_words = set(stopwords.words("english"))

def clean_text(text: str) -> str:
    if not isinstance(text, str):
        return ""

    text = re.sub(r"\d+(\.\d+)?", "", text)
    text = re.sub(
        r"\b(cup|cups|tbsp|tsp|tablespoon|tablespoons|ounce|oz|grams|g|kg|pound|lb)\b",
        "",
        text,
        flags=re.IGNORECASE,
    )
    text = re.sub(r"[^a-zA-Z\s]", "", text)
    text = text.lower()
    text = re.sub(r"\s+", " ", text).strip()
    tokens = [w for w in text.split() if w not in stop_words]
    return " ".join(tokens)

df["clean_text"] = (df["name"] + " " + df["ingredients"]).apply(clean_text)

# --------------------
# TF-IDF model (built once)
# --------------------
vectorizer = TfidfVectorizer(stop_words="english")
tfidf_matrix = vectorizer.fit_transform(df["clean_text"])

print("TF-IDF model ready.")

# --------------------
# API schemas
# --------------------
class RecommendationRequest(BaseModel):
    query: str
    top_k: int=5
    forbidden_ingredients: list[str] = []
    strict: bool = False

def filter_by_allergies(results_df, forbidden):
    if not forbidden:
        return results_df

    pattern = "|".join(forbidden)
    return results_df[
        ~results_df["ingredients"].str.contains(
            pattern, case=False, regex=True
        )
    ]
    
def filter_by_strict_mode(results_df, user_ingredients, strict):
    if not strict:
        return results_df

    def is_valid(recipe_ingredients):
        recipe_set = set(clean_text(recipe_ingredients).split())
        return recipe_set.issubset(user_ingredients)

    return results_df[results_df["ingredients"].apply(is_valid)]

class RecommendationResponse(BaseModel):
    id: int
    name: str
    ingredients: str
    img_src: Optional[str] = None

# --------------------
# Recommendation endpoint
# --------------------
@app.post("/recommend", response_model=list[RecommendationResponse])
def recommend(data: RecommendationRequest):
    cleaned_query = clean_text(data.query)
    query_vec = vectorizer.transform([cleaned_query])
    user_ingredients = set(cleaned_query.split())

    scores = cosine_similarity(query_vec, tfidf_matrix)[0]

    # ia mai multi candidati ca sa poti filtra
    top_idx = scores.argsort()[-(data.top_k * 5):][::-1]
    candidates = df.iloc[top_idx]

    # filtreaza alergii
    filtered = filter_by_allergies(
        candidates,
        data.forbidden_ingredients
    )
    filtered = filter_by_strict_mode(
        filtered,
        user_ingredients,
        data.strict
    )


    results = filtered.head(data.top_k)[["id","name","ingredients","img_src"]]


    return results.to_dict(orient="records")

