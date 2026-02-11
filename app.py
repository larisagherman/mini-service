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
from nltk.stem import WordNetLemmatizer

nltk.download("stopwords")
nltk.download("wordnet")

lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words("english"))
PREPARATION_WORDS = {
    "softened", "packed", "chopped", "optional", "melted",
    "ground", "large", "small", "medium",
    "room", "temperature", "beaten", "divided",
    "unsalted", "salted", "fresh", "dried",
    "sliced", "minced", "crushed", "peeled",
    "shredded", "grated", "to", "taste"
}

# --------------------
# App setup
# --------------------
app = FastAPI(title="Recipe Recommendation API")
load_dotenv()

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

# --------------------
# Load & prepare data ONCE
# --------------------
response = supabase.table("recipe").select("*").execute()
df = pd.DataFrame(response.data)
df["name"] = df["name"].astype(str)
df["ingredients"] = df["ingredients"].astype(str)

# --------------------
# Normalizare ingrediente / text
# --------------------
def normalize_ingredient(text: str) -> str:
    """Lowercase, remove numbers/units/punctuation, lemmatize, remove stopwords"""
    text = re.sub(r"\d+(\.\d+)?", "", text)  # remove numbers
    text = re.sub(r"\b(cup|cups|tbsp|tsp|tablespoon|tablespoons|ounce|oz|grams|g|kg|pound|lb)\b", "", text, flags=re.IGNORECASE)
    text = re.sub(r"[^a-zA-Z\s]", "", text)
    text = text.lower().strip()
    tokens = [
        lemmatizer.lemmatize(t)
        for t in text.split()
        if t not in stop_words and t not in PREPARATION_WORDS
    ]
    return " ".join(tokens)

def get_recipe_tokens(recipe_ingredients: str) -> set:
    """Return set of normalized tokens for a recipe"""
    tokens = set()
    for line in re.split(r'[\n,]', recipe_ingredients):
        line = line.strip()
        if line:
            normalized = normalize_ingredient(line)
            if normalized:
                tokens.add(normalized)
    return tokens

def clean_text(text: str) -> str:
    """Simple cleaning for TF-IDF"""
    return " ".join([normalize_ingredient(word) for word in text.split()])

df["clean_text"] = (df["name"] + " " + df["ingredients"]).apply(clean_text)

# --------------------
# TF-IDF model
# --------------------
vectorizer = TfidfVectorizer(stop_words="english")
tfidf_matrix = vectorizer.fit_transform(df["clean_text"])

# --------------------
# Schemas
# --------------------
class RecommendationRequest(BaseModel):
    query: str
    top_k: int = 5
    forbidden_ingredients: list[str] = []
    strict: bool = False

class RecommendationResponse(BaseModel):
    id: int
    name: str
    ingredients: str
    img_src: Optional[str] = None
    missing_count: Optional[int] = None
    missing_ingredients: Optional[list[str]] = None

# --------------------
# Endpoint
# --------------------
@app.post("/recommend", response_model=list[RecommendationResponse])
def recommend(data: RecommendationRequest):
    # 1️⃣ Normalize user input
    user_tokens = set()
    for i in re.split(r'[\n,]', data.query):
        i = i.strip()
        if i:
            normalized = normalize_ingredient(i)
            if normalized:
                user_tokens.add(normalized)
    
    # 2️⃣ TF-IDF similarity
    query_vec = vectorizer.transform([clean_text(data.query)])
    tfidf_scores = cosine_similarity(query_vec, tfidf_matrix)[0]
    
    # 3️⃣ Candidates
    top_idx = tfidf_scores.argsort()[-(data.top_k*5):][::-1]
    candidates = df.iloc[top_idx].copy()
    
    # 4️⃣ Filter forbidden ingredients
    if data.forbidden_ingredients:
        pattern = "|".join(data.forbidden_ingredients)
        candidates = candidates[~candidates["ingredients"].str.contains(pattern, case=False, regex=True)]
    
    # 5️⃣ Calculate missing ingredients & score
    recipe_scores = []
    exact_match_found = False

    for i, row in candidates.iterrows():
        recipe_tokens = get_recipe_tokens(row["ingredients"])
        missing_tokens = recipe_tokens - user_tokens
        missing_count = len(missing_tokens)

        if missing_count == 0:
            exact_match_found = True

        # strict mode removes non-exact matches ONLY if exact exists
        if data.strict and missing_count > 0:
            continue

        score = tfidf_scores[i]
        recipe_scores.append((score, missing_count, missing_tokens, row))


    # ---------- STRICT FALLBACK ----------
    fallback_mode = False

    if data.strict and not exact_match_found:
        fallback_mode = True
        recipe_scores = []

        for i, row in candidates.iterrows():
            recipe_tokens = get_recipe_tokens(row["ingredients"])
            missing_tokens = recipe_tokens - user_tokens
            missing_count = len(missing_tokens)
            score = tfidf_scores[i]
            recipe_scores.append((score, missing_count, missing_tokens, row))


    # ---------- SORTING ----------
    if fallback_mode:
        # PRIORITIZE fewest ingredients to buy
        recipe_scores.sort(key=lambda x: (x[1], -x[0]))
    else:
        # normal ranking
        recipe_scores.sort(key=lambda x: (-x[0], x[1]))

    
    # 6️⃣ Prepare results
    results = []
    for score, missing_count, missing_tokens, row in recipe_scores[:data.top_k]:
        results.append({
            "id": row["id"],
            "name": row["name"],
            "ingredients": row["ingredients"],
            "img_src": row.get("img_src"),
            "missing_count": missing_count,
            "missing_ingredients": sorted(list(missing_tokens))
        })
    
    return results
