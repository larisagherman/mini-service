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
from symspellpy import SymSpell, Verbosity

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
    text = re.sub(r"\b(cup|cups|tbsp|tsp|tablespoon|tablespoons|teaspoon|teaspoons|ounce|oz|grams|g|kg|pound|lb)\b", "", text, flags=re.IGNORECASE)
    text = re.sub(r"[^a-zA-Z\s]", "", text)
    text = text.lower().strip()
    tokens = [
        lemmatizer.lemmatize(t)
        for t in text.split()
        if t not in stop_words and t not in PREPARATION_WORDS
    ]
    return " ".join(tokens)
# --------------------
# Build known ingredient vocabulary
# --------------------
ALL_INGREDIENTS = set()

for ing_list in df["ingredients"]:
    for ing in re.split(r'[\n,]', str(ing_list)):
        norm = normalize_ingredient(ing)
        if norm:
            ALL_INGREDIENTS.add(norm)

ALL_INGREDIENTS = list(ALL_INGREDIENTS)

sym_spell = SymSpell(max_dictionary_edit_distance=2, prefix_length=7)

for word in ALL_INGREDIENTS:
    sym_spell.create_dictionary_entry(word, 1)


def symspell_correct(token: str) -> str:
    suggestions = sym_spell.lookup(
        token,
        Verbosity.CLOSEST,
        max_edit_distance=2
    )

    if suggestions:
        return suggestions[0].term

    return token

 
def tokenize(text: str) -> set:
    tokens = set()
    for line in re.split(r'[\n,]', text):
        normalized = normalize_ingredient(line)
        if normalized:
            tokens.add(normalized)   # 🔥 add whole ingredient
    return tokens
# Accept either a string (from user) or a list (from Gemini)
def parse_user_query(query) -> set:
    tokens = []

    if isinstance(query, str):
        # Split string by commas or spaces
        items = re.split(r'[\n,]', query)
        items = [i.strip() for i in items if i.strip()]
    elif isinstance(query, list):
        # Already a list from Gemini
        items = query
    else:
        items = []

    for item in items:
        raw = item.lower().strip()

        corrected = symspell_correct(raw)
        normalized = normalize_ingredient(corrected)

        if normalized:
            tokens.append(normalized)

    return set(tokens)
def get_recipe_tokens(recipe_ingredients: str) -> set:
    """Return set of normalized tokens for a recipe"""
    tokens = set()
    for line in re.split(r'[\n,]', recipe_ingredients):
        line = line.strip()
        if line:
            corrected = symspell_correct(line.lower()) 
            normalized = normalize_ingredient(line)
            if normalized:
                tokens.add(normalized)
    return tokens

def clean_text(text: str) -> str:
    """Simple cleaning for TF-IDF"""
    return " ".join([normalize_ingredient(word) for word in text.split()])

df["tokens"] = df["ingredients"].apply(tokenize)
df["clean_text"] = df["name"].apply(normalize_ingredient) + " " + df["ingredients"].apply(lambda x: " ".join(tokenize(x)))
# --------------------
# TF-IDF model
# --------------------
vectorizer = TfidfVectorizer(stop_words="english")
tfidf_matrix = vectorizer.fit_transform(df["clean_text"])

# --------------------
# Schemas
# --------------------
class RecommendationRequest(BaseModel):
    query: list[str]
    top_k: int
    forbidden_ingredients: list[str] = []
    strict: bool = False

class RecommendationResponse(BaseModel):
    id: int
    name: str
    ingredients: str
    img_src: Optional[str] = None
    missing_count: Optional[int] = None
    missing_ingredients: Optional[list[str]] = None


def process_user_input(query_list):
    tokens = set()

    for item in query_list:
        tokens_raw = re.split(r"\s+", item.lower().strip())

        for token in tokens_raw:
            corrected = symspell_correct(token)
            normalized = normalize_ingredient(corrected)

            if normalized:
                tokens.add(normalized)
    return tokens

# --------------------
# Endpoint
# --------------------
@app.post("/recommend", response_model=list[RecommendationResponse])
def recommend(data: RecommendationRequest):
    # 1️⃣ Parse ingredients (handles list or string)
    user_tokens = process_user_input(data.query)
    
    # 🔍 DEBUG OUTPUT (ADD THIS)
    print("RAW INPUT:", data.query)
    print("PROCESSED TOKENS (after fuzzy + normalization):", user_tokens)
    print("SYM SPELL TEST:", sym_spell.lookup("chery", Verbosity.CLOSEST, max_edit_distance=2))    
    query_string = " ".join(user_tokens)
    query_vec = vectorizer.transform([query_string])
    tfidf_scores = cosine_similarity(query_vec, tfidf_matrix)[0]

    # 2️⃣ Candidate selection
    top_idx = tfidf_scores.argsort()[-(data.top_k*5):][::-1]
    candidates = df.iloc[top_idx].copy()

    # 3️⃣ Filter forbidden ingredients safely
    if data.forbidden_ingredients:
        pattern = "|".join(re.escape(i) for i in data.forbidden_ingredients)
        candidates = candidates[~candidates["ingredients"].str.contains(pattern, case=False, regex=True)]

    # 4️⃣ Calculate missing ingredients & score
    recipe_scores = []
    exact_match_found = False

    for idx in candidates.index:
        row = df.loc[idx]
        recipe_tokens = get_recipe_tokens(row["ingredients"])
        missing_tokens = recipe_tokens - user_tokens
        missing_count = len(missing_tokens)

        if missing_count == 0:
            exact_match_found = True

        if data.strict and missing_count > 0:
            continue

        score = tfidf_scores[idx]
        recipe_scores.append((score, missing_count, missing_tokens, row))

    # 5️⃣ Strict fallback
    if data.strict and not exact_match_found:
        recipe_scores = []
        for idx in candidates.index:
            row = df.loc[idx]
            recipe_tokens = get_recipe_tokens(row["ingredients"])
            missing_tokens = recipe_tokens - user_tokens
            missing_count = len(missing_tokens)
            score = tfidf_scores[idx]
            recipe_scores.append((score, missing_count, missing_tokens, row))

    # 6️⃣ Sorting
    if data.strict and not exact_match_found:
        # prioritize fewest missing ingredients
        recipe_scores.sort(key=lambda x: (x[1], -x[0]))
    else:
        recipe_scores.sort(key=lambda x: (-x[0], x[1]))

    # 7️⃣ Prepare results
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
