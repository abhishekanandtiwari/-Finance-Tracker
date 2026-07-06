"""
Transaction categorization engine.

Two layers of categorization:
1. Rule-based keyword matcher (fast, no training needed, works out of the box)
2. ML classifier (TF-IDF + Logistic Regression) trained on the user's own
   labeled/corrected transactions, which improves accuracy over time and
   handles merchants the rule list doesn't know about.

The ML model is optional: if there isn't enough labeled data yet, the
rule-based categorizer is used as a fallback.
"""

import re
import os
import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

MODEL_PATH = os.path.join(os.path.dirname(__file__), "category_model.joblib")

# ----------------------------------------------------------------------
# Rule-based keyword map: category -> list of keywords/regex fragments
# matched (case-insensitive) against the transaction description.
# ----------------------------------------------------------------------
CATEGORY_RULES = {
    "Groceries": ["walmart", "kroger", "trader joe", "whole foods", "safeway",
                  "grocery", "aldi", "costco", "supermarket", "big bazaar",
                  "reliance fresh", "dmart", "more supermarket"],
    "Dining & Restaurants": ["restaurant", "starbucks", "mcdonald", "cafe",
                             "coffee", "pizza", "swiggy", "zomato", "kfc",
                             "burger", "dominos", "doordash", "ubereats",
                             "bar ", "diner", "bistro"],
    "Transport": ["uber", "lyft", "ola ", "taxi", "metro", "fuel", "petrol",
                  "gas station", "shell", "bp ", "chevron", "parking",
                  "irctc", "railway", "toll", "indian oil"],
    "Shopping": ["amazon", "ebay", "flipkart", "myntra", "target", "best buy",
                 "mall", "store", "shop", "ikea", "etsy", "h&m", "zara"],
    "Utilities": ["electric", "electricity", "water bill", "gas bill",
                  "internet", "broadband", "wifi", "comcast", "verizon",
                  "at&t", "airtel", "jio", "vodafone", "utility", "power corp"],
    "Rent & Housing": ["rent", "landlord", "apartment", "mortgage", "housing society"],
    "Entertainment": ["netflix", "spotify", "hulu", "disney+", "movie",
                       "cinema", "theatre", "theater", "game", "steam",
                       "prime video", "hotstar", "youtube premium"],
    "Healthcare": ["pharmacy", "hospital", "clinic", "doctor", "medical",
                   "dental", "cvs", "walgreens", "health insurance", "apollo"],
    "Insurance": ["insurance", "premium payment", "lic ", "policybazaar"],
    "Education": ["tuition", "school", "college", "university", "course",
                  "udemy", "coursera", "books"],
    "Travel": ["airbnb", "hotel", "flight", "airlines", "booking.com",
               "makemytrip", "expedia", "indigo", "spicejet"],
    "Income / Salary": ["salary", "payroll", "deposit", "paycheck", "income",
                         "refund", "interest credit", "dividend"],
    "Transfers & Payments": ["transfer", "paypal", "venmo", "zelle", "upi",
                              "credit card payment", "loan payment", "emi"],
    "Subscriptions": ["subscription", "membership", "gym", "fitness"],
    "Fees & Charges": ["fee", "charge", "atm withdrawal", "service charge",
                        "late fee", "penalty"],
}

DEFAULT_CATEGORY = "Other / Uncategorized"


def rule_based_categorize(description: str) -> str:
    """Return a category for a single description using keyword matching."""
    if not isinstance(description, str):
        return DEFAULT_CATEGORY
    text = description.lower()
    for category, keywords in CATEGORY_RULES.items():
        for kw in keywords:
            if kw in text:
                return category
    return DEFAULT_CATEGORY


def _clean_text(s: str) -> str:
    s = str(s).lower()
    s = re.sub(r"[^a-z0-9\s]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


class MLCategorizer:
    """TF-IDF + Logistic Regression classifier trained on labeled transactions."""

    def __init__(self):
        self.pipeline = None

    def is_trained(self) -> bool:
        return self.pipeline is not None

    def train(self, descriptions, labels, min_samples_per_class: int = 2):
        """Train on a list of descriptions and their category labels."""
        df = pd.DataFrame({"desc": descriptions, "label": labels})
        df["desc"] = df["desc"].apply(_clean_text)
        counts = df["label"].value_counts()
        valid_labels = counts[counts >= min_samples_per_class].index
        df = df[df["label"].isin(valid_labels)]

        if df["label"].nunique() < 2 or len(df) < 4:
            # Not enough data / class diversity to train a meaningful model
            return False

        self.pipeline = Pipeline([
            ("tfidf", TfidfVectorizer(ngram_range=(1, 2), min_df=1)),
            ("clf", LogisticRegression(max_iter=1000)),
        ])
        self.pipeline.fit(df["desc"], df["label"])
        return True

    def predict(self, descriptions):
        if not self.is_trained():
            return [None] * len(descriptions)
        cleaned = [_clean_text(d) for d in descriptions]
        return self.pipeline.predict(cleaned)

    def predict_proba_max(self, descriptions):
        """Return the max class probability for each prediction (confidence)."""
        if not self.is_trained():
            return [0.0] * len(descriptions)
        cleaned = [_clean_text(d) for d in descriptions]
        proba = self.pipeline.predict_proba(cleaned)
        return proba.max(axis=1)

    def save(self, path: str = MODEL_PATH):
        if self.pipeline is not None:
            joblib.dump(self.pipeline, path)

    def load(self, path: str = MODEL_PATH) -> bool:
        if os.path.exists(path):
            self.pipeline = joblib.load(path)
            return True
        return False


def categorize_dataframe(df: pd.DataFrame, desc_col: str = "Description",
                          ml_model: MLCategorizer = None,
                          confidence_threshold: float = 0.45) -> pd.DataFrame:
    """
    Add a 'Category' column to df.
    Strategy: if a trained ML model is available and confident (>= threshold),
    use its prediction; otherwise fall back to rule-based keyword matching.
    """
    df = df.copy()
    rule_preds = df[desc_col].apply(rule_based_categorize)

    if ml_model is not None and ml_model.is_trained():
        ml_preds = ml_model.predict(df[desc_col].tolist())
        confidences = ml_model.predict_proba_max(df[desc_col].tolist())
        final = []
        for rule_cat, ml_cat, conf in zip(rule_preds, ml_preds, confidences):
            if conf >= confidence_threshold:
                final.append(ml_cat)
            else:
                final.append(rule_cat)
        df["Category"] = final
    else:
        df["Category"] = rule_preds

    return df
