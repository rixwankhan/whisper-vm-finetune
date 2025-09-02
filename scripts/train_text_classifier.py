\
import argparse, pandas as pd, numpy as np, joblib, re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.calibration import CalibratedClassifierCV

IGNORE_WORDS = ["aaaaaaaa"]  # optional: keep ignoring these as requested

def norm(t: str) -> str:
    t = str(t)
    t = t.lower().strip()
    for w in IGNORE_WORDS:
        t = re.sub(rf"\\b{re.escape(w)}\\b", " ", t)
    t = re.sub(r"\\s+", " ", t)
    return t.strip()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True, help="CSV with columns: text,label (human/machine)")
    ap.add_argument("--out", required=True, help="Output joblib path")
    ap.add_argument("--word_ngrams", type=str, default="1,2", help="word n-gram range, e.g., 1,2")
    ap.add_argument("--char_ngrams", type=str, default="3,6", help="char n-gram range, e.g., 3,6")
    args = ap.parse_args()

    w_lo, w_hi = map(int, args.word_ngrams.split(","))
    c_lo, c_hi = map(int, args.char_ngrams.split(","))

    df = pd.read_csv(args.csv)
    assert "text" in df.columns and "label" in df.columns, "CSV must contain columns text,label"
    df["text"] = df["text"].fillna("").apply(norm)
    y = df["label"].map({"machine":0, "human":1}).to_numpy()

    # TF-IDF that will *implicitly* model phrases like "leave a" via word bigrams and char n-grams
    word_v = TfidfVectorizer(ngram_range=(w_lo, w_hi), min_df=2, max_df=0.95, analyzer="word")
    char_v = TfidfVectorizer(ngram_range=(c_lo, c_hi), min_df=2, max_df=0.95, analyzer="char")
    Xw = word_v.fit_transform(df["text"])
    Xc = char_v.fit_transform(df["text"])

    from scipy import sparse
    X = sparse.hstack([Xw, Xc])

    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.1, random_state=42, stratify=y)

    # Linear model with probability calibration for stable confidence
    base_clf = LogisticRegression(max_iter=2000, class_weight="balanced", n_jobs=-1)
    clf = CalibratedClassifierCV(base_clf, method="sigmoid", cv=3)
    clf.fit(Xtr, ytr)
    ypr = clf.predict(Xte)
    print(classification_report(yte, ypr, target_names=["machine","human"]))

    joblib.dump({"clf": clf, "word_v": word_v, "char_v": char_v}, args.out)
    print("Saved", args.out)

if __name__ == "__main__":
    main()
