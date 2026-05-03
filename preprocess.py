"""
CICIDS 2017 — Data Cleaning & Preprocessing Pipeline
(creates training data for machine learning)

================================================
Input : cicids2017_cleaned.csv  (raw, 2.52M rows, 53 cols)
Output: X_scaled.npy            (float32, scaled features)
        y_encoded.npy           (int, encoded labels)
        feature_names.npy       (52 feature name strings)
        scaler.pkl              (fitted RobustScaler)
        label_encoder.pkl       (fitted LabelEncoder)

Label mapping
  0 = Bots         3 = DoS
  1 = Brute Force  4 = Normal Traffic
  2 = DDoS         5 = Port Scanning
                   6 = Web Attacks
"""
try:
    import numpy as np
    import pandas as pd
    import joblib
    import os.path
    from sklearn.preprocessing import RobustScaler, LabelEncoder
except Exception as e:
    print(f"Error loading imports: {e}\n Please install required imports and try again")

try:
    INPUT_CSV   = "cicids2017_cleaned.csv"
    CLEAN_CSV   = "cicids2017_cleaned_v2.csv"   # intermediate, kept for inspection
    SAMPLE_SIZE = 200_000   # rows used to fit scaler (saves memory on large datasets)
    CHUNK_SIZE  = 200_000   # rows processed at once during transform

    while not (os.path.isfile(INPUT_CSV)):
        print  ("File not found - Checking other directories...")
        alt_path = os.path.join("..", "data", INPUT_CSV)
        if os.path.isfile(alt_path):
            INPUT_CSV = alt_path
            print(f"File found: {INPUT_CSV}")
        else:
            print("cicids2017_cleaned.csv file does not exist in same folder nor the data folder\nPlease add the csv file to the working directory/folder.\n or add data to ~/intrusiongui/data/ (assuming this gets filed in ~/intrusiongui/gui)\n and try again")
            print("cicids2017_cleaned data set can be downloaded via https://www.kaggle.com/datasets/ericanacletoribeiro/cicids2017-cleaned-and-preprocessed")
            exit()

    def preprocess(INPUT_CSV="cicids2017_cleaned.csv", CLEAN_CSV="cicids2017_cleaned_v2.csv", SAMPLE_SIZE=200_000, CHUNK_SIZE=200_000):
        print("Starting preprocessing …\n")
        
    # ── 1. Load ────────────────────────────────────────────────────────────────────
    print("Loading …")
    print("This may take a few minutes")
    df = pd.read_csv(INPUT_CSV)
    print(f"  Original : {df.shape[0]:,} rows × {df.shape[1]} cols")

    # ── 2. Dedup ───────────────────────────────────────────────────────────────────
    df = df.drop_duplicates()
    print(f"  After dedup : {df.shape[0]:,} rows")

    # ── 3. Remove impossible negatives ────────────────────────────────────────────
    df = df[df["Flow Duration"] >= 0]
    print(f"  After removing negative Flow Duration : {df.shape[0]:,} rows")

    # Clamp to 0 — these columns cannot be negative
    CLAMP_COLS = [
        "Flow IAT Min",
        "Fwd IAT Min",
        "Fwd Header Length",
        "Bwd Header Length",
        "Flow Bytes/s",
        "Flow Packets/s",
        "min_seg_size_forward",
    ]
    for col in CLAMP_COLS:
        if col in df.columns:
            n_neg = (df[col] < 0).sum()
            if n_neg:
                df[col] = df[col].clip(lower=0)
                print(f"  Clamped {n_neg:,} negatives in '{col}'")

    # Note: Init_Win_bytes_forward / backward are intentionally left negative
    # (-1 is a TCP sentinel value for non-applicable flows)

    df.to_csv(CLEAN_CSV, index=False)
    print(f"\nClean CSV saved → {CLEAN_CSV}")
    print(f"Total rows removed: {2_520_751 - len(df):,}")

    # ── 4. Split features / labels ─────────────────────────────────────────────────
    LABEL_COL   = "Attack Type"
    feature_cols = [c for c in df.columns if c != LABEL_COL]

    y_raw = df[LABEL_COL].copy()
    le    = LabelEncoder()
    y_enc = le.fit_transform(y_raw)

    print("\nLabel encoding:")
    for idx, cls in enumerate(le.classes_):
        count = (y_enc == idx).sum()
        print(f"  {idx} = {cls:<20} ({count:>9,} rows)")

    # Cast to float32 — halves memory vs float64
    X = df[feature_cols].astype(np.float32)
    del df  # free original dataframe

    # ── 5. Scale (RobustScaler, memory-efficient) ──────────────────────────────────
    print("\nFitting RobustScaler on random sample …")
    rng        = np.random.default_rng(42)
    sample_idx = rng.choice(len(X), size=min(SAMPLE_SIZE, len(X)), replace=False)
    scaler     = RobustScaler()
    scaler.fit(X.iloc[sample_idx])

    print("Transforming full dataset in chunks …")
    chunks = []
    for start in range(0, len(X), CHUNK_SIZE):
        chunk = X.iloc[start : start + CHUNK_SIZE]
        chunks.append(scaler.transform(chunk).astype(np.float32))
        print(f"  {min(start + CHUNK_SIZE, len(X)):>9,} / {len(X):,}")

    X_scaled = np.vstack(chunks)
    del chunks

    # ── 6. Save artifacts ──────────────────────────────────────────────────────────
    np.save("X_scaled.npy",      X_scaled)
    np.save("y_encoded.npy",     y_enc)
    np.save("feature_names.npy", np.array(feature_cols))
    joblib.dump(scaler, "scaler.pkl")
    joblib.dump(le,     "label_encoder.pkl")

    print("\nArtifacts saved:")
    print(f"  X_scaled.npy      {X_scaled.shape}  {X_scaled.dtype}")
    print(f"  y_encoded.npy     {y_enc.shape}")
    print(f"  feature_names.npy ({len(feature_cols)} features)")
    print(f"  scaler.pkl")
    print(f"  label_encoder.pkl")
    print("\nPreprocessing complete.")
    
except Exception as e:
    import traceback
    print(f"An unexpected error has ocurred: {e}")
    print(traceback.format_exc()) 

