"""
CICIDS 2017 — Chunked SGD Baseline Model
=========================================
Trains a memory-efficient linear classifier using SGDClassifier + partial_fit().
Reads X_scaled.npy from disk in chunks via memory-mapping — never loads the
full array into RAM.

Built by Jacob Amundsen with help from the internet, source code from ACMP, LLM magic, and a lot of coffee ☕️

Requirements
------------
    pip install scikit-learn numpy joblib

Inputs  (must be in the same directory)
-------
    X_scaled.npy        preprocessed features  (2.52M × 52, float32)
    y_encoded.npy       integer labels          (2.52M,)
    label_encoder.pkl   fitted LabelEncoder

Outputs
-------
    model_sgd.pkl       trained SGDClassifier
    results_sgd.txt     classification report
"""
try:
    import numpy as np
    import joblib
    import seaborn as sns
    import matplotlib.pyplot as plt
    import time
    from PyQt6.QtWidgets import QApplication
    from sklearn.linear_model import SGDClassifier
    from sklearn.metrics import classification_report
    from sklearn.model_selection import train_test_split
    from sklearn.utils.class_weight import compute_class_weight
except Exception as e:
    print(f"Error importing modules: {e}")

'''
After imports are loaded, the train_class verifies the files loaded correctly
'''

class train_model:
    def __init__(self, user_chunks_input=200000, seed=42, user_enochs_input=5, log=None) -> None:

        user_chunks_input: int
        user_enochs_input: int
        seed: int
        #
        self.CHUNK_SIZE  = user_chunks_input
        self.RANDOM_SEED = seed
        self.N_EPOCHS    = user_enochs_input
        self.TEST_SIZE   = 0.2
        self.tainingcompleteflag = False
        #
        self.y = None
        self.le = None
        self.classes = None
        self.class_weights = None
        self.class_weight_dict = None
        self.train_idx = None
        self.test_idx = None
        self.X_mmap = None
        self.y_test = None
        self.y_pred = None
        
    
    def load_files(self, y_encoded, label_encoder, log=print) -> None:

        y_encoded: str
        label_encoder: str

        try:
            self.y  = np.load(y_encoded)  # loads entire array into RAM — small enough to fit (2.52M int labels ~ 10MB)
            self.le = joblib.load(label_encoder)
            self.classes = np.unique(self.y)
            log(f"  Total samples : {len(self.y):,}")
            log(f"  Classes       : {list(self.le.classes_)}\n")
        except Exception as e:
            import traceback
            log(f"Error while loading files: {e}")
            log(traceback.format_exc())  # prints the full stack trace to GUI log


    '''
    The next function computes the class weights for the model and splits the data into training & testing sets.
    The weights are used to penalise the model more for misclassifying rarer classes, which helps with class imbalance.
    '''

    def compute_weights(self, log=print) -> None:

        try:
            self.class_weights = compute_class_weight('balanced', classes=self.classes, y=self.y)
            weights          = compute_class_weight("balanced", classes=self.classes, y=self.y)
            self.class_weight_dict = dict(zip(self.classes, weights))

            # Creates a dictionary mapping class labels to their corresponding weights

            log("Computing class weights ; (higher = rarer class penalised more):")
            QApplication.processEvents() # < - allows GUI to update before continuing with training
            for cls, w in zip(self.le.classes_, weights):
                log(f"  {cls:<22} {w:.4f}")

            # ── Train / test split (indices only — no data loaded yet) ────────────────────

            log("\nSplitting indices …")
            QApplication.processEvents() # < - allows GUI to update before continuing with training
            idx = np.arange(len(self.y))

            train_idx, test_idx = train_test_split(
                idx, test_size=self.TEST_SIZE, stratify=self.y, random_state=self.RANDOM_SEED
            )

            # Sort indices so memory-mapped reads are sequential (much faster)

            self.train_idx = train_idx[np.argsort(train_idx)]
            self.test_idx  = test_idx[np.argsort(test_idx)]

            log(f"Ready to begin training\n  Train : {len(self.train_idx):,}  |  Test : {len(self.test_idx):,}")

        except Exception as e:
            import traceback
            log(f"Error while computing weights: {e}")
            log(traceback.format_exc())  # prints the full stack trace to GUI log


    '''
    The train function implements the actual training loop.
    It initializes an SGDClassifier with logistic regression loss and iteratively calls partial_fit() on chunks of the training data.
    The model is trained for a specified number of epochs, and the training data is shuffled each epoch for better convergence.
    '''

    def train(self, X_scaled, log=print) -> None:

        X_scaled: str

        try:
            log("\nInitializing model …")
            self.model = SGDClassifier(
                loss="log_loss",  # logistic regression
                max_iter=1,       # epochs handled manually
                warm_start=True,  # keep model between .fit() calls
                class_weight=self.class_weight_dict,
                random_state=self.RANDOM_SEED
            )
            log(f"\nTraining ({self.N_EPOCHS} epoch(s), chunk size {self.CHUNK_SIZE:,}) …")

            QApplication.processEvents() # < - allows GUI to update before continuing with training

            self.X_mmap = np.load(X_scaled, mmap_mode="r")   # memory-mapped — stays on disk
            t0 = time.time()

            for epoch in range(1, self.N_EPOCHS + 1):
                log(f"\n  Epoch {epoch}/{self.N_EPOCHS}")
                # Shuffle train indices each epoch for better SGD convergence
                rng = np.random.default_rng(self.RANDOM_SEED + epoch)
                shuffled = rng.permutation(self.train_idx)

                for start in range(0, len(shuffled), self.CHUNK_SIZE):
                    chunk_idx    = shuffled[start : start + self.CHUNK_SIZE]
                    X_chunk      = self.X_mmap[chunk_idx]
                    y_chunk      = self.y[chunk_idx]
                    sample_weight = np.array([self.class_weight_dict[lbl] for lbl in y_chunk])

                    self.model.partial_fit(X_chunk, y_chunk, classes=self.classes, sample_weight=sample_weight)

                    done = min(start + self.CHUNK_SIZE, len(shuffled))
                    log(f"    {done:>9,} / {len(shuffled):,}")

                log(f"    {len(shuffled):,} / {len(shuffled):,}  ✓")

                QApplication.processEvents() # < - allows GUI to update before continuing with training

            log(f"\nTraining complete in {(time.time()-t0)/60:.1f} min")
            self.tainingcompleteflag = True

        except Exception as e:
            import traceback
            log(f"Error in training function: {e}")
            log(traceback.format_exc())  # prints the full stack trace to GUI log

    '''
    After training is complete, the evaluate_results function is called to assess the model's permormance on the test set.
    It generates a classification report and stores the predicted labels for later use in plotting the confusion matrix and F1 score chart.
    '''    
    
    def evaluate_results(self, log=print) -> (str | dict):
        try:
            log(f"\nEvaluating model on test set …")
            X_test = self.X_mmap[self.test_idx]
            self.y_test = self.y[self.test_idx]
            self.y_pred = self.model.predict(X_test)
            report = classification_report(self.y_test, self.y_pred, target_names=self.le.classes_)
            log(report)
        except Exception as e:
            import traceback
            log(f"Error while evaluating results: {e}")
            log(traceback.format_exc())  # prints the full stack trace to GUI log


    ''' The next two functions generate the confusion matrix and F1 score bar chart, respectively, using seaborn and matplotlib
        These plots show the model's performance across different classes, highlighting areas where it may be struggling (e.g., low F1 scores or confusion between certain classes).
    '''

    def plot_f1_bars(self, log=print) -> tuple:
        try:
            log("Generating F1 score chart...")
            report = classification_report(self.y_test, self.y_pred, 
                                        target_names=self.le.classes_, 
                                        output_dict=True)
            classes = self.le.classes_
            f1_scores = [report[c]['f1-score'] for c in classes]
            
            fig, ax = plt.subplots(figsize=(10, 5))
            bars = ax.barh(classes, f1_scores, color='steelblue')
            ax.set_xlim(0, 1)
            ax.set_xlabel("F1 Score")
            ax.set_title("F1 Score per Class")
            ax.bar_label(bars, fmt='%.2f', padding=3)
            plt.tight_layout()
            return fig
        except Exception as e:
            import traceback
            log(f"Error while plotting F1 score chart: {e}")
            log(traceback.format_exc())  # prints the full stack trace to GUI log


    
        