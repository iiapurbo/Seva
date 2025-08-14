import numpy as np
import pickle
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
from enhanced_clinical_config import QUESTION_ORDER, ATEC_QUESTIONS

class IncrementalChildLearningModel:
    """
    A stable, robust incremental learning model. This definitive version corrects
    all previous state management and initialization bugs.
    It now includes periodic retraining for improved long-term scalability.
    """
    # <<< MODIFICATION: SCALABILITY >>>
    # Only retrain the model every N logs to improve performance.
    TRAIN_INTERVAL = 5 
    
    def __init__(self, child_id, baseline_responses):
        self.child_id = child_id
        self.scaler = StandardScaler()
        self.anomaly_model = IsolationForest(contamination='auto', random_state=42)
        self.baseline_vector = self._normalize_baseline(baseline_responses)
        self.training_data = []
        self.scaler_fitted = False
        self.model_trained = False
        # Counter for periodic retraining
        self.logs_since_last_train = 0
        self.model_metadata = {"version": "final_stable_v4.1_scalable"}

    def process_new_log(self, processed_log):
        current_features = processed_log["question_level_features"]
        anomaly_result = self.detect_personalized_anomaly(current_features)
        
        self.training_data.append({"features": current_features})
        self.logs_since_last_train += 1 # Increment counter
        
        self.train_incremental_model()
        return anomaly_result

    def train_incremental_model(self):
        if not self.training_data: return
        
        X = np.array([sample["features"] for sample in self.training_data])
        
        # The scaler must be fitted on all available data to be consistent
        self.scaler.fit(X)
        self.scaler_fitted = True
        X_scaled = self.scaler.transform(X)
        
        # Condition to (re)train the model
        is_initial_training = not self.model_trained
        is_time_to_retrain = self.logs_since_last_train >= self.TRAIN_INTERVAL
        
        # Must have at least 2 samples to train IsolationForest
        if len(X_scaled) < 2:
            return

        # Train only if it's the first time or the interval has been reached
        if is_initial_training or is_time_to_retrain:
            print(f"🧠 Retraining anomaly model for child {self.child_id} (logs since last train: {self.logs_since_last_train})...")
            deviations = X_scaled - self.baseline_vector
            self.anomaly_model.fit(deviations)
            self.model_trained = True
            self.logs_since_last_train = 0 # Reset the counter
            print("✓ Model retraining complete.")
        # <<< END MODIFICATION >>>

    def detect_personalized_anomaly(self, current_features):
        if not self.scaler_fitted:
            return {"anomaly": False, "confidence": 0.0}
        current_vector = np.array(current_features).reshape(1, -1)
        current_scaled = self.scaler.transform(current_vector)
        deviation = current_scaled - self.baseline_vector
        is_pattern_outlier = False
        if self.model_trained:
            is_pattern_outlier = self.anomaly_model.predict(deviation)[0] == -1
        return {"anomaly": bool(is_pattern_outlier), "confidence": 1.0 if is_pattern_outlier else 0.0}

    def save_model(self, filepath):
        with open(filepath, 'wb') as f: pickle.dump(self, f)
        print(f"✓ Incremental model saved to {filepath}")

    @classmethod
    def load_model(cls, filepath):
        with open(filepath, 'rb') as f: model = pickle.load(f)
        # When loading, reset the counter to ensure it retrains on the next interval
        model.logs_since_last_train = 0
        print(f"✓ Incremental model loaded with {len(model.training_data)} training samples")
        return model

    def _normalize_baseline(self, baseline_responses):
        normalized = []
        for qid in QUESTION_ORDER:
            response = baseline_responses.get(qid, 0)
            max_score = ATEC_QUESTIONS[qid]["max_score"]
            normalized.append(response / max_score if max_score > 0 else 0)
        return np.array(normalized)