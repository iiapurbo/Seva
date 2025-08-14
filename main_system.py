# main_system.py

import json
from datetime import datetime
from pathlib import Path
import numpy as np

# Import the new dynamic path generator
from enhanced_clinical_config import get_child_folder_structure
from enhanced_behavior_extraction import extract_question_level_behaviors, compute_question_level_features
from enhanced_clinical_validation import validate_individual_atec_baseline
from child_specific_learning import IncrementalChildLearningModel
from enhanced_clinical_anomaly_detector import EnhancedClinicalAnomalyDetector

class IncrementalBehavioralMonitoringSystem:
    """
    A multi-tenant behavioral monitoring system. Each instance is tied to a specific child.
    """
    def __init__(self, child_id: str):
        if not child_id:
            raise ValueError("A child_id must be provided to initialize the system.")
        self.child_id = child_id
        self.child_folders = get_child_folder_structure(child_id)
        self.child_model = None
        self.baseline_data = None
        self.is_ready = False
        self._initialize_folders()

    def _initialize_folders(self):
        """Initializes all necessary folders for the specific child."""
        for folder_path in self.child_folders.values():
            Path(folder_path).mkdir(parents=True, exist_ok=True)
        print(f"✓ Folder structure verified for child: {self.child_id}")

    def set_baseline(self, baseline_payload: dict):
        """Sets the baseline data, validates it, and initializes the child-specific model."""
        validation = validate_individual_atec_baseline(baseline_payload["individual_responses"])
        if not validation["valid"]:
            raise ValueError(f"Invalid baseline data: {validation['errors']}")
        
        self.baseline_data = {
            "individual_responses": baseline_payload["individual_responses"],
            "child_id": self.child_id
        }
        
        # <<< MODIFICATION: API STATEFULNESS >>>
        # Save the baseline data to disk so it can be reloaded on API restart.
        baseline_path = Path(self.child_folders["child_root"]) / "baseline_data.json"
        with open(baseline_path, 'w', encoding='utf-8') as f:
            json.dump(baseline_payload, f, indent=2)
        print(f"✓ Baseline data saved to {baseline_path} for child: {self.child_id}")
        # <<< END MODIFICATION >>>

        self.child_model = IncrementalChildLearningModel(
            self.child_id,
            self.baseline_data["individual_responses"]
        )
        print(f"✓ New incremental learning model initialized for child {self.child_id}.")
        self.is_ready = True
        print("✅ System is ready for log analysis.")

    def analyze_log(self, log_entry: dict):
        """Analyzes a single log entry for the specific child."""
        if not self.is_ready:
            raise RuntimeError("System is not ready. A valid baseline must be set.")

        print(f"\n▶️  Analyzing new log for child {self.child_id} (timestamp: {log_entry.get('timestamp')})")
        
        filename = f"log_{log_entry.get('timestamp', datetime.now().isoformat()).replace(':', '-')}.json"
        processed_log = self._process_single_log(log_entry, filename)
        
        # The child model is now an instance variable, so it maintains its state
        processed_log["immediate_anomaly_analysis"] = self.child_model.process_new_log(processed_log)
        
        detector = EnhancedClinicalAnomalyDetector(self.baseline_data["individual_responses"])
        final_analysis = detector.detect_anomalies([processed_log])[0]

        self._display_single_log_report(final_analysis)
        
        # Save artifacts to the child's specific folders
        processed_logs_dir = Path(self.child_folders["processed_logs"])
        with open(processed_logs_dir / f"processed_{filename}", 'w', encoding='utf-8') as f:
            json.dump(processed_log, f, indent=2)
        
        model_path = Path(self.child_folders["child_models"]) / "incremental_child_model.pkl"
        self.child_model.save_model(model_path)
        
        self._save_single_anomaly_result(final_analysis)
        
        print(f"\n✅ Analysis complete for child {self.child_id}!")
        return final_analysis

    def _process_single_log(self, log_entry, filename):
        """Processes a single raw log into a structured dictionary with features."""
        raw_log = log_entry.get("log", "")
        detected_behaviors = extract_question_level_behaviors(raw_log)
        feature_vector = compute_question_level_features(detected_behaviors)

        processed_log = {
            "child_id": self.child_id,
            "filename": filename,
            "timestamp": log_entry.get("timestamp", datetime.now().isoformat()),
            "raw_log": raw_log,
            "question_level_behaviors": detected_behaviors,
            "question_level_features": feature_vector.tolist() if isinstance(feature_vector, np.ndarray) else feature_vector,
        }
        return processed_log

    def _save_single_anomaly_result(self, result):
        results_dir = Path(self.child_folders["anomaly_results"])
        result_filename = f"analysis_{Path(result['filename']).stem}.json"
        with open(results_dir / result_filename, 'w') as f: json.dump(result, f, indent=2, ensure_ascii=False)
        print(f"✓ Detailed analysis saved to: {results_dir / result_filename}")
        
    def _display_single_log_report(self, result):
        print("\n--- Analysis Report ---")
        print(f"Child ID: {self.child_id}")
        print(f"Log File: {result['filename']} ({result['timestamp']})")
        print("-" * 25)
        if result['anomaly_detected']: print(f"🚨 ANOMALY DETECTED (Confidence: {result['confidence']:.2f})")
        else: print("✅ NO ANOMALY DETECTED")
        print("\n📋 Detailed Explanation:")
        print(result['detailed_explanation'])
        print("\n💡 Recommendations:")
        for rec in result['clinical_recommendations']: print(f"- {rec}")
        print("-" * 25)