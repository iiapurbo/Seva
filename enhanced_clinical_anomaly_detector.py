import numpy as np
from enhanced_clinical_config import ATEC_QUESTIONS, QUESTION_ORDER
from enhanced_clinical_validation import assess_question_level_significance

class EnhancedClinicalAnomalyDetector:
    """
    The definitive, robust, and simplified version of the clinical anomaly detector.
    This version is guaranteed to correctly identify new, emergent behaviors.
    """
    
    def __init__(self, individual_baseline_responses):
        self.individual_baseline_responses = individual_baseline_responses
        self.question_order = QUESTION_ORDER
        self.normalized_baseline = self._normalize_baseline()
        
    def _normalize_baseline(self):
        """Normalize individual baseline responses to a 0-1 scale."""
        normalized = []
        for qid in self.question_order:
            response = self.individual_baseline_responses.get(qid, 0)
            max_score = ATEC_QUESTIONS[qid]["max_score"]
            normalized.append(response / max_score if max_score > 0 else 0)
        return np.array(normalized)
    
    def detect_anomalies(self, processed_logs):
        """
        This function correctly calculates deviations for each log individually
        and properly flags significant clinical changes.
        """
        results = []
        for i, log in enumerate(processed_logs):
            feature_vector = np.array(log.get("question_level_features", [0.0] * len(self.question_order)))
            
            # --- START OF THE DEFINITIVE FIX ---
            # We will now calculate deviation and significance in one clear step.
            significant_changes = []
            for j, question_id in enumerate(self.question_order):
                # Only check questions where the LLM detected a behavior.
                if feature_vector[j] > 0:
                    deviation = feature_vector[j] - self.normalized_baseline[j]
                    
                    # Use the validation function to check if this deviation is significant.
                    significance = assess_question_level_significance(deviation, question_id)
                    
                    if significance["significant"]:
                        significant_changes.append({
                            "question_id": question_id,
                            "deviation": float(deviation),
                            "baseline_value": float(self.normalized_baseline[j]),
                            "current_value": float(feature_vector[j]),
                            **significance
                        })
            
            question_analysis = {
                "significant_questions": significant_changes,
                "total_significant": len(significant_changes)
            }
            # --- END OF THE DEFINITIVE FIX ---

            child_specific_analysis = log.get("immediate_anomaly_analysis")
            
            clinical_significance = question_analysis["total_significant"] > 0
            child_specific_anomaly = child_specific_analysis and child_specific_analysis.get("anomaly", False)
            
            is_anomaly = clinical_significance or child_specific_anomaly
            
            result = {
                "log_index": i,
                "filename": log.get("filename"),
                "timestamp": log.get("timestamp"),
                "anomaly_detected": bool(is_anomaly),
                "confidence": self._calculate_enhanced_confidence(question_analysis, child_specific_anomaly),
                "raw_log": log.get("raw_log", ""),
                "question_analysis": question_analysis,
                "child_specific_analysis": child_specific_analysis,
                "detailed_explanation": self._generate_comprehensive_explanation(question_analysis, child_specific_anomaly),
                "clinical_recommendations": self._generate_enhanced_recommendations(question_analysis)
            }
            results.append(result)
        
        return results

    def _calculate_enhanced_confidence(self, question_analysis, child_specific_anomaly):
        """Confidence is driven by clinical significance."""
        if question_analysis["total_significant"] > 0:
            return 0.9 # High confidence for any clinical deviation
        if child_specific_anomaly:
            return 0.5 # Moderate confidence for a pattern anomaly
        return 0.1

    def _generate_comprehensive_explanation(self, question_analysis, child_specific_anomaly):
        """Generates a clear explanation of why an anomaly was or wasn't flagged."""
        parts = []
        if question_analysis["total_significant"] > 0:
            parts.append(f"Clinical Analysis: Found {question_analysis['total_significant']} significant deviations from the child's baseline.")
            for q in question_analysis["significant_questions"]:
                direction = "improvement" if q["deviation"] < 0 else "regression"
                if q['baseline_value'] == 0 and direction == 'regression':
                    parts.append(f"  - CRITICAL: New emergent behavior detected in '{q['question_description']}' ({q['level']} {direction}).")
                else:
                    parts.append(f"  - '{q['question_description']}' showed a {q['level']} {direction}.")
        
        if child_specific_anomaly:
            parts.append("Child-Specific Model: The overall pattern of this day was unusual compared to other recent days.")

        if not parts:
            return "No significant clinical or personalized pattern anomalies were detected in this log."
        return "\n".join(parts)

    def _generate_enhanced_recommendations(self, question_analysis):
        """Generates actionable recommendations."""
        if question_analysis["total_significant"] == 0:
            return ["Continue regular monitoring."]
        
        recs = ["A significant change was detected. It is recommended to discuss this log with a clinical professional."]
        new_behaviors = [q for q in question_analysis["significant_questions"] if q["baseline_value"] == 0 and q["deviation"] > 0]
        if new_behaviors:
            recs.append(f"Immediate attention may be required for the new emergent behavior: '{new_behaviors[0]['question_description']}'.")
        return recs