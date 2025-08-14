from collections import defaultdict
import numpy as np
from enhanced_clinical_config import QUESTION_ORDER, ATEC_QUESTIONS
from llm_behavior_processor import LLMBehaviorProcessor
from behavioral_keywords import BEHAVIORAL_KEYWORD_MAPPING # Import the keywords for validation

# <<< MODIFICATION: EFFICIENCY >>>
# Instantiate the processor once at the module level to be reused.
# This avoids re-initializing the OpenAI client on every log analysis.
print("Initializing LLM Behavior Processor...")
LLM_PROCESSOR_SINGLETON = LLMBehaviorProcessor()
print("✅ LLM Behavior Processor is ready.")

def extract_question_level_behaviors(text):
    """
    The definitive version with a two-stage filter:
    1. Calls the powerful 24B LLM for nuanced detections.
    2. Uses a keyword-based "sanity check" to discard any potential hallucinations.
    """
    if not text or not text.strip():
        return {}
        
    print("🤖 Calling LLM for behavior extraction...")
    # Use the singleton instance instead of creating a new one.
    llm_detections = LLM_PROCESSOR_SINGLETON.process_log(text)
    # <<< END MODIFICATION >>>
    
    # --- The Sanity Check Filter ---
    validated_detections = []
    for detection in llm_detections:
        q_id = detection.get("question_id")
        evidence = detection.get("evidence_quote", "").lower()
        
        if q_id in BEHAVIORAL_KEYWORD_MAPPING:
            keywords = BEHAVIORAL_KEYWORD_MAPPING[q_id]
            if any(keyword in evidence for keyword in keywords):
                validated_detections.append(detection)
            else:
                print(f"⚠️ Discarding potential hallucination: '{q_id}' is not supported by evidence '{evidence}'.")
    
    print(f"✅ LLM processing complete. Found {len(validated_detections)} validated behaviors.")
    
    detected_questions = defaultdict(list)
    for detection in validated_detections:
        detected_questions[detection.get("question_id")].append(detection)
    return dict(detected_questions)

def compute_question_level_features(detected_questions):
    """This function correctly processes the validated detections."""
    feature_vector = np.zeros(len(QUESTION_ORDER))
    severity_map = {"no_problem": 0, "mild": 1, "moderate": 2, "severe": 3}

    for q_id, behaviors in detected_questions.items():
        if not behaviors: continue
        max_score_for_q = 0
        for behavior in behaviors:
            severity = behavior.get("severity", "").lower()
            score = severity_map.get(severity, 0)
            if score > max_score_for_q: max_score_for_q = score
        
        if max_score_for_q > 0:
            try:
                index = QUESTION_ORDER.index(q_id)
                max_possible_score = ATEC_QUESTIONS[q_id]["max_score"]
                clamped_score = min(max_score_for_q, max_possible_score)
                if max_possible_score > 0:
                    feature_vector[index] = float(clamped_score) / max_possible_score
            except ValueError:
                pass
    
    return feature_vector

def get_behavior_summary(detected_questions):
    return {"total_questions_with_behaviors": len(detected_questions)}