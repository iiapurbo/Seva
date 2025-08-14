import numpy as np
from enhanced_clinical_config import ATEC_QUESTIONS, QUESTION_ORDER, SUBSCALE_QUESTIONS

def validate_individual_atec_baseline(individual_responses):
    """Validate individual ATEC question responses."""
    validation_results = {
        "valid": True,
        "errors": [],
        "warnings": [],
        "summary": {}
    }
    
    # Check all questions are present
    missing_questions = []
    for question_id in QUESTION_ORDER:
        if question_id not in individual_responses:
            missing_questions.append(question_id)
    
    if missing_questions:
        validation_results["valid"] = False
        validation_results["errors"].append(f"Missing questions: {missing_questions}")
    
    # Validate score ranges
    for question_id, response in individual_responses.items():
        if question_id in ATEC_QUESTIONS:
            max_score = ATEC_QUESTIONS[question_id]["max_score"]
            
            if not isinstance(response, (int, float)) or not (0 <= response <= max_score):
                validation_results["valid"] = False
                validation_results["errors"].append(
                    f"Invalid score for {question_id}: {response} (must be 0-{max_score})"
                )
    
    # Calculate subscale totals
    subscale_totals = {}
    for subscale, questions in SUBSCALE_QUESTIONS.items():
        total = sum(individual_responses.get(q, 0) for q in questions)
        subscale_totals[subscale] = total
    
    validation_results["summary"] = {
        "total_questions": len(QUESTION_ORDER),
        "responses_provided": len(individual_responses),
        "subscale_totals": subscale_totals,
        "overall_total": sum(subscale_totals.values())
    }
    
    return validation_results

def normalize_individual_baseline(individual_responses):
    """Normalize individual question responses to 0-1 scale."""
    normalized_vector = []
    
    for question_id in QUESTION_ORDER:
        response = individual_responses.get(question_id, 0)
        max_score = ATEC_QUESTIONS[question_id]["max_score"]
        normalized = response / max_score if max_score > 0 else 0
        normalized_vector.append(normalized)
    
    return np.array(normalized_vector)

def assess_question_level_significance(deviation, question_id):
    """Assess clinical significance of deviation for individual question."""
    from enhanced_clinical_config import INDIVIDUAL_QUESTION_THRESHOLDS
    
    threshold = INDIVIDUAL_QUESTION_THRESHOLDS.get(question_id, 0.25)
    abs_deviation = abs(deviation)
    
    if abs_deviation < threshold:
        return {
            "significant": False,
            "level": "normal",
            "description": "Within normal variation",
            "question_description": ATEC_QUESTIONS[question_id]["description"]
        }
    elif abs_deviation < threshold * 2:
        return {
            "significant": True,
            "level": "mild",
            "description": "Mild deviation from baseline",
            "question_description": ATEC_QUESTIONS[question_id]["description"]
        }
    elif abs_deviation < threshold * 3:
        return {
            "significant": True,
            "level": "moderate",
            "description": "Moderate deviation from baseline",
            "question_description": ATEC_QUESTIONS[question_id]["description"]
        }
    else:
        return {
            "significant": True,
            "level": "severe",
            "description": "Severe deviation from baseline",
            "question_description": ATEC_QUESTIONS[question_id]["description"]
        }
