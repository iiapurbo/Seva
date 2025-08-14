# enhanced_clinical_config.py

import numpy as np

# ... (ATEC_QUESTIONS, QUESTION_ORDER, SUBSCALE_QUESTIONS remain the same) ...
ATEC_QUESTIONS = {
    # Speech/Language/Communication (14 questions, 0-2 scale)
    "Q1": {"subscale": "Speech/Language/Communication", "max_score": 2, "description": "Responds to name"},
    "Q2": {"subscale": "Speech/Language/Communication", "max_score": 2, "description": "Comes when called"},
    "Q3": {"subscale": "Speech/Language/Communication", "max_score": 2, "description": "Follows simple commands"},
    "Q4": {"subscale": "Speech/Language/Communication", "max_score": 2, "description": "Uses gestures to communicate"},
    "Q5": {"subscale": "Speech/Language/Communication", "max_score": 2, "description": "Nods for yes/shakes head for no"},
    "Q6": {"subscale": "Speech/Language/Communication", "max_score": 2, "description": "Can point to 5 body parts"},
    "Q7": {"subscale": "Speech/Language/Communication", "max_score": 2, "description": "Tries to sing or dance to music"},
    "Q8": {"subscale": "Speech/Language/Communication", "max_score": 2, "description": "Mimics you"},
    "Q9": {"subscale": "Speech/Language/Communication", "max_score": 2, "description": "Makes eye contact"},
    "Q10": {"subscale": "Speech/Language/Communication", "max_score": 2, "description": "Responds to emotions"},
    "Q11": {"subscale": "Speech/Language/Communication", "max_score": 2, "description": "Imitates words"},
    "Q12": {"subscale": "Speech/Language/Communication", "max_score": 2, "description": "Communicates needs"},
    "Q13": {"subscale": "Speech/Language/Communication", "max_score": 2, "description": "Looks at pictures in book"},
    "Q14": {"subscale": "Speech/Language/Communication", "max_score": 2, "description": "Babbles conversationally"},
    
    # Sociability (20 questions, 0-2 scale)
    "Q15": {"subscale": "Sociability", "max_score": 2, "description": "Knows own name"},
    "Q16": {"subscale": "Sociability", "max_score": 2, "description": "Responds to praise"},
    "Q17": {"subscale": "Sociability", "max_score": 2, "description": "Looks at people and animals"},
    "Q18": {"subscale": "Sociability", "max_score": 2, "description": "Comes to you when hurt/upset"},
    "Q19": {"subscale": "Sociability", "max_score": 2, "description": "Initiates peek-a-boo"},
    "Q20": {"subscale": "Sociability", "max_score": 2, "description": "Tries to please you"},
    "Q21": {"subscale": "Sociability", "max_score": 2, "description": "Cuddles"},
    "Q22": {"subscale": "Sociability", "max_score": 2, "description": "Kisses family members"},
    "Q23": {"subscale": "Sociability", "max_score": 2, "description": "Happy to see you"},
    "Q24": {"subscale": "Sociability", "max_score": 2, "description": "Shows affection"},
    "Q25": {"subscale": "Sociability", "max_score": 2, "description": "Shares or shows things"},
    "Q26": {"subscale": "Sociability", "max_score": 2, "description": "Plays with others"},
    "Q27": {"subscale": "Sociability", "max_score": 2, "description": "Plays peek-a-boo"},
    "Q28": {"subscale": "Sociability", "max_score": 2, "description": "Plays simple games"},
    "Q29": {"subscale": "Sociability", "max_score": 2, "description": "Plays with toys appropriately"},
    "Q30": {"subscale": "Sociability", "max_score": 2, "description": "Interested in other children"},
    "Q31": {"subscale": "Sociability", "max_score": 2, "description": "Responds to other children"},
    "Q32": {"subscale": "Sociability", "max_score": 2, "description": "Imitates other children"},
    "Q33": {"subscale": "Sociability", "max_score": 2, "description": "Plays appropriately with toys"},
    "Q34": {"subscale": "Sociability", "max_score": 2, "description": "Shares enjoyment with you"},
    
    # Sensory/Cognitive Awareness (18 questions, 0-2 scale)
    "Q35": {"subscale": "Sensory/Cognitive Awareness", "max_score": 2, "description": "Spins objects"},
    "Q36": {"subscale": "Sensory/Cognitive Awareness", "max_score": 2, "description": "Rocks back and forth"},
    "Q37": {"subscale": "Sensory/Cognitive Awareness", "max_score": 2, "description": "Shows sameness"},
    "Q38": {"subscale": "Sensory/Cognitive Awareness", "max_score": 2, "description": "Difficulty with changes"},
    "Q39": {"subscale": "Sensory/Cognitive Awareness", "max_score": 2, "description": "Visual fixations"},
    "Q40": {"subscale": "Sensory/Cognitive Awareness", "max_score": 2, "description": "Covers ears to sound"},
    "Q41": {"subscale": "Sensory/Cognitive Awareness", "max_score": 2, "description": "Mouths or licks objects"},
    "Q42": {"subscale": "Sensory/Cognitive Awareness", "max_score": 2, "description": "Touches everything"},
    "Q43": {"subscale": "Sensory/Cognitive Awareness", "max_score": 2, "description": "Spins self"},
    "Q44": {"subscale": "Sensory/Cognitive Awareness", "max_score": 2, "description": "Hurt self"},
    "Q45": {"subscale": "Sensory/Cognitive Awareness", "max_score": 2, "description": "Repetitive movements"},
    "Q46": {"subscale": "Sensory/Cognitive Awareness", "max_score": 2, "description": "Stares at nothing"},
    "Q47": {"subscale": "Sensory/Cognitive Awareness", "max_score": 2, "description": "Unaware of surroundings"},
    "Q48": {"subscale": "Sensory/Cognitive Awareness", "max_score": 2, "description": "Prefers to be alone"},
    "Q49": {"subscale": "Sensory/Cognitive Awareness", "max_score": 2, "description": "No fear of danger"},
    "Q50": {"subscale": "Sensory/Cognitive Awareness", "max_score": 2, "description": "Odd attachments to objects"},
    "Q51": {"subscale": "Sensory/Cognitive Awareness", "max_score": 2, "description": "Repetitive speech"},
    "Q52": {"subscale": "Sensory/Cognitive Awareness", "max_score": 2, "description": "Echoes words or phrases"},
    
    # Health/Physical/Behavior (25 questions, 0-3 scale)
    "Q53": {"subscale": "Health/Physical/Behavior", "max_score": 3, "description": "Wets bed"},
    "Q54": {"subscale": "Health/Physical/Behavior", "max_score": 3, "description": "Soils pants"},
    "Q55": {"subscale": "Health/Physical/Behavior", "max_score": 3, "description": "Diarrhea"},
    "Q56": {"subscale": "Health/Physical/Behavior", "max_score": 3, "description": "Constipation"},
    "Q57": {"subscale": "Health/Physical/Behavior", "max_score": 3, "description": "Sleep problems"},
    "Q58": {"subscale": "Health/Physical/Behavior", "max_score": 3, "description": "Eats limited foods"},
    "Q59": {"subscale": "Health/Physical/Behavior", "max_score": 3, "description": "Resists changes in routine"},
    "Q60": {"subscale": "Health/Physical/Behavior", "max_score": 3, "description": "Hits or injures self"},
    "Q61": {"subscale": "Health/Physical/Behavior", "max_score": 3, "description": "Hits or injures others"},
    "Q62": {"subscale": "Health/Physical/Behavior", "max_score": 3, "description": "Destructive"},
    "Q63": {"subscale": "Health/Physical/Behavior", "max_score": 3, "description": "Sound sensitive"},
    "Q64": {"subscale": "Health/Physical/Behavior", "max_score": 3, "description": "Anxious or fearful"},
    "Q65": {"subscale": "Health/Physical/Behavior", "max_score": 3, "description": "Unhappy or crying"},
    "Q66": {"subscale": "Health/Physical/Behavior", "max_score": 3, "description": "Seizures"},
    "Q67": {"subscale": "Health/Physical/Behavior", "max_score": 3, "description": "Obsessive speech"},
    "Q68": {"subscale": "Health/Physical/Behavior", "max_score": 3, "description": "Rigid thinking"},
    "Q69": {"subscale": "Health/Physical/Behavior", "max_score": 3, "description": "Demands sameness"},
    "Q70": {"subscale": "Health/Physical/Behavior", "max_score": 3, "description": "Often agitated"},
    "Q71": {"subscale": "Health/Physical/Behavior", "max_score": 3, "description": "Not sensitive to pain"},
    "Q72": {"subscale": "Health/Physical/Behavior", "max_score": 3, "description": "Hyperactive"},
    "Q73": {"subscale": "Health/Physical/Behavior", "max_score": 3, "description": "Lethargic"},
    "Q74": {"subscale": "Health/Physical/Behavior", "max_score": 3, "description": "Odd, repetitive behaviors"},
    "Q75": {"subscale": "Health/Physical/Behavior", "max_score": 3, "description": "Temper tantrums"},
    "Q76": {"subscale": "Health/Physical/Behavior", "max_score": 3, "description": "Difficulty concentrating"},
    "Q77": {"subscale": "Health/Physical/Behavior", "max_score": 3, "description": "Impulsive"}
}
QUESTION_ORDER = [f"Q{i}" for i in range(1, 78)]
SUBSCALE_QUESTIONS = {
    "Speech/Language/Communication": [f"Q{i}" for i in range(1, 15)],
    "Sociability": [f"Q{i}" for i in range(15, 35)],
    "Sensory/Cognitive Awareness": [f"Q{i}" for i in range(35, 53)],
    "Health/Physical/Behavior": [f"Q{i}" for i in range(53, 78)]
}

# --- DYNAMIC FOLDER STRUCTURE ---
BASE_DATA_DIR = "child_data"

def get_child_folder_structure(child_id: str) -> dict:
    """Generates a dictionary of paths for a specific child."""
    if not child_id or not isinstance(child_id, str) or "/" in child_id or "\\" in child_id:
        raise ValueError("Invalid child_id provided.")
    
    child_root = f"{BASE_DATA_DIR}/{child_id}"
    return {
        "child_root": child_root,
        "processed_logs": f"{child_root}/processed_logs",
        "anomaly_results": f"{child_root}/anomaly_results",
        "child_models": f"{child_root}/models",
    }

# Thresholds and Warnings remain the same
INDIVIDUAL_QUESTION_THRESHOLDS = {qid: 0.25 for qid in QUESTION_ORDER}
CLINICAL_WARNINGS = [
    "Individual question analysis with child-specific learning provides enhanced precision",
    "Results should be interpreted by qualified ASD professionals", 
    "This system is for screening and monitoring, not diagnosis",
    "Child-specific models improve over time with more data"
]