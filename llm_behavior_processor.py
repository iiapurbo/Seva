import os
import json
import openai
from dotenv import load_dotenv
from enhanced_clinical_config import ATEC_QUESTIONS

# --- User Configuration ---
# This is set back to the model you specified.
OPENROUTER_MODEL = "mistralai/mistral-small-3.2-24b-instruct:free"
HTTP_REFERER = "http://localhost"
APP_NAME = "ASD Anomaly Detector"

class LLMBehaviorProcessor:
    """
    The definitive version for OpenRouter, with a stricter prompt and a 'no_problem' option
    to prevent the LLM from hallucinating problems.
    """
    def __init__(self):
        load_dotenv() 
        api_key = os.getenv("OPENROUTER_API_KEY")
        if not api_key:
            raise ValueError("OPENROUTER_API_KEY not found. Make sure it's in your .env file.")

        self.client = openai.OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=api_key,
            default_headers={"HTTP-Referer": HTTP_REFERER, "X-Title": APP_NAME},
        )

    def _build_prompt(self, parent_log_text):
        """Constructs the definitive, stricter prompt for the LLM."""
        
        atec_questions_formatted = "\n".join([f'- {qid}: {q_data["description"]}' for qid, q_data in ATEC_QUESTIONS.items()])

        prompt = f"""
        You are a highly-trained clinical assistant. Your task is to analyze a parent's journal entry and classify observed behaviors with extreme precision.

        **CRITICAL INSTRUCTIONS:**
        1.  **BE LITERAL AND CONSERVATIVE:** Only map a behavior if there is a CLEAR and DIRECT mention in the text. DO NOT generalize or infer. For example, 'interacted with museum exhibits' is NOT the same as 'plays with toys'. If there is no direct match, DO NOT report it.
        2.  **HANDLE NEGATION:** If the parent says a behavior did NOT happen (e.g., "no meltdown"), DO NOT report it.
        3.  **CLASSIFY SEVERITY:** For each identified behavior, classify it using one of four levels:
            - "no_problem": Use this if the behavior is mentioned in a POSITIVE, NEUTRAL, or AGE-APPROPRIATE context (e.g., "played nicely", "happy hand-flapping").
            - "mild": For minor, brief, or occasional problems.
            - "moderate": For typical, sustained, or standard problems.
            - "severe": For intense, extreme, or highly disruptive problems.
        4.  **Output JSON:** Your output MUST be a single, valid JSON object with a root key "detections".

        **PARENT'S LOG:**
        ---
        {parent_log_text}
        ---

        **ATEC QUESTIONNAIRE REFERENCE:**
        ---
        {atec_questions_formatted}
        ---

        **JSON OUTPUT STRUCTURE:**
        {{
          "detections": [
            {{
                "question_id": "string (e.g., 'Q45')",
                "severity": "string ('no_problem', 'mild', 'moderate', or 'severe')",
                "reasoning": "string (brief justification)",
                "evidence_quote": "string (direct quote from the log)"
            }}
          ]
        }}

        Now, generate the JSON output based on these strict rules.
        """
        return prompt

    def process_log(self, parent_log_text):
        if not parent_log_text or not parent_log_text.strip(): return []
        prompt = self._build_prompt(parent_log_text)
        try:
            response = self.client.chat.completions.create(
                model=OPENROUTER_MODEL,
                messages=[{'role': 'system', 'content': 'You are a clinical assistant outputting JSON.'}, {'role': 'user', 'content': prompt}],
                response_format={"type": "json_object"},
                temperature=0.0
            )
            response_content = response.choices[0].message.content
            json_data = json.loads(response_content)
            detections = json_data.get("detections", [])
            if not isinstance(detections, list): return []
            return [d for d in detections if "question_id" in d and "severity" in d]
        except Exception as e:
            print(f"🔴 Error calling OpenRouter API or parsing response: {e}")
            return []