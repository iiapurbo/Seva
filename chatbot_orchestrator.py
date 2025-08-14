# chatbot_orchestrator.py
from openai import OpenAI
from typing import Dict, Optional, Tuple
import config
from knowledge_base_manager import BookKnowledgeBase
from anomaly_data_manager import AnomalyDataManager
import requests
import json
from datetime import datetime
import asyncio # Added for streaming

class Chatbot:
    """
    The main chatbot orchestrator. It uses a router-retriever-generator pattern
    to provide safe, grounded, and child-specific answers.
    VERSION 3.0: Includes a detailed Interpretation Guide in the prompt for robust, expert-level explanations.
    NOW WITH STREAMING SUPPORT.
    """
    def __init__(self, child_id: str, book_kb: BookKnowledgeBase):
        self.child_id = child_id
        self.book_kb = book_kb
        self.anomaly_manager = AnomalyDataManager()
        
        if not config.OPENROUTER_API_KEY:
            raise ValueError("OPENROUTER_API_KEY not found in .env file.")
            
        self.client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=config.OPENROUTER_API_KEY,
            default_headers={"HTTP-Referer": config.HTTP_REFERER, "X-Title": config.APP_NAME},
        )

    def _route_query(self, query: str) -> str:
        """Uses an LLM to classify the user's intent."""
        prompt = f"""
        You are a query routing assistant for a chatbot that helps parents of children with autism.
        Classify the user's question into one of four categories based on its content:

        1. `ANOMALY_REPORT`: The user is asking about their child's most recent state, the very latest update, or a specific day.
           Examples: "How was he yesterday?", "Any new updates?", "What did the last log say?"

        2. `TRENDS_QUERY`: The user is asking about their child's progress over a period of time.
           Examples: "How has he been doing this past month?", "Show me his progress for the last two weeks.", "What are his trends since July?"

        3. `GENERAL_KNOWLEDGE`: The user is asking for general information, definitions, or strategies about autism-related topics.
           Examples: "What is stimming?", "How can I help with tantrums?", "Tell me about potty training."

        4. `COMBINED_QUERY`: The user is connecting a recent event to a request for general information.
           Examples: "The report mentioned a meltdown, what should I do about them?"

        User Question: "{query}"
        
        Return only the category name.
        """
        try:
            response = self.client.chat.completions.create(
                model=config.ROUTER_MODEL, messages=[{'role': 'user', 'content': prompt}],
                max_tokens=20, temperature=0.0
            )
            content = response.choices[0].message.content.strip().replace("`", "")
            if "ANOMALY_REPORT" in content: return "ANOMALY_REPORT"
            if "TRENDS_QUERY" in content: return "TRENDS_QUERY"
            if "GENERAL_KNOWLEDGE" in content: return "GENERAL_KNOWLEDGE"
            if "COMBINED_QUERY" in content: return "COMBINED_QUERY"
            return "GENERAL_KNOWLEDGE"
        except Exception as e:
            print(f"❌ Error in routing query: {e}")
            return "GENERAL_KNOWLEDGE"

    def _extract_dates_for_trends(self, query: str) -> Optional[Tuple[str, str]]:
        """Uses an LLM to extract start and end dates from a natural language query."""
        today = datetime.now().strftime('%Y-%m-%d')
        prompt = f"""
        Today's date is {today}.
        Analyze the user's query and extract a start_date and end_date in YYYY-MM-DD format.
        - "last month": Go from the first day of the previous month to the last day.
        - "this week": Go from last Monday to today.
        - "last 7 days": Go from 7 days ago to today.
        
        User Query: "{query}"

        Return a JSON object with "start_date" and "end_date".
        {{
          "start_date": "YYYY-MM-DD",
          "end_date": "YYYY-MM-DD"
        }}
        """
        try:
            response = self.client.chat.completions.create(
                model=config.ROUTER_MODEL, messages=[{'role': 'user', 'content': prompt}],
                response_format={"type": "json_object"}, max_tokens=100, temperature=0.0
            )
            dates = json.loads(response.choices[0].message.content)
            return dates.get("start_date"), dates.get("end_date")
        except Exception as e:
            print(f"❌ Error extracting dates: {e}")
            return None

    def _get_trend_report(self, start_date: str, end_date: str) -> str:
        """Makes an HTTP request to the anomaly system's API to get the trend report."""
        url = f"{config.ANOMALY_API_URL}/report/trends/{self.child_id}"
        params = {"start_date": start_date, "end_date": end_date}
        print(f"📞 Calling Trend Report API: {url} with params {params}")
        try:
            response = requests.get(url, params=params)
            if response.status_code == 200:
                return response.text
            elif response.status_code == 404:
                return f"I couldn't find enough data for the period between {start_date} and {end_date} to generate a trend report. You may need to broaden the date range or analyze more logs."
            else:
                return f"Sorry, I encountered an error (HTTP {response.status_code}) while trying to generate the trend report."
        except requests.exceptions.RequestException as e:
            print(f"❌ API call to trend report failed: {e}")
            return "I'm sorry, I was unable to connect to the trend analysis service right now. Please try again later."

    def _synthesize_anomaly_report(self, analysis_data: Dict) -> Tuple[str, Optional[str]]:
        """Converts anomaly JSON into a human-readable summary and identifies a follow-up topic."""
        if not analysis_data:
            return "No analysis data was found for the child.", None
        
        if not analysis_data.get('anomaly_detected'):
            return "No significant anomalies were detected in the latest log.", None

        summary_parts = [f"Summary of the latest log analysis (from {analysis_data.get('timestamp', 'a recent log')}):"]
        explanation = analysis_data.get('detailed_explanation', 'An anomaly was detected.')
        summary_parts.append(f"- Anomaly Status: DETECTED.")
        summary_parts.append(f"- System Explanation: {explanation.replace('  -', '-')}")
        
        follow_up_topic = None
        if analysis_data.get("question_analysis", {}).get("significant_questions"):
            first_issue = analysis_data["question_analysis"]["significant_questions"][0]
            desc = first_issue.get("question_description", "").lower()
            if "behavior" in desc: follow_up_topic = "strategies for difficult behaviors"
            elif "tantrum" in desc: follow_up_topic = "how to handle tantrums"
            elif "routine" in desc: follow_up_topic = "strategies for routines"
            elif "sleep" in desc: follow_up_topic = "help with sleeping"
            elif "eating" in desc: follow_up_topic = "help with picky eating"

        return "\n".join(summary_parts), follow_up_topic

    def generate_response(self, query: str) -> str:
        """Generates a complete, non-streaming response."""
        intent = self._route_query(query)
        print(f"🧠 Query routed with intent: {intent}")

        anomaly_context, book_context, trend_context = "", "", ""
        follow_up_suggestion = ""

        if intent == "TRENDS_QUERY":
            dates = self._extract_dates_for_trends(query)
            if dates and dates[0] and dates[1]:
                trend_context = self._get_trend_report(dates[0], dates[1])
            else:
                trend_context = "I'm sorry, I couldn't determine the date range from your question. Could you please be more specific, for example, by saying 'show me the trend for last month'?"

        elif intent in ["ANOMALY_REPORT", "COMBINED_QUERY"]:
            latest_analysis = self.anomaly_manager.get_latest_analysis(self.child_id)
            if latest_analysis:
                anomaly_summary, follow_up_topic = self._synthesize_anomaly_report(latest_analysis)
                anomaly_context = anomaly_summary
                if follow_up_topic:
                    follow_up_suggestion = f"\n\nIf you'd like, you can ask me for '{follow_up_topic}' to see what the book suggests."
            else:
                anomaly_context = f"No recent analysis report is available for child {self.child_id}."

        if intent in ["GENERAL_KNOWLEDGE", "COMBINED_QUERY"]:
            search_results = self.book_kb.search(query, n_results=1)
            if search_results:
                chunk = search_results[0]
                book_context = f"Excerpt from the book 'Turn Autism Around' (Chapter {chunk['metadata']['chapter_number']}):\n{chunk['text']}"
            else:
                book_context = "No relevant information was found in the book for this query."
        
        interpretation_guidelines = """
        --- INTERPRETATION GUIDE ---
        This is your rulebook for explaining technical data to the parent.

        **1. Explaining Anomaly Reports:**
        - **If the report says "No significant anomalies":** This is great news. Be positive and reassuring. Say something like, "I've reviewed the latest log, and it's good news! The system didn't detect any significant changes or unusual patterns. It seems like it was a stable day, which is wonderful to see."
        - **If an anomaly IS detected, explain WHY in simple terms:**
          - **If due to "New emergent behavior":** Explain this calmly. Say: "The system highlighted this because it's a behavior that wasn't present in his initial baseline profile. It's not a cause for alarm, but it's something to be aware of and monitor."
          - **If due to "regression":** Explain this as a change from the usual. Say: "This means the behavior of '[behavior description]' was noted as more frequent or intense than is typical for him based on his baseline."
          - **If due to the "Child-Specific Model" (a pattern anomaly):** Use an analogy. Say: "Interestingly, no single behavior was a major issue. However, the system's learning model, which understands his day-to-day patterns, noted that the *overall combination* of behaviors for the day was statistically unusual compared to his recent history. Think of it as a day with a different 'rhythm' than normal."

        **2. Explaining Trend Reports:**
        - **If there are "Areas of Regression":** Translate this carefully. Say: "The report identified some areas of regression. This doesn't mean there was a bad day, but rather that behaviors like '[behavior description]' have been slowly increasing in frequency over the past [time period]. This helps us see what to focus on."
        - **If there are "Areas of Improvement":** Frame this positively. Say: "The report found some wonderful areas of improvement. This means that behaviors like '[behavior description]' have been slowly decreasing over the past [time period], which is a fantastic trend."
        - **If the report says "generally stable":** Explain this as a good thing. Say: "The analysis shows that his behavior has remained generally stable over the past [time period], with no major increasing or decreasing trends. Consistency is a positive sign."
        - **ALWAYS include this disclaimer for trends:** "Remember, this is a statistical analysis to help spot patterns over time. It's a tool to help us, not a final judgment."
        --- END OF INTERPRETATION GUIDE ---
        """

        final_prompt = f"""
        You are an expert, empathetic assistant named 'Aida' for a parent of a child with autism.
        Your knowledge is STRICTLY LIMITED to the information and rules provided below. Do not add outside information, opinions, or medical advice.
        You are speaking to the parent of child '{self.child_id}'.

        {interpretation_guidelines}

        --- CHILD'S LATEST ANOMALY REPORT ---
        {anomaly_context if anomaly_context else "Not requested."}
        --- END OF ANOMALY REPORT ---

        --- CHILD'S LONG-TERM TREND REPORT ---
        {trend_context if trend_context else "Not requested."}
        --- END OF TREND REPORT ---
        
        --- GENERAL KNOWLEDGE FROM BOOK ---
        {book_context if book_context else "Not requested."}
        --- END OF GENERAL KNOWLEDGE ---

        Based ONLY on the provided information and your interpretation guide, answer the parent's question: "{query}"
        - If explaining an anomaly, add the follow-up suggestion: "{follow_up_suggestion}"
        """
        
        try:
            response = self.client.chat.completions.create(
                model=config.LLM_MODEL, messages=[{'role': 'user', 'content': final_prompt}],
                max_tokens=1024, temperature=0.2
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            print(f"❌ Error generating final response: {e}")
            return "I'm sorry, but I encountered an error while processing your request. Please try again."

    async def generate_streaming_response(self, query: str):
        """
        An asynchronous generator that yields response chunks from the LLM as they arrive.
        """
        # --- 1. Context Gathering (Same as the non-streaming method) ---
        intent = self._route_query(query)
        print(f"🧠 Streaming Query routed with intent: {intent}")

        anomaly_context, book_context, trend_context = "", "", ""
        follow_up_suggestion = ""

        if intent == "TRENDS_QUERY":
            dates = self._extract_dates_for_trends(query)
            if dates and dates[0] and dates[1]:
                trend_context = self._get_trend_report(dates[0], dates[1])
            else:
                trend_context = "I'm sorry, I couldn't determine the date range from your question..."
        elif intent in ["ANOMALY_REPORT", "COMBINED_QUERY"]:
            latest_analysis = self.anomaly_manager.get_latest_analysis(self.child_id)
            if latest_analysis:
                anomaly_summary, follow_up_topic = self._synthesize_anomaly_report(latest_analysis)
                anomaly_context = anomaly_summary
                if follow_up_topic:
                    follow_up_suggestion = f"\n\nIf you'd like, you can ask me for '{follow_up_topic}'..."
            else:
                anomaly_context = f"No recent analysis report is available for child {self.child_id}."
        if intent in ["GENERAL_KNOWLEDGE", "COMBINED_QUERY"]:
            search_results = self.book_kb.search(query, n_results=1)
            if search_results:
                chunk = search_results[0]
                book_context = f"Excerpt from the book 'Turn Autism Around' (Chapter {chunk['metadata']['chapter_number']}):\n{chunk['text']}"
            else:
                book_context = "No relevant information was found in the book for this query."
        
        # --- 2. Prompt Building (Same as the non-streaming method) ---
        interpretation_guidelines = """
        --- INTERPRETATION GUIDE ---
        This is your rulebook for explaining technical data to the parent.

        **1. Explaining Anomaly Reports:**
        - **If the report says "No significant anomalies":** This is great news. Be positive and reassuring. Say something like, "I've reviewed the latest log, and it's good news! The system didn't detect any significant changes or unusual patterns. It seems like it was a stable day, which is wonderful to see."
        - **If an anomaly IS detected, explain WHY in simple terms:**
          - **If due to "New emergent behavior":** Explain this calmly. Say: "The system highlighted this because it's a behavior that wasn't present in his initial baseline profile. It's not a cause for alarm, but it's something to be aware of and monitor."
          - **If due to "regression":** Explain this as a change from the usual. Say: "This means the behavior of '[behavior description]' was noted as more frequent or intense than is typical for him based on his baseline."
          - **If due to the "Child-Specific Model" (a pattern anomaly):** Use an analogy. Say: "Interestingly, no single behavior was a major issue. However, the system's learning model, which understands his day-to-day patterns, noted that the *overall combination* of behaviors for the day was statistically unusual compared to his recent history. Think of it as a day with a different 'rhythm' than normal."

        **2. Explaining Trend Reports:**
        - **If there are "Areas of Regression":** Translate this carefully. Say: "The report identified some areas of regression. This doesn't mean there was a bad day, but rather that behaviors like '[behavior description]' have been slowly increasing in frequency over the past [time period]. This helps us see what to focus on."
        - **If there are "Areas of Improvement":** Frame this positively. Say: "The report found some wonderful areas of improvement. This means that behaviors like '[behavior description]' have been slowly decreasing over the past [time period], which is a fantastic trend."
        - **If the report says "generally stable":** Explain this as a good thing. Say: "The analysis shows that his behavior has remained generally stable over the past [time period], with no major increasing or decreasing trends. Consistency is a positive sign."
        - **ALWAYS include this disclaimer for trends:** "Remember, this is a statistical analysis to help spot patterns over time. It's a tool to help us, not a final judgment."
        --- END OF INTERPRETATION GUIDE ---
        """

        final_prompt = f"""
        You are an expert, empathetic assistant named 'Aida' for a parent of a child with autism.
        Your knowledge is STRICTLY LIMITED to the information and rules provided below. Do not add outside information, opinions, or medical advice.
        You are speaking to the parent of child '{self.child_id}'.

        {interpretation_guidelines}

        --- CHILD'S LATEST ANOMALY REPORT ---
        {anomaly_context if anomaly_context else "Not requested."}
        --- END OF ANOMALY REPORT ---

        --- CHILD'S LONG-TERM TREND REPORT ---
        {trend_context if trend_context else "Not requested."}
        --- END OF TREND REPORT ---
        
        --- GENERAL KNOWLEDGE FROM BOOK ---
        {book_context if book_context else "Not requested."}
        --- END OF GENERAL KNOWLEDGE ---

        Based ONLY on the provided information and your interpretation guide, answer the parent's question: "{query}"
        - If explaining an anomaly, add the follow-up suggestion: "{follow_up_suggestion}"
        """
        
        # --- 3. Streaming LLM Call and Yielding ---
        try:
            stream = self.client.chat.completions.create(
                model=config.LLM_MODEL,
                messages=[{'role': 'user', 'content': final_prompt}],
                max_tokens=1024,
                temperature=0.2,
                stream=True
            )
            for chunk in stream:
                content = chunk.choices[0].delta.content
                if content is not None:
                    yield content
                    await asyncio.sleep(0.01)
        
        except Exception as e:
            error_message = f"❌ An error occurred while generating the response: {e}"
            print(error_message)
            yield error_message