# generate_report.py

import json
from datetime import datetime, date # <<< ADD date IMPORT
from pathlib import Path
import numpy as np
from typing import Optional # <<< ADD Optional IMPORT

from enhanced_clinical_config import get_child_folder_structure, ATEC_QUESTIONS, QUESTION_ORDER

# TrendAnalyzer class remains exactly the same.
class TrendAnalyzer:
    """
    Analyzes a given list of processed logs to identify long-term behavioral trends.
    """
    def __init__(self, child_id, processed_logs):
        if not processed_logs:
            # A more specific message for the date-filtered case
            raise ValueError("No logs found for the specified date range.")
        if len(processed_logs) < 3:
            raise ValueError("Trend analysis requires at least 3 data points. Broaden your date range.")
        
        self.child_id = child_id
        self.logs = sorted(processed_logs, key=lambda x: x.get('timestamp', ''))
        self.question_descriptions = {qid: data['description'] for qid, data in ATEC_QUESTIONS.items()}
        self.feature_matrix = np.array([log['question_level_features'] for log in self.logs])
        self.timestamps = [log.get('timestamp', '') for log in self.logs]

    def generate_report_text(self):
        """
        Performs the trend analysis and returns the report as a string.
        This is the primary method used by the API.
        """
        improvements, regressions = {}, {}
        time_steps = np.arange(len(self.logs))

        for i, q_id in enumerate(QUESTION_ORDER):
            behavior_scores = self.feature_matrix[:, i]
            if np.any(behavior_scores > 0):
                slope, _ = np.polyfit(time_steps, behavior_scores, 1)
                description = self.question_descriptions.get(q_id, q_id)
                if slope < -0.05: improvements[description] = slope
                elif slope > 0.05: regressions[description] = slope
        
        start_date_str = datetime.fromisoformat(self.timestamps[0].replace("Z", "+00:00")).strftime('%Y-%m-%d')
        end_date_str = datetime.fromisoformat(self.timestamps[-1].replace("Z", "+00:00")).strftime('%Y-%m-%d')
        
        report = [
            "="*80, f"LONGITUDINAL BEHAVIORAL TREND REPORT", "="*80,
            f"Child ID: {self.child_id}", f"Analysis Period: {start_date_str} to {end_date_str}",
            f"Total Logs Analyzed: {len(self.logs)}", "\n--- EXECUTIVE SUMMARY ---"
        ]

        if not improvements and not regressions:
            report.append("The child's behavior has remained generally stable across the selected period.")
        else:
            report.append(f"Found {len(regressions)} areas of potential regression and {len(improvements)} areas of improvement.")

        if regressions:
            report.append("\n--- AREAS OF REGRESSION (Worsening Trend) ---")
            for i, (desc, _) in enumerate(sorted(regressions.items(), key=lambda item: item[1], reverse=True)[:5]):
                report.append(f"{i+1}. {desc}")
        
        if improvements:
            report.append("\n--- AREAS OF IMPROVEMENT (Improving Trend) ---")
            for i, (desc, _) in enumerate(sorted(improvements.items(), key=lambda item: item[1])[:5]):
                report.append(f"{i+1}. {desc}")

        report.extend(["\n--- NOTE ---", "This is an automated analysis of statistical trends.", "="*80])
        return "\n".join(report)


# <<< MODIFIED HELPER FUNCTION SIGNATURE AND LOGIC >>>
def load_all_processed_logs_for_child(
    child_id: str, 
    start_date: Optional[date] = None, 
    end_date: Optional[date] = None
):
    """
    Loads all processed logs for a child, with optional date filtering.
    """
    child_folders = get_child_folder_structure(child_id)
    processed_logs_dir = Path(child_folders["processed_logs"])
    
    if not processed_logs_dir.exists():
        raise FileNotFoundError(f"No data found for child '{child_id}'.")
        
    all_processed_logs = []
    log_files = sorted(list(processed_logs_dir.glob("processed_*.json")))
    if not log_files:
        raise FileNotFoundError(f"No processed logs found for child '{child_id}'.")
        
    for f in log_files:
        with open(f, 'r', encoding='utf-8') as log_f:
            log_data = json.load(log_f)
            
            # --- DATE FILTERING LOGIC ---
            is_after_start = True
            if start_date:
                log_date = datetime.fromisoformat(log_data['timestamp'].replace("Z", "+00:00")).date()
                if log_date < start_date:
                    is_after_start = False
            
            is_before_end = True
            if end_date:
                log_date = datetime.fromisoformat(log_data['timestamp'].replace("Z", "+00:00")).date()
                if log_date > end_date:
                    is_before_end = False

            if is_after_start and is_before_end:
                all_processed_logs.append(log_data)
                
    return all_processed_logs

# The main function can be updated to show how to test this from CLI
def main():
    """Example of running the report script from the command line."""
    import sys
    if len(sys.argv) < 2:
        print("Usage: python generate_report.py <child_id> [start_date] [end_date]")
        print("Dates should be in YYYY-MM-DD format.")
        sys.exit(1)
    
    child_id_to_report = sys.argv[1]
    start = date.fromisoformat(sys.argv[2]) if len(sys.argv) > 2 else None
    end = date.fromisoformat(sys.argv[3]) if len(sys.argv) > 3 else None

    print(f"🧠 Generating Trend Report for child: {child_id_to_report} (From: {start or 'start'}, To: {end or 'end'})")
    try:
        all_logs = load_all_processed_logs_for_child(child_id_to_report, start, end)
        analyzer = TrendAnalyzer(child_id_to_report, all_logs)
        report_text = analyzer.generate_report_text()
        
        child_folders = get_child_folder_structure(child_id_to_report)
        report_filename = f"trend_report_{datetime.now().strftime('%Y%m%d')}.txt"
        report_path = Path(child_folders["anomaly_results"]) / report_filename
        with open(report_path, 'w', encoding='utf-8') as f: f.write(report_text)
            
        print(f"\n✅ Trend analysis complete. Report saved to: {report_path}")
        print(report_text)
    except (ValueError, FileNotFoundError) as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    main()