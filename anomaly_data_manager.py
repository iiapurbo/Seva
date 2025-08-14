# anomaly_data_manager.py
import json
from pathlib import Path
from typing import Dict, Optional
import config

class AnomalyDataManager:
    """
    Handles retrieving and parsing child-specific anomaly detection results.
    """
    def __init__(self, base_dir: str = config.CHILD_DATA_BASE_DIR):
        self.base_dir = Path(base_dir)

    def get_latest_analysis(self, child_id: str) -> Optional[Dict]:
        """
        Finds and loads the most recent anomaly analysis JSON for a given child.
        """
        print(f"📊 Retrieving latest analysis for child: {child_id}")
        try:
            results_dir = self.base_dir / child_id / "anomaly_results"
            if not results_dir.exists():
                print(f"⚠️ No anomaly results directory found for child '{child_id}'.")
                return None

            json_files = sorted(
                results_dir.glob("analysis_*.json"),
                key=lambda f: f.stat().st_mtime,
                reverse=True
            )

            if not json_files:
                print(f"⚠️ No analysis files found for child '{child_id}'.")
                return None

            latest_file = json_files[0]
            with open(latest_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            print(f"❌ Error reading anomaly data for child '{child_id}': {e}")
            return None