# api.py

import logging
import json
import os
from datetime import date
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import PlainTextResponse
from pydantic import BaseModel, Field
from typing import Dict, Optional
from pathlib import Path

from main_system import IncrementalBehavioralMonitoringSystem
from generate_report import TrendAnalyzer, load_all_processed_logs_for_child
# <<< ADDED IMPORTS FOR STATEFULNESS >>>
from enhanced_clinical_config import BASE_DATA_DIR
from child_specific_learning import IncrementalChildLearningModel

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Multi-Child ASD Behavioral Monitoring API",
    description="An API to manage and analyze behavioral data for multiple children.",
    version="3.2.0-Stateful"
)
child_systems: Dict[str, IncrementalBehavioralMonitoringSystem] = {}

# <<< MODIFICATION: API STATEFULNESS - START >>>
def get_or_create_system(child_id: str) -> IncrementalBehavioralMonitoringSystem:
    if child_id not in child_systems:
        logger.info(f"Creating a new system instance for child: {child_id}")
        child_systems[child_id] = IncrementalBehavioralMonitoringSystem(child_id)
    return child_systems[child_id]

@app.on_event("startup")
async def load_existing_systems():
    """
    On API startup, scan the data directory and reload existing child systems
    to make the API stateful across restarts.
    """
    logger.info("🚀 API starting up. Scanning for existing child data...")
    root_data_dir = Path(BASE_DATA_DIR)
    if not root_data_dir.exists():
        logger.warning(f"Base data directory '{root_data_dir}' not found. Starting with a clean state.")
        return

    for child_dir in root_data_dir.iterdir():
        if child_dir.is_dir():
            child_id = child_dir.name
            logger.info(f"Found data for child '{child_id}'. Attempting to restore state.")
            
            try:
                system = get_or_create_system(child_id)
                
                # 1. Restore baseline data
                baseline_path = child_dir / "baseline_data.json"
                if not baseline_path.exists():
                    logger.warning(f"  - No baseline_data.json for {child_id}. Cannot restore.")
                    continue
                
                with open(baseline_path, 'r') as f:
                    baseline_data = json.load(f)
                
                # This sets the baseline and marks the system as ready, but creates a *new* model
                system.set_baseline(baseline_data)

                # 2. Restore the trained incremental model, overwriting the new one
                model_path = Path(system.child_folders["child_models"]) / "incremental_child_model.pkl"
                if model_path.exists():
                    logger.info(f"  - Found saved model for {child_id}. Loading...")
                    # Overwrite the fresh model with the saved, trained one
                    system.child_model = IncrementalChildLearningModel.load_model(str(model_path))
                    logger.info(f"  - Successfully restored model with {len(system.child_model.training_data)} samples.")
                
                logger.info(f"✅ System for child '{child_id}' is fully restored and ready.")

            except Exception as e:
                logger.error(f"❌ Failed to restore state for child '{child_id}': {e}", exc_info=True)

# <<< MODIFICATION: API STATEFULNESS - END >>>

class BaselinePayload(BaseModel):
    child_id: str = Field(..., example="child_01", description="A unique identifier for the child.")
    individual_responses: Dict[str, int] = Field(..., example={"Q1": 0, "Q9": 1, "Q45": 2, "Q75": 1})

class LogPayload(BaseModel):
    child_id: str = Field(..., example="child_01", description="The ID of the child this log belongs to.")
    timestamp: str = Field(..., example="2023-10-27T10:00:00Z", description="ISO 8601 timestamp.")
    log: str = Field(..., example="He played with his toys and made eye contact.")


@app.post("/baseline", status_code=202, summary="Set or Update Child Baseline")
async def set_baseline_data(baseline_data: BaselinePayload):
    try:
        system = get_or_create_system(baseline_data.child_id)
        system.set_baseline(baseline_data.dict())
        return {"message": f"Baseline successfully set for child '{baseline_data.child_id}'. Ready for log analysis."}
    except ValueError as e:
        raise HTTPException(status_code=400, detail=f"Validation failed: {e}")
    except Exception as e:
        logger.error(f"Error during baseline setting for {baseline_data.child_id}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error.")

@app.post("/log", summary="Analyze a Parent Log for a Specific Child")
async def analyze_single_log(log_data: LogPayload):
    system = child_systems.get(log_data.child_id)
    if not system or not system.is_ready:
        raise HTTPException(status_code=400, detail=f"System not initialized for child '{log_data.child_id}'. Please POST to /baseline first.")
    
    try:
        analysis_result = system.analyze_log(log_data.dict())
        return analysis_result
    except Exception as e:
        logger.error(f"Error during log analysis for {log_data.child_id}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Internal server error: {e}")


@app.get("/report/trends/{child_id}", response_class=PlainTextResponse, summary="Generate Trend Report for a Child")
async def get_trend_report(
    child_id: str,
    start_date: Optional[date] = Query(None, description="Start date for the report (YYYY-MM-DD). If omitted, uses the earliest available log."),
    end_date: Optional[date] = Query(None, description="End date for the report (YYYY-MM-DD). If omitted, uses the latest available log.")
):
    """
    Analyzes processed logs for a specific child to generate a trend report.
    You can filter the logs by providing a start and/or end date.
    """
    logger.info(f"Generating trend report for child: {child_id} (Start: {start_date}, End: {end_date})")
    try:
        # Pass the optional dates to the loading function
        all_logs = load_all_processed_logs_for_child(child_id, start_date, end_date)
        
        analyzer = TrendAnalyzer(child_id, all_logs)
        report_text = analyzer.generate_report_text()
        return PlainTextResponse(content=report_text)
    except (ValueError, FileNotFoundError) as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"Error during report generation for {child_id}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error.")