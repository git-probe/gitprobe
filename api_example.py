from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional

from services.analysis_service import AnalysisService
from utils.logging_config import setup_logging

# --- App Setup ---
setup_logging()
app = FastAPI(
    title="GitProbe API",
    description="Analyze GitHub repositories to generate call graphs and insights.",
    version="1.0.0",
)


# --- Pydantic Models ---
class AnalyzeRequest(BaseModel):
    github_url: str
    include_patterns: Optional[List[str]] = None
    exclude_patterns: Optional[List[str]] = None

class AnalysisResponse(BaseModel):
    status: str
    message: Optional[str] = None
    analysis_id: Optional[str] = None
    data: Optional[dict] = None


# --- API Endpoints ---
@app.post("/analyze", response_model=AnalysisResponse)
async def analyze_repository(request: AnalyzeRequest):
    """
    Analyze a GitHub repository and return the full analysis results.
    """
    try:
        analysis_service = AnalysisService()
        result = analysis_service.analyze_repository_full(
            str(request.github_url), request.include_patterns, request.exclude_patterns
        )
        return {
            "status": "success",
            "analysis_id": result.repository.analysis_id,
            "data": result.dict(),
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/analyze/llm-context")
async def get_llm_context(request: AnalyzeRequest):
    """Get clean, LLM-optimized analysis data."""
    try:
        analysis_service = AnalysisService()
        analysis_service.analyze_repository_full(
            str(request.github_url), request.include_patterns, request.exclude_patterns
        )

        # Generate clean LLM format
        llm_data = analysis_service.call_graph_analyzer.generate_llm_format()

        return {"status": "success", "data": llm_data}

    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"LLM context analysis failed: {str(e)}"
        ) 