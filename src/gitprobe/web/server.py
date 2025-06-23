"""
GitProbe FastAPI Server

Main web server providing REST API endpoints for repository analysis.
Coordinates between different GitProbe services to provide comprehensive code analysis.
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, field_validator
from typing import Optional, List

from gitprobe.analysis.analysis_service import AnalysisService
from gitprobe.analysis.cloning import sanitize_github_url

app = FastAPI(
    title="GitProbe API",
    description="Repository analysis API using GitProbe services",
    version="1.0.0",
)


class AnalyzeRequest(BaseModel):
    github_url: str
    include_patterns: Optional[List[str]] = None
    exclude_patterns: Optional[List[str]] = None

    @field_validator("github_url")
    @classmethod
    def sanitize_url(cls, v):
        if not v:
            raise ValueError("GitHub URL is required")

        sanitized = sanitize_github_url(v)

        if "github.com" not in sanitized:
            raise ValueError("Must be a valid GitHub URL")

        return sanitized


class AnalysisResponse(BaseModel):
    status: str
    data: dict


@app.post("/analyze", response_model=AnalysisResponse)
async def analyze_repo(request: AnalyzeRequest):
    """Complete repository analysis including call graphs."""
    try:
        analysis_service = AnalysisService()
        analysis_result = analysis_service.analyze_repository_full(
            request.github_url, request.include_patterns, request.exclude_patterns
        )
        return AnalysisResponse(status="success", data=analysis_result.model_dump())
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")


@app.post("/analyze/structure-only", response_model=AnalysisResponse)
async def analyze_structure_only(request: AnalyzeRequest):
    """Lightweight repository structure analysis without call graphs."""
    try:
        analysis_service = AnalysisService()
        result = analysis_service.analyze_repository_structure_only(request.github_url)
        return AnalysisResponse(status="success", data=result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Structure analysis failed: {str(e)}")


@app.post("/analyze/llm-context")
async def get_llm_context(request: AnalyzeRequest):
    """Get clean, LLM-optimized analysis data."""
    try:
        analysis_service = AnalysisService()
        result = analysis_service.analyze_repository_full(
            request.github_url, request.include_patterns, request.exclude_patterns
        )
        llm_data = analysis_service.call_graph_analyzer.generate_llm_format()
        return {"status": "success", "data": llm_data}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"LLM context analysis failed: {str(e)}")


@app.get("/")
async def root():
    return {"message": "GitProbe API is running"}


@app.get("/health")
async def health_check():
    return {"status": "healthy"}


def cli_main():
    """CLI entry point for gitprobe-server command."""
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)


if __name__ == "__main__":
    cli_main()
