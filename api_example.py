"""
Example API Implementation using GitProbe Services
Shows how the API layer coordinates the analysis workflow.
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, HttpUrl
from typing import Optional, List
from services.analysis_orchestrator import analyze_repository
from services.utils import cleanup_repository

app = FastAPI(
    title="GitProbe API",
    description="Repository analysis API using GitProbe services",
    version="1.0.0",
)


class AnalyzeRequest(BaseModel):
    github_url: HttpUrl
    include_patterns: Optional[List[str]] = None
    exclude_patterns: Optional[List[str]] = None


class StructureOnlyRequest(BaseModel):
    github_url: HttpUrl


class AnalysisResponse(BaseModel):
    status: str
    data: dict


class ErrorResponse(BaseModel):
    status: str
    message: str


@app.post("/analyze", response_model=AnalysisResponse)
async def analyze_repo(request: AnalyzeRequest):
    """
    API endpoint for repository analysis.

    Example flow:
    1. Clone repo (utils.clone_repository)
    2. Call repo_analyzer
    3. Call call_graph_analyzer
    4. Combine results in API response
    5. Cleanup temp dir after 200 response
    """
    temp_dir = None
    try:
        # Step 1-4: Clone -> RepoAnalyzer -> CallGraphAnalyzer -> Combine
        analysis_results, temp_dir = analyze_repository(
            str(request.github_url), request.include_patterns, request.exclude_patterns
        )

        # Step 5: Return response and cleanup
        response = AnalysisResponse(status="success", data=analysis_results)

        # Cleanup after successful response
        cleanup_repository(temp_dir)

        return response

    except Exception as e:
        # Cleanup on error too
        if temp_dir:
            cleanup_repository(temp_dir)

        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")


@app.post("/analyze/structure-only", response_model=AnalysisResponse)
async def analyze_structure_only(request: StructureOnlyRequest):
    """
    API endpoint for repository structure analysis only.
    Lighter weight - no call graph analysis.
    """
    from services.analysis_orchestrator import analyze_repository_structure_only

    temp_dir = None
    try:
        # Analyze structure only
        results, temp_dir = analyze_repository_structure_only(str(request.github_url))

        # Return response and cleanup
        response = AnalysisResponse(status="success", data=results)

        cleanup_repository(temp_dir)
        return response

    except Exception as e:
        if temp_dir:
            cleanup_repository(temp_dir)

        raise HTTPException(
            status_code=500, detail=f"Structure analysis failed: {str(e)}"
        )


@app.get("/")
async def root():
    """Health check endpoint"""
    return {"message": "GitProbe API is running"}


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy"}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)


# Example usage:
"""
curl -X POST http://localhost:8000/analyze \
  -H "Content-Type: application/json" \
  -d '{
    "github_url": "https://github.com/user/repo",
    "include_patterns": ["*.py", "*.js"],
    "exclude_patterns": ["*test*", "*spec*"]
  }'

curl -X POST http://localhost:8000/analyze/structure-only \
  -H "Content-Type: application/json" \
  -d '{
    "github_url": "https://github.com/user/repo"
  }'

# Access interactive API docs at: http://localhost:8000/docs
# Access alternative docs at: http://localhost:8000/redoc
"""
