"""
Example API Implementation using GitProbe Services
Shows how the API layer coordinates the analysis workflow.
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, HttpUrl
from typing import Optional, List
from services.analysis_service import AnalysisService

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
    API endpoint for complete repository analysis including call graphs.
    
    Supports multiple programming languages:
    - Python (fully supported)
    - JavaScript/TypeScript (coming soon)
    
    The analysis service handles all orchestration including:
    - Repository cloning and cleanup
    - File structure analysis with filtering
    - Multi-language AST parsing
    - Call graph generation and visualization
    """
    try:
        # Use the centralized analysis service
        analysis_service = AnalysisService()
        analysis_result = analysis_service.analyze_repository_full(
            str(request.github_url), 
            request.include_patterns, 
            request.exclude_patterns
        )

        # Convert AnalysisResult to dict for API response
        response_data = {
            "repository": analysis_result.repository.model_dump(),
            "file_tree": analysis_result.file_tree,
            "file_summary": {k: v for k, v in analysis_result.summary.items() if k != "analysis_type"},
            "call_graph": {k: v for k, v in analysis_result.summary.items() if k in ["total_functions", "total_calls", "languages_found", "files_analyzed"]},
            "functions": [func.model_dump() for func in analysis_result.functions],
            "relationships": [rel.model_dump() for rel in analysis_result.relationships],
            "summary": analysis_result.summary
        }

        return AnalysisResponse(status="success", data=response_data)

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")


@app.post("/analyze/structure-only", response_model=AnalysisResponse)
async def analyze_structure_only(request: StructureOnlyRequest):
    """
    API endpoint for lightweight repository structure analysis.
    
    Performs file tree analysis without call graph generation:
    - Faster execution
    - Lower resource usage
    - Ideal for large repositories or quick overviews
    """
    try:
        # Use the centralized analysis service for structure-only analysis
        analysis_service = AnalysisService()
        result = analysis_service.analyze_repository_structure_only(str(request.github_url))

        return AnalysisResponse(status="success", data=result)

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Structure analysis failed: {str(e)}")


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
