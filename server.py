"""
Example API Implementation using GitProbe Services
Shows how the API layer coordinates the analysis workflow.
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, HttpUrl, field_validator
from typing import Optional, List
from services.analysis_service import AnalysisService
from services.cloning import sanitize_github_url

app = FastAPI(
    title="GitProbe API",
    description="Repository analysis API using GitProbe services",
    version="1.0.0",
)


class AnalyzeRequest(BaseModel):
    github_url: HttpUrl
    include_patterns: Optional[List[str]] = None
    exclude_patterns: Optional[List[str]] = None


class FlexibleAnalyzeRequest(BaseModel):
    github_url: str
    include_patterns: Optional[List[str]] = None
    exclude_patterns: Optional[List[str]] = None

    @field_validator('github_url')
    @classmethod
    def sanitize_url(cls, v):
        """Sanitize and validate GitHub URL"""
        if not v:
            raise ValueError("GitHub URL is required")
        
        # Use our sanitize function to clean up the URL
        sanitized = sanitize_github_url(v)
        
        # Basic validation that it looks like a GitHub URL
        if 'github.com' not in sanitized:
            raise ValueError("Must be a valid GitHub URL")
            
        return sanitized


class StructureOnlyRequest(BaseModel):
    github_url: HttpUrl


class AnalysisResponse(BaseModel):
    status: str
    data: dict


class ErrorResponse(BaseModel):
    status: str
    message: str


@app.post("/analyze", response_model=AnalysisResponse)
async def analyze_repo(request: FlexibleAnalyzeRequest):
    """
    API endpoint for complete repository analysis including call graphs.

        Supports multiple programming languages:
    - Python (fully supported)
    - JavaScript/TypeScript (fully supported)
    - C/C++ (fully supported)
    - Go (fully supported)
    - Rust (fully supported)

    The analysis service handles all orchestration including:
    - Repository cloning and cleanup
    - File structure analysis with filtering
    - Multi-language AST parsing
    - Call graph generation and visualization
    """
    try:
        print(f"🔧 /analyze Debug: Received URL: {request.github_url}")
        
        # Use the centralized analysis service
        analysis_service = AnalysisService()
        analysis_result = analysis_service.analyze_repository_full(
            request.github_url, request.include_patterns, request.exclude_patterns
        )

        # Convert AnalysisResult to dict for API response
        return AnalysisResponse(status="success", data=analysis_result.model_dump())

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
        result = analysis_service.analyze_repository_structure_only(
            str(request.github_url)
        )

        return AnalysisResponse(status="success", data=result)

    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Structure analysis failed: {str(e)}"
        )


@app.post("/analyze/llm-context")
async def get_llm_context(request: AnalyzeRequest):
    """Get clean, LLM-optimized analysis data."""
    try:
        analysis_service = AnalysisService()
        result = analysis_service.analyze_repository_full(
            str(request.github_url), request.include_patterns, request.exclude_patterns
        )

        # Generate clean LLM format
        llm_data = analysis_service.call_graph_analyzer.generate_llm_format()

        return {"status": "success", "data": llm_data}

    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"LLM context analysis failed: {str(e)}"
        )


@app.get("/")
async def root():
    """Health check endpoint"""
    return {"message": "GitProbe API is running"}


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy"}


@app.post("/api/probe", response_model=AnalysisResponse)
async def probe_repo(request: FlexibleAnalyzeRequest):
    """
    API endpoint for repository analysis that accepts flexible URL formats.
    
    This endpoint automatically sanitizes GitHub URLs and supports formats like:
    - github.com/owner/repo
    - owner/repo  
    - https://github.com/owner/repo/tree/branch/path
    
    Performs complete repository analysis including call graphs.
    """
    try:
        print(f"🔧 API Debug: Received URL: {request.github_url}")
        
        # Use the centralized analysis service
        analysis_service = AnalysisService()
        analysis_result = analysis_service.analyze_repository_full(
            request.github_url, request.include_patterns, request.exclude_patterns
        )

        # Convert AnalysisResult to dict for API response
        return AnalysisResponse(status="success", data=analysis_result.model_dump())

    except Exception as e:
        print(f"❌ API Error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)


# Example usage:
"""
# Full analysis with visualization data
curl -X POST http://localhost:8000/analyze \
  -H "Content-Type: application/json" \
  -d '{
    "github_url": "https://github.com/user/repo",
    "include_patterns": ["*.py", "*.js"],
    "exclude_patterns": ["*test*", "*spec*"]
  }'

# Structure-only analysis (lightweight)
curl -X POST http://localhost:8000/analyze/structure-only \
  -H "Content-Type: application/json" \
  -d '{
    "github_url": "https://github.com/user/repo"
  }'

# LLM-optimized context (clean, structured for AI consumption)
curl -X POST http://localhost:8000/analyze/llm-context \
  -H "Content-Type: application/json" \
  -d '{
    "github_url": "https://github.com/user/repo",
    "include_patterns": ["*.py"],
    "exclude_patterns": ["*test*", "*spec*"]
  }'

# Access interactive API docs at: http://localhost:8000/docs
# Access alternative docs at: http://localhost:8000/redoc
"""
