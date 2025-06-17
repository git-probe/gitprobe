"""
Simple GitProbe API Routes
Clean, minimal API using simple models only.
"""

from fastapi import APIRouter, HTTPException
from typing import Optional, List
import logging
from datetime import datetime

from models import Function, CallRelationship, Repository, AnalysisResult, NodeSelection, ExportData
from services import SimpleAnalysisService, SimpleExportService

logger = logging.getLogger(__name__)
router = APIRouter()

# Simple in-memory storage (for production, use Redis)
_analysis_cache: dict[str, AnalysisResult] = {}
_selections: dict[str, NodeSelection] = {}

# Services
analysis_service = SimpleAnalysisService()
export_service = SimpleExportService()


@router.post("/analyze", response_model=AnalysisResult)
async def analyze_repository(
    repo_url: str,
    include_patterns: Optional[List[str]] = None,
    exclude_patterns: Optional[List[str]] = None
):
    """
    Analyze a GitHub repository and return the complete call graph.
    
    Args:
        repo_url: GitHub repository URL
        include_patterns: File patterns to include (e.g., ["*.py", "src/"])
        exclude_patterns: File patterns to exclude (e.g., ["*.md", "test/"])
    """
    try:
        logger.info(f"Analyzing repository: {repo_url}")
        
        # Validate URL
        if not repo_url or not repo_url.strip():
            raise HTTPException(status_code=400, detail="Repository URL is required")
        
        # Perform analysis with filters
        filters = {}
        if include_patterns:
            filters["include_patterns"] = include_patterns
        if exclude_patterns:
            filters["exclude_patterns"] = exclude_patterns
            
        result = await analysis_service.analyze_repository_with_filters(
            repo_url.strip(), filters if filters else None
        )
        
        # Store in cache with URL as key
        analysis_id = repo_url.strip().replace('/', '_').replace(':', '_')
        _analysis_cache[analysis_id] = result
        _selections[analysis_id] = NodeSelection()
        
        logger.info(f"Analysis completed: {len(result.functions)} functions, {len(result.relationships)} relationships")
        
        return result
        
    except Exception as e:
        logger.error(f"Analysis failed: {e}")
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")


@router.get("/analysis/{analysis_id}", response_model=AnalysisResult)
async def get_analysis(analysis_id: str):
    """Get a previously completed analysis."""
    if analysis_id not in _analysis_cache:
        raise HTTPException(status_code=404, detail="Analysis not found")
    return _analysis_cache[analysis_id]


@router.post("/analysis/{analysis_id}/select")
async def select_nodes(analysis_id: str, node_ids: List[str]):
    """Select specific nodes for operations."""
    if analysis_id not in _analysis_cache:
        raise HTTPException(status_code=404, detail="Analysis not found")
    
    if analysis_id not in _selections:
        _selections[analysis_id] = NodeSelection()
    
    _selections[analysis_id].selected_nodes = node_ids
    
    return {
        "analysis_id": analysis_id,
        "selected_nodes": node_ids,
        "count": len(node_ids)
    }


@router.put("/analysis/{analysis_id}/rename/{node_id}")
async def rename_node(analysis_id: str, node_id: str, new_name: str):
    """Rename a function node."""
    if analysis_id not in _analysis_cache:
        raise HTTPException(status_code=404, detail="Analysis not found")
    
    if analysis_id not in _selections:
        _selections[analysis_id] = NodeSelection()
    
    _selections[analysis_id].custom_names[node_id] = new_name
    
    return {
        "node_id": node_id,
        "new_name": new_name,
        "success": True
    }


@router.get("/analysis/{analysis_id}/export/json", response_model=ExportData)
async def export_json(
    analysis_id: str, 
    selected_only: bool = False,
    include_code: bool = True,
    include_relationships: bool = True
):
    """
    Export analysis as JSON with configurable options.
    
    Args:
        analysis_id: Analysis ID
        selected_only: Export only selected nodes
        include_code: Include function code snippets (for LLM)
        include_relationships: Include call relationships
    """
    if analysis_id not in _analysis_cache:
        raise HTTPException(status_code=404, detail="Analysis not found")
    
    analysis = _analysis_cache[analysis_id]
    selection = _selections.get(analysis_id) if selected_only else None
    
    return export_service.export_json_configurable(
        analysis, selection, include_code, include_relationships
    )


@router.get("/analysis/{analysis_id}/export/cytoscape")
async def export_cytoscape(analysis_id: str, selected_only: bool = False):
    """Export analysis as Cytoscape.js format."""
    if analysis_id not in _analysis_cache:
        raise HTTPException(status_code=404, detail="Analysis not found")
    
    analysis = _analysis_cache[analysis_id]
    selection = _selections.get(analysis_id) if selected_only else None
    
    return export_service.export_cytoscape(analysis, selection)


@router.get("/analysis/{analysis_id}/export/svg")
async def export_svg(analysis_id: str, selected_only: bool = False):
    """Export analysis as SVG."""
    if analysis_id not in _analysis_cache:
        raise HTTPException(status_code=404, detail="Analysis not found")
    
    analysis = _analysis_cache[analysis_id]
    selection = _selections.get(analysis_id) if selected_only else None
    
    svg_content = export_service.export_svg(analysis, selection)
    
    from fastapi.responses import Response
    return Response(content=svg_content, media_type="image/svg+xml")


@router.get("/analysis/{analysis_id}/functions", response_model=List[Function])
async def list_functions(analysis_id: str):
    """List all functions in the analysis."""
    if analysis_id not in _analysis_cache:
        raise HTTPException(status_code=404, detail="Analysis not found")
    
    return _analysis_cache[analysis_id].functions


@router.get("/analysis/{analysis_id}/functions/{node_id}")
async def get_function_details(analysis_id: str, node_id: str):
    """Get detailed information about a specific function node."""
    if analysis_id not in _analysis_cache:
        raise HTTPException(status_code=404, detail="Analysis not found")
    
    analysis = _analysis_cache[analysis_id]
    
    # Find the function
    function = None
    for func in analysis.functions:
        if f"{func.file_path}:{func.name}" == node_id:
            function = func
            break
    
    if not function:
        raise HTTPException(status_code=404, detail="Function not found")
    
    # Get custom name if set
    selection = _selections.get(analysis_id)
    display_name = function.name
    if selection and node_id in selection.custom_names:
        display_name = selection.custom_names[node_id]
    
    # Find incoming and outgoing calls
    incoming_calls = [
        r for r in analysis.relationships 
        if r.callee == node_id and r.is_resolved
    ]
    outgoing_calls = [
        r for r in analysis.relationships 
        if r.caller == node_id and r.is_resolved
    ]
    
    return {
        "node_id": node_id,
        "original_name": function.name,
        "display_name": display_name,
        "file_path": function.file_path,
        "line_start": function.line_start,
        "line_end": function.line_end,
        "parameters": function.parameters,
        "docstring": function.docstring,
        "is_method": function.is_method,
        "class_name": function.class_name,
        "code_snippet": function.code_snippet,
        "calls": {
            "incoming": len(incoming_calls),
            "outgoing": len(outgoing_calls),
            "incoming_details": [
                {
                    "caller": r.caller,
                    "line": r.call_line
                } for r in incoming_calls
            ],
            "outgoing_details": [
                {
                    "callee": r.callee,
                    "line": r.call_line
                } for r in outgoing_calls
            ]
        }
    }


@router.get("/analysis/{analysis_id}/functions/{node_id}/code")
async def get_function_code(analysis_id: str, node_id: str):
    """Get just the code snippet for a function (for copy functionality)."""
    if analysis_id not in _analysis_cache:
        raise HTTPException(status_code=404, detail="Analysis not found")
    
    analysis = _analysis_cache[analysis_id]
    
    # Find the function
    function = None
    for func in analysis.functions:
        if f"{func.file_path}:{func.name}" == node_id:
            function = func
            break
    
    if not function:
        raise HTTPException(status_code=404, detail="Function not found")
    
    from fastapi.responses import PlainTextResponse
    return PlainTextResponse(
        content=function.code_snippet or "",
        media_type="text/plain"
    )


@router.get("/analysis/{analysis_id}/file-tree")
async def get_file_tree(analysis_id: str):
    """Get repository file tree structure."""
    if analysis_id not in _analysis_cache:
        raise HTTPException(status_code=404, detail="Analysis not found")
    
    analysis = _analysis_cache[analysis_id]
    
    return {
        "repository": analysis.repository.dict(),
        "file_tree": analysis.file_tree,
        "summary": {
            "total_files": _count_files_in_tree(analysis.file_tree),
            "total_size_kb": _calculate_tree_size(analysis.file_tree)
        }
    }


@router.get("/analysis/{analysis_id}/summary")
async def get_summary(analysis_id: str):
    """Get analysis summary."""
    if analysis_id not in _analysis_cache:
        raise HTTPException(status_code=404, detail="Analysis not found")
    
    analysis = _analysis_cache[analysis_id]
    
    return {
        "repository": analysis.repository.dict(),
        "summary": analysis.summary,
        "function_count": len(analysis.functions),
        "relationship_count": len(analysis.relationships),
        "resolved_relationships": len([r for r in analysis.relationships if r.is_resolved]),
        "files": list(set(f.file_path for f in analysis.functions)),
        "classes": list(set(f.class_name for f in analysis.functions if f.class_name))
    }


def _count_files_in_tree(tree: dict) -> int:
    """Count total files in tree."""
    if not tree or tree.get("type") == "file":
        return 1 if tree else 0
    return sum(_count_files_in_tree(child) for child in tree.get("children", []))


def _calculate_tree_size(tree: dict) -> float:
    """Calculate total size in KB."""
    if not tree:
        return 0.0
    if tree.get("type") == "file":
        return tree.get("size_kb", 0)
    return sum(_calculate_tree_size(child) for child in tree.get("children", []))


@router.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "service": "GitProbe Simple API",
        "timestamp": datetime.utcnow(),
        "cached_analyses": len(_analysis_cache)
    }


@router.get("/")
async def api_info():
    """API information."""
    return {
        "name": "GitProbe Simple API",
        "description": "Simple, direct repository call graph analysis",
        "version": "1.0.0",
        "features": [
            "GitHub repository analysis with filtering",
            "Complete file tree structure",
            "Function call graph extraction",
            "Interactive node details and code viewing", 
            "Node selection and custom renaming",
            "Configurable exports (JSON, SVG, Cytoscape.js)",
            "LLM-optimized JSON with optional code inclusion"
        ],
        "endpoints": {
            "analyze": "POST /analyze?include_patterns&exclude_patterns",
            "get_analysis": "GET /analysis/{id}",
            "file_tree": "GET /analysis/{id}/file-tree",
            "functions": "GET /analysis/{id}/functions",
            "function_details": "GET /analysis/{id}/functions/{node_id}",
            "function_code": "GET /analysis/{id}/functions/{node_id}/code",
            "select_nodes": "POST /analysis/{id}/select",
            "rename_node": "PUT /analysis/{id}/rename/{node_id}",
            "summary": "GET /analysis/{id}/summary",
            "export_json": "GET /analysis/{id}/export/json?include_code&include_relationships&selected_only",
            "export_cytoscape": "GET /analysis/{id}/export/cytoscape?selected_only",
            "export_svg": "GET /analysis/{id}/export/svg?selected_only"
        }
    } 