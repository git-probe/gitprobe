"""
Simple Analysis Service
Clean, minimal implementation using simple models only.
"""

import asyncio
import logging
from typing import Dict, List, Optional, Any
from datetime import datetime

from models import Function, CallRelationship, Repository, AnalysisResult
from services.core_analyzer import CallGraphAnalyzer, RepoAnalyzer

logger = logging.getLogger(__name__)


class SimpleAnalysisService:
    """Simple analysis service. Engineering is simplicity."""
    
    def __init__(self):
        self.repo_analyzer = RepoAnalyzer()
        self.call_graph_analyzer = CallGraphAnalyzer(self.repo_analyzer)
        logger.info("Simple analysis service initialized")
    
    async def analyze_repository_with_filters(
        self, repo_url: str, filters: Optional[Dict[str, Any]] = None
    ) -> AnalysisResult:
        """Analyze repository and return simple result."""
        try:
            # Apply filters to repo analyzer if provided
            if filters:
                if "include_patterns" in filters:
                    self.repo_analyzer.include_patterns = filters["include_patterns"]
                if "exclude_patterns" in filters:
                    self.repo_analyzer.exclude_patterns.extend(filters["exclude_patterns"])
            
            # Run analysis
            result = await asyncio.to_thread(
                self.call_graph_analyzer.analyze_repository, 
                repo_url
            )
            
            # Convert to simple models
            repo_data = result["repository"]
            repository = Repository(
                owner=repo_data["owner"],
                name=repo_data["name"], 
                url=repo_data["url"],
                analyzed_at=datetime.utcnow()
            )
            
            # Convert functions
            functions = []
            for func_data in result["functions"]:
                function = Function(
                    name=func_data["name"],
                    file_path=func_data["file_path"],
                    line_start=func_data["line_start"],
                    line_end=func_data["line_end"],
                    parameters=func_data.get("parameters", []),
                    docstring=func_data.get("docstring"),
                    is_method=func_data.get("is_method", False),
                    class_name=func_data.get("class_name"),
                    code_snippet=func_data.get("code_snippet")
                )
                functions.append(function)
            
            # Convert relationships
            relationships = []
            for rel_data in result["relationships"]:
                relationship = CallRelationship(
                    caller=rel_data["caller"],
                    callee=rel_data["callee"],
                    call_line=rel_data["call_line"],
                    is_resolved=rel_data.get("is_resolved", False)
                )
                relationships.append(relationship)
            
            # Create summary
            summary = {
                "total_functions": len(functions),
                "total_relationships": len(relationships),
                "resolved_relationships": len([r for r in relationships if r.is_resolved]),
                "files_analyzed": result["call_graph"]["files_analyzed"],
                "languages": result["call_graph"]["languages_found"]
            }
            
            return AnalysisResult(
                repository=repository,
                functions=functions,
                relationships=relationships,
                file_tree=result.get("file_tree", {}),
                summary=summary
            )
            
        except Exception as e:
            logger.error(f"Analysis failed for {repo_url}: {e}")
            # Return error result
            return AnalysisResult(
                repository=Repository(
                    owner="unknown",
                    name="error",
                    url=repo_url,
                    analyzed_at=datetime.utcnow()
                ),
                functions=[],
                relationships=[],
                file_tree={},
                summary={"error": str(e), "success": False}
            )
    
    async def analyze_repository(self, repo_url: str) -> AnalysisResult:
        """Analyze repository without filters (backwards compatibility)."""
        return await self.analyze_repository_with_filters(repo_url, None) 