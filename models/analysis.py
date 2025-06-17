from pydantic import BaseModel
from typing import List, Optional, Dict, Any
from datetime import datetime
from .core import Function, CallRelationship, Repository


class AnalysisResult(BaseModel):
    """Result of analyzing a repository"""
    id: str
    repository: Repository
    functions: List[Function]
    relationships: List[CallRelationship]
    file_tree: Dict[str, Any]
    created_at: datetime


class NodeSelection(BaseModel):
    """Selected nodes for partial export"""
    selected_nodes: List[str]
    include_relationships: bool = True 