from pydantic import BaseModel
from typing import List, Optional, Dict, Any
from datetime import datetime
from .core import Function, CallRelationship, Repository


class AnalysisResult(BaseModel):
    """Result of analyzing a repository"""
    repository: Repository
    functions: List[Function]
    relationships: List[CallRelationship]
    file_tree: Dict[str, Any]
    summary: Dict[str, Any]


class NodeSelection(BaseModel):
    """Selected nodes for partial export"""
    selected_nodes: List[str] = []
    include_relationships: bool = True
    custom_names: Dict[str, str] = {} 