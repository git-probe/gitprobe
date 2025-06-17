from pydantic import BaseModel
from typing import Optional
from datetime import datetime
from .analysis import AnalysisResult, NodeSelection


class ExportData(BaseModel):
    """Data prepared for export"""
    analysis: AnalysisResult
    selection: Optional[NodeSelection] = None
    export_type: str = "json"
    generated_at: datetime 