from pydantic import BaseModel
from typing import List, Optional, Dict, Any
from datetime import datetime


class Function(BaseModel):
    """A function found in the codebase"""
    name: str
    file_path: str
    line_number: int
    display_name: Optional[str] = None  # For custom naming in UI
    
    def get_display_name(self) -> str:
        """Get the name to display (custom or original)"""
        return self.display_name or self.name


class CallRelationship(BaseModel):
    """A call relationship between two functions"""
    caller: str
    callee: str


class Repository(BaseModel):
    """Basic repository information"""
    url: str
    name: str
    clone_path: str
    analysis_id: str 