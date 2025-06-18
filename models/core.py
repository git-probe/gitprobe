from pydantic import BaseModel
from typing import List, Optional, Dict, Any
from datetime import datetime


class Function(BaseModel):
    """A function found in the codebase"""

    name: str
    file_path: str
    line_start: int
    line_end: Optional[int] = None
    parameters: Optional[List[str]] = None
    docstring: Optional[str] = None
    is_method: bool = False
    class_name: Optional[str] = None
    code_snippet: Optional[str] = None
    display_name: Optional[str] = None  # For custom naming in UI

    def get_display_name(self) -> str:
        """Get the name to display (custom or original)"""
        return self.display_name or self.name


class CallRelationship(BaseModel):
    """A call relationship between two functions"""

    caller: str
    callee: str
    call_line: Optional[int] = None
    is_resolved: bool = False


class Repository(BaseModel):
    """Basic repository information"""

    url: str
    name: str
    clone_path: str
    analysis_id: str
