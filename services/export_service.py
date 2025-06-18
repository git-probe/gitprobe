"""
Export Service
"""

import json
from datetime import datetime
from typing import Dict, List, Optional, Any

from models.analysis import AnalysisResult, NodeSelection
from models.export import ExportData


class ExportService:
    """Export service. Engineering is simplicity."""

    def __init__(self):
        pass

    def export_json(
        self, analysis: AnalysisResult, selection: Optional[NodeSelection] = None
    ) -> ExportData:
        """Export analysis as JSON (backwards compatibility)."""
        return self.export_json_configurable(analysis, selection, True, True)

    def export_json_configurable(
        self,
        analysis: AnalysisResult,
        selection: Optional[NodeSelection] = None,
        include_code: bool = True,
        include_relationships: bool = True,
    ) -> ExportData:
        """Export analysis as JSON with configurable options."""

        # Filter data if selection provided
        if selection and selection.selected_nodes:
            selected_functions = [
                f
                for f in analysis.functions
                if f"{f.file_path}:{f.name}" in selection.selected_nodes
            ]

            # Apply custom names and optionally remove code
            for func in selected_functions:
                func_id = f"{func.file_path}:{func.name}"
                if func_id in selection.custom_names:
                    # Keep original name but add display_name for LLM context
                    func_dict = func.dict()
                    func_dict["display_name"] = selection.custom_names[func_id]
                    func_dict["original_name"] = func.name

                if not include_code:
                    func.code_snippet = None

            # Filter relationships if needed
            if include_relationships:
                selected_relationships = [
                    r
                    for r in analysis.relationships
                    if (
                        r.caller in selection.selected_nodes
                        or r.callee in selection.selected_nodes
                    )
                    and r.is_resolved
                ]
            else:
                selected_relationships = []

            filtered_analysis = AnalysisResult(
                repository=analysis.repository,
                functions=selected_functions,
                relationships=selected_relationships,
                file_tree=analysis.file_tree if selection.selected_nodes else {},
                summary={
                    **analysis.summary,
                    "filtered": True,
                    "selected_count": len(selected_functions),
                    "includes_code": include_code,
                    "includes_relationships": include_relationships,
                },
            )
        else:
            # Full analysis
            functions = analysis.functions.copy()
            if not include_code:
                for func in functions:
                    func.code_snippet = None

            relationships = analysis.relationships if include_relationships else []

            filtered_analysis = AnalysisResult(
                repository=analysis.repository,
                functions=functions,
                relationships=relationships,
                file_tree=analysis.file_tree,
                summary={
                    **analysis.summary,
                    "includes_code": include_code,
                    "includes_relationships": include_relationships,
                },
            )

        return ExportData(
            analysis=filtered_analysis,
            selection=selection,
            export_type="json",
            generated_at=datetime.utcnow(),
        )

    def export_cytoscape(
        self, analysis: AnalysisResult, selection: Optional[NodeSelection] = None
    ) -> Dict[str, Any]:
        """Export as Cytoscape.js format."""

        functions = analysis.functions
        relationships = analysis.relationships

        # Filter if selection
        if selection and selection.selected_nodes:
            functions = [
                f
                for f in functions
                if f"{f.file_path}:{f.name}" in selection.selected_nodes
            ]
            relationships = [
                r
                for r in relationships
                if (
                    r.caller in selection.selected_nodes
                    or r.callee in selection.selected_nodes
                )
            ]

        # Create elements
        elements = []

        # Add nodes
        for func in functions:
            func_id = f"{func.file_path}:{func.name}"
            display_name = func.name

            # Apply custom name if exists
            if selection and func_id in selection.custom_names:
                display_name = selection.custom_names[func_id]

            elements.append(
                {
                    "data": {
                        "id": func_id,
                        "label": display_name,
                        "file": func.file_path,
                        "type": "method" if func.is_method else "function",
                        "class": func.class_name or "",
                    },
                    "classes": f"node-{'method' if func.is_method else 'function'}",
                }
            )

        # Add edges
        for rel in relationships:
            if rel.is_resolved:
                elements.append(
                    {
                        "data": {
                            "id": f"{rel.caller}->{rel.callee}",
                            "source": rel.caller,
                            "target": rel.callee,
                            "line": rel.call_line,
                        },
                        "classes": "edge-call",
                    }
                )

        return {
            "elements": elements,
            "style": [
                {
                    "selector": "node",
                    "style": {
                        "background-color": "#3498db",
                        "label": "data(label)",
                        "width": "60px",
                        "height": "60px",
                        "text-valign": "center",
                        "text-halign": "center",
                        "font-size": "12px",
                    },
                },
                {"selector": ".node-method", "style": {"background-color": "#e74c3c"}},
                {
                    "selector": ".node-function",
                    "style": {"background-color": "#2ecc71"},
                },
                {
                    "selector": "edge",
                    "style": {
                        "width": 2,
                        "line-color": "#34495e",
                        "target-arrow-color": "#34495e",
                        "target-arrow-shape": "triangle",
                        "curve-style": "bezier",
                    },
                },
            ],
        }

    def export_svg(
        self,
        analysis: AnalysisResult,
        selection: Optional[NodeSelection] = None,
        width: int = 800,
        height: int = 600,
    ) -> str:
        """Export as simple SVG."""

        functions = analysis.functions
        relationships = analysis.relationships

        # Filter if selection
        if selection and selection.selected_nodes:
            functions = [
                f
                for f in functions
                if f"{f.file_path}:{f.name}" in selection.selected_nodes
            ]

        # Simple circular layout
        import math

        center_x, center_y = width // 2, height // 2
        radius = min(width, height) // 3

        svg_parts = [
            f'<svg width="{width}" height="{height}" xmlns="http://www.w3.org/2000/svg">',
            "<style>",
            ".function { fill: #2ecc71; stroke: #27ae60; }",
            ".method { fill: #e74c3c; stroke: #c0392b; }",
            ".label { font-family: Arial; font-size: 10px; text-anchor: middle; }",
            "</style>",
        ]

        # Draw nodes
        for i, func in enumerate(functions):
            angle = 2 * math.pi * i / len(functions) if len(functions) > 1 else 0
            x = center_x + radius * math.cos(angle)
            y = center_y + radius * math.sin(angle)

            node_class = "method" if func.is_method else "function"
            display_name = func.name

            if selection and f"{func.file_path}:{func.name}" in selection.custom_names:
                display_name = selection.custom_names[f"{func.file_path}:{func.name}"]

            svg_parts.append(
                f'<circle cx="{x}" cy="{y}" r="20" class="{node_class}" />'
            )
            svg_parts.append(
                f'<text x="{x}" y="{y+4}" class="label">{display_name[:10]}</text>'
            )

        svg_parts.append("</svg>")
        return "\n".join(svg_parts)
