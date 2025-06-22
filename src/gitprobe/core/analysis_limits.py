"""
Shared Analysis Limits

Common analysis limits and performance controls used across all language analyzers.
Ensures consistent behavior and resource management across Python, JavaScript, Go, Rust, etc.
"""

import time
import logging
from typing import Optional

logger = logging.getLogger(__name__)


class GlobalLimitTracker:
    """
    Global limit tracker shared across ALL language analyzers.
    Enforces hard caps on total functions and relationships across the entire analysis.
    """
    
    def __init__(self, max_total_functions: int = 5000, max_total_relationships: int = 8000):
        self.max_total_functions = max_total_functions
        self.max_total_relationships = max_total_relationships
        self.total_functions = 0
        self.total_relationships = 0
        self.global_limit_reached = False
    
    def can_add_function(self) -> bool:
        """Check if we can add another function without exceeding global limits."""
        if self.global_limit_reached:
            return False
        return self.total_functions < self.max_total_functions
    
    def can_add_relationship(self) -> bool:
        """Check if we can add another relationship without exceeding global limits."""
        if self.global_limit_reached:
            return False
        return self.total_relationships < self.max_total_relationships
    
    def add_function(self) -> bool:
        """Add a function and return True if global limit reached."""
        if self.global_limit_reached:
            return True
            
        self.total_functions += 1
        if self.total_functions >= self.max_total_functions:
            logger.warning(f"Global function limit reached: {self.max_total_functions}")
            self.global_limit_reached = True
            return True
        return False
    
    def add_relationship(self) -> bool:
        """Add a relationship and return True if global limit reached."""
        if self.global_limit_reached:
            return True
            
        self.total_relationships += 1
        if self.total_relationships >= self.max_total_relationships:
            logger.warning(f"Global relationship limit reached: {self.max_total_relationships}")
            self.global_limit_reached = True
            return True
        return False
    
    def should_stop(self) -> bool:
        """Check if analysis should stop due to global limits."""
        return self.global_limit_reached


# Global instance to be shared across all analyzers
_global_tracker = None

def get_global_tracker() -> GlobalLimitTracker:
    """Get the global limit tracker instance."""
    global _global_tracker
    if _global_tracker is None:
        _global_tracker = GlobalLimitTracker()
    return _global_tracker

def reset_global_tracker():
    """Reset the global tracker (for testing or new analysis runs)."""
    global _global_tracker
    _global_tracker = GlobalLimitTracker()


class AnalysisLimits:
    """
    Unified analysis limits for all language analyzers.
    
    Provides consistent resource management and performance controls across
    Python, JavaScript, TypeScript, Go, Rust, and C/C++ analyzers.
    
    Uses a hybrid approach with both per-file and global limits to ensure:
    - Breadth over depth (sample from many files vs exhaustive analysis of few)
    - Fast analysis for real-time LLM interactions  
    - Representative sampling across the codebase
    - Consistent resource usage across languages
    """
    
    def __init__(
        self, 
        max_nodes_per_file: int = 3000,    # Higher limit per file for large codebases
        max_time_per_file: float = 15.0,   # More generous time per file
        max_files_analyzed: int = 999999,  # Effectively unlimited files
        max_total_time: float = 180.0,     # 3 minutes total per language
        language: str = "unknown"
    ):
        # Configuration
        self.max_nodes_per_file = max_nodes_per_file
        self.max_time_per_file = max_time_per_file
        self.max_files_analyzed = max_files_analyzed
        self.max_total_time = max_total_time
        self.language = language
        
        # Per-file state
        self.nodes_processed = 0
        self.start_time: Optional[float] = None
        self.limit_reached = False
        
        # Global state (shared across all files of this language)
        self.files_analyzed = 0
        self.global_start_time: Optional[float] = None
        self.global_limit_reached = False
        
        # Reference to global tracker
        self.global_tracker = get_global_tracker()
    
    def start_new_file(self) -> bool:
        """
        Start analyzing a new file. Returns True if global limits allow it.
        Resets per-file counters but maintains global state.
        """
        # Check global tracker first
        if self.global_tracker.should_stop():
            logger.info(f"Skipping {self.language} file - global analysis limits reached")
            return False
        
        # Initialize global timer on first file
        if self.global_start_time is None:
            self.global_start_time = time.time()
        
        # Check global limits before starting new file
        if self.files_analyzed >= self.max_files_analyzed:
            logger.info(f"Skipping {self.language} file - reached global file limit: {self.max_files_analyzed}")
            self.global_limit_reached = True
            return False
            
        if self.global_start_time:
            global_elapsed = time.time() - self.global_start_time
            if global_elapsed >= self.max_total_time:
                logger.info(f"Skipping {self.language} file - reached global time limit: {self.max_total_time}s")
                self.global_limit_reached = True
                return False
        
        # Reset per-file counters
        self.nodes_processed = 0
        self.start_time = time.time()
        self.limit_reached = False
        self.files_analyzed += 1
        
        return True
    
    def increment(self) -> bool:
        """
        Increment node counter and check all limits. 
        Returns True if any limit exceeded and analysis should stop.
        """
        if self.limit_reached or self.global_limit_reached:
            return True
        
        self.nodes_processed += 1
        
        # Check per-file limits
        if self.start_time:
            elapsed = time.time() - self.start_time
            if elapsed >= self.max_time_per_file:
                logger.debug(f"{self.language} analysis hit per-file time limit: {self.max_time_per_file}s")
                self.limit_reached = True
                return True
        
        if self.nodes_processed >= self.max_nodes_per_file:
            logger.debug(f"{self.language} analysis hit per-file node limit: {self.max_nodes_per_file} nodes")
            self.limit_reached = True
            return True
        
        # Check global limits
        if self.files_analyzed >= self.max_files_analyzed:
            logger.warning(f"{self.language} analysis hit global file limit: {self.max_files_analyzed} files")
            self.global_limit_reached = True
            return True
            
        if self.global_start_time:
            global_elapsed = time.time() - self.global_start_time
            if global_elapsed >= self.max_total_time:
                logger.warning(f"{self.language} analysis hit global time limit: {self.max_total_time}s")
                self.global_limit_reached = True
                return True
        
        return False
    
    def should_stop(self) -> bool:
        """Check if analysis should stop due to any limits."""
        return self.limit_reached or self.global_limit_reached or self.global_tracker.should_stop()
    
    def can_add_function(self) -> bool:
        """Check if we can add another function without exceeding global limits."""
        return self.global_tracker.can_add_function() and not self.should_stop()
    
    def can_add_relationship(self) -> bool:
        """Check if we can add another relationship without exceeding global limits."""
        return self.global_tracker.can_add_relationship() and not self.should_stop()
    
    def add_function(self) -> bool:
        """Add a function to global count. Returns True if global limit reached."""
        return self.global_tracker.add_function()
    
    def add_relationship(self) -> bool:
        """Add a relationship to global count. Returns True if global limit reached."""
        return self.global_tracker.add_relationship()
    
    def get_stats(self) -> dict:
        """Get current analysis statistics."""
        global_elapsed = 0.0
        if self.global_start_time:
            global_elapsed = time.time() - self.global_start_time
            
        file_elapsed = 0.0
        if self.start_time:
            file_elapsed = time.time() - self.start_time
        
        return {
            "language": self.language,
            "files_analyzed": self.files_analyzed,
            "max_files": self.max_files_analyzed,
            "global_time_elapsed": round(global_elapsed, 2),
            "max_global_time": self.max_total_time,
            "current_file_nodes": self.nodes_processed,
            "max_nodes_per_file": self.max_nodes_per_file,
            "current_file_time": round(file_elapsed, 2),
            "max_time_per_file": self.max_time_per_file,
            "limit_reached": self.limit_reached,
            "global_limit_reached": self.global_limit_reached
        }
    
    def __str__(self) -> str:
        """String representation for logging."""
        return (f"AnalysisLimits({self.language}: "
                f"{self.max_nodes_per_file} nodes/file, "
                f"{self.max_time_per_file}s/file, "
                f"{self.max_total_time}s total)")


# Language-specific factory functions for common configurations
def create_python_limits() -> AnalysisLimits:
    """Create analysis limits optimized for Python files."""
    return AnalysisLimits(
        max_nodes_per_file=300,   # Much stricter - keep total under 1000
        max_time_per_file=5.0,
        max_total_time=60.0,      # 1 minute for Python
        language="python"
    )


def create_javascript_limits() -> AnalysisLimits:
    """Create analysis limits optimized for JavaScript/TypeScript files."""
    return AnalysisLimits(
        max_nodes_per_file=250,   # Much stricter - keep total under 1000
        max_time_per_file=3.0,
        max_total_time=45.0,      # 45 seconds for JS
        language="javascript"
    )


def create_go_limits() -> AnalysisLimits:
    """Create analysis limits optimized for Go files."""
    return AnalysisLimits(
        max_nodes_per_file=200,   # Much stricter - keep total under 1000
        max_time_per_file=3.0,
        max_total_time=30.0,      # 30 seconds for Go
        language="go"
    )


def create_rust_limits() -> AnalysisLimits:
    """Create analysis limits optimized for Rust files."""
    return AnalysisLimits(
        max_nodes_per_file=200,   # Much stricter - keep total under 1000
        max_time_per_file=4.0,
        max_total_time=30.0,      # 30 seconds for Rust
        language="rust"
    )


def create_c_cpp_limits() -> AnalysisLimits:
    """Create analysis limits optimized for C/C++ files."""
    return AnalysisLimits(
        max_nodes_per_file=200,   # Much stricter - keep total under 1000
        max_time_per_file=4.0,
        max_total_time=30.0,      # 30 seconds for C/C++
        language="c_cpp"
    ) 