import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Set, Tuple
import graphviz

@dataclass
class TraceEvent:
    """Represents a single function call trace event."""
    event_type: str  # 'call' or 'return'
    function_name: str
    module_name: str
    line_number: int
    caller: Optional[str] = None  # Format: "module.function"
    timestamp: Optional[float] = None

class RuntimeVisualizer:
    """Generates visual diagrams of runtime call stacks."""

    def __init__(
        self,
        output_format: str = "png",
        include_line_numbers: bool = True,
        include_modules: bool = True,
        theme: str = "default"
    ):
        """Check if Graphviz system executable is installed first"""
        import shutil
        if not shutil.which('dot'):
            raise RuntimeError(
                "Graphviz system executable (dot) is not installed.\n"
                "The Python graphviz package is installed, but the system executable is missing.\n"
                "Please install Graphviz from https://graphviz.org/download/\n"
                "After installation, you may need to restart your terminal/IDE."
            )
            
        """Initialize the runtime visualizer.
        
        Args:
            output_format: Output file format ('png' or 'svg')
            include_line_numbers: Whether to show line numbers in nodes
            include_modules: Whether to show module names in nodes
            theme: Visual theme for the diagram ('default', 'light', or 'dark')
        """
        self.output_format = output_format.lower()
        if self.output_format not in {"png", "svg"}:
            raise ValueError("Output format must be 'png' or 'svg'")
        
        self.include_line_numbers = include_line_numbers
        self.include_modules = include_modules
        self.theme = theme
        
        # Theme configurations
        self.themes = {
            "default": {
                "node_color": "#2B2B2B",
                "edge_color": "#4A4A4A",
                "background": "white",
                "font": "Arial",
            },
            "light": {
                "node_color": "#6B9080",
                "edge_color": "#A4C3B2",
                "background": "#F6FFF8",
                "font": "Arial",
            },
            "dark": {
                "node_color": "#BB86FC",
                "edge_color": "#03DAC6",
                "background": "#121212",
                "font": "Consolas",
            }
        }

    def _create_node_label(
        self,
        function_name: str,
        module_name: str,
        line_number: Optional[int] = None
    ) -> str:
        """Create a formatted node label.
        
        Args:
            function_name: Name of the function
            module_name: Name of the module
            line_number: Line number where the function is defined
            
        Returns:
            Formatted label string
        """
        parts = []
        if self.include_modules:
            parts.append(f"Module: {module_name}")
        parts.append(f"Function: {function_name}")
        if self.include_line_numbers and line_number is not None:
            parts.append(f"Line: {line_number}")
        return "\n".join(parts)

    def _get_node_id(self, module_name: str, function_name: str) -> str:
        """Generate a unique node ID for a function.
        
        Args:
            module_name: Name of the module
            function_name: Name of the function
            
        Returns:
            Unique node identifier
        """
        return f"{module_name}.{function_name}"

    def _setup_graph(self) -> graphviz.Digraph:
        """Create and configure a new Graphviz diagram.
        
        Returns:
            Configured Graphviz Digraph object
        """
        theme = self.themes.get(self.theme, self.themes["default"])
        
        dot = graphviz.Digraph(
            comment="Runtime Call Stack Visualization",
            format=self.output_format
        )
        
        # Graph attributes
        dot.attr(
            rankdir="TB",
            bgcolor=theme["background"],
            fontname=theme["font"]
        )
        
        # Default node attributes
        dot.attr(
            "node",
            shape="box",
            style="rounded,filled",
            fillcolor=theme["node_color"],
            color=theme["node_color"],
            fontcolor="white" if self.theme == "dark" else "black",
            fontname=theme["font"]
        )
        
        # Default edge attributes
        dot.attr(
            "edge",
            color=theme["edge_color"],
            fontcolor=theme["edge_color"],
            fontname=theme["font"]
        )
        
        return dot

    def generate_diagram(
        self,
        trace_data: List[TraceEvent],
        output_path: str = "runtime_diagram"
    ) -> str:
        """Generate a call stack diagram from trace data.
        
        Args:
            trace_data: List of TraceEvent objects representing the call stack
            output_path: Path where the diagram should be saved (without extension)
            
        Returns:
            Path to the generated diagram file
            
        Raises:
            graphviz.ExecutableNotFound: If Graphviz is not installed
            ValueError: If trace_data is empty or invalid
        """
        if not trace_data:
            raise ValueError("No trace data provided")

        # Create new diagram
        dot = self._setup_graph()
        
        # Track nodes and edges to avoid duplicates
        nodes: Set[str] = set()
        edges: Set[Tuple[str, str]] = set()
        
        # Process trace events
        for event in trace_data:
            if event.event_type not in {"call", "return", "error"}:
                continue
                
            # For error events, create a special error node
            if event.event_type == "error":
                label = "Error occurred in main script"
                dot.node("error", label, color="red", style="filled", fillcolor="#ffcccc")
                continue
                
            # Create node for current function if it doesn't exist
            node_id = self._get_node_id(event.module_name, event.function_name)
            if node_id not in nodes:
                label = self._create_node_label(
                    event.function_name,
                    event.module_name,
                    event.line_number
                )
                # Use red color for nodes involved in error path
                if event.module_name == "__main__":
                    dot.node(node_id, label, color="red", style="filled", fillcolor="#ffcccc")
                else:
                    dot.node(node_id, label)
                nodes.add(node_id)
            
            # Add edge for function calls
            if event.event_type == "call" and event.caller:
                caller_module, caller_func = event.caller.split(".")
                caller_id = self._get_node_id(caller_module, caller_func)
                edge = (caller_id, node_id)
                if edge not in edges:
                    dot.edge(*edge)
                    edges.add(edge)

        # Ensure output directory exists
        output_dir = os.path.dirname(output_path)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)

        # Render the diagram
        try:
            output_file = dot.render(
                filename=output_path,
                cleanup=True  # Remove the DOT source file
            )
            return output_file
        except graphviz.ExecutableNotFound:
            raise RuntimeError(
                "Graphviz executable not found. Please install Graphviz on your system."
            )

    def clear(self) -> None:
        """Clear any stored state (if implemented in future)."""
        pass
