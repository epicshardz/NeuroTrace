import os
import shutil
import pytest
from unittest.mock import Mock, patch
import graphviz
from neurotrace.runtime_visualizer import RuntimeVisualizer, TraceEvent

@pytest.fixture
def visualizer():
    """Create a RuntimeVisualizer instance."""
    return RuntimeVisualizer()

@pytest.fixture
def mock_digraph():
    """Mock Graphviz Digraph."""
    with patch('graphviz.Digraph') as mock:
        # Configure the mock to return itself for method chaining
        instance = mock.return_value
        instance.attr.return_value = instance
        instance.node.return_value = instance
        instance.edge.return_value = instance
        instance.render.return_value = "test_output.png"
        yield mock

def test_initialization():
    """Test visualizer initialization with different parameters."""
    # Default initialization
    vis = RuntimeVisualizer()
    assert vis.output_format == "png"
    assert vis.include_line_numbers is True
    assert vis.theme == "default"
    
    # Custom initialization
    vis = RuntimeVisualizer(
        output_format="svg",
        include_line_numbers=False,
        theme="dark"
    )
    assert vis.output_format == "svg"
    assert vis.include_line_numbers is False
    assert vis.theme == "dark"

def test_invalid_format():
    """Test initialization with invalid format."""
    with pytest.raises(ValueError, match="Output format must be 'png' or 'svg'"):
        RuntimeVisualizer(output_format="invalid")

def test_invalid_theme():
    """Test initialization with invalid theme."""
    with pytest.raises(ValueError, match="Invalid theme"):
        RuntimeVisualizer(theme="nonexistent")

def test_simple_call_stack(visualizer, mock_digraph):
    """Test generating a diagram for a simple call stack."""
    trace_data = [
        TraceEvent(
            event_type="call",
            function_name="main",
            module_name="test_module",
            line_number=1
        ),
        TraceEvent(
            event_type="call",
            function_name="helper",
            module_name="test_module",
            line_number=5,
            caller="test_module.main"
        )
    ]
    
    output_file = visualizer.generate_diagram(trace_data)
    assert output_file == "test_output.png"
    
    # Verify graph construction
    mock_digraph.assert_called_once()
    instance = mock_digraph.return_value
    assert instance.node.call_count == 2
    assert instance.edge.call_count == 1

def test_complex_call_stack(visualizer, mock_digraph):
    """Test generating a diagram for a complex call stack with multiple paths."""
    trace_data = [
        TraceEvent("call", "main", "module_a", 1),
        TraceEvent("call", "helper1", "module_b", 5, "module_a.main"),
        TraceEvent("call", "helper2", "module_b", 10, "module_a.main"),
        TraceEvent("call", "sub_helper", "module_c", 15, "module_b.helper1"),
        TraceEvent("return", "sub_helper", "module_c", 15),
        TraceEvent("return", "helper1", "module_b", 5),
        TraceEvent("return", "helper2", "module_b", 10),
        TraceEvent("return", "main", "module_a", 1)
    ]
    
    visualizer.generate_diagram(trace_data)
    
    instance = mock_digraph.return_value
    # Should have 4 nodes (main, helper1, helper2, sub_helper)
    assert instance.node.call_count == 4
    # Should have 3 edges (main->helper1, main->helper2, helper1->sub_helper)
    assert instance.edge.call_count == 3

def test_empty_trace_data(visualizer):
    """Test handling of empty trace data."""
    with pytest.raises(ValueError, match="No trace data provided"):
        visualizer.generate_diagram([])

def test_theme_configuration(mock_digraph):
    """Test different theme configurations."""
    themes = ["default", "light", "dark"]
    for theme in themes:
        visualizer = RuntimeVisualizer(theme=theme)
        trace_data = [TraceEvent("call", "main", "test", 1)]
        visualizer.generate_diagram(trace_data)
        
        instance = mock_digraph.return_value
        theme_config = visualizer.themes[theme]
        
        # Verify graph attributes
        graph_attrs = next(
            call.kwargs for call in instance.attr.call_args_list 
            if not call.args or call.args[0] not in {"node", "edge"}
        )
        assert graph_attrs["bgcolor"] == theme_config["background"]
        assert graph_attrs["fontname"] == theme_config["font"]
        
        # Verify node attributes
        node_attrs = next(
            call.kwargs for call in instance.attr.call_args_list 
            if call.args and call.args[0] == "node"
        )
        assert node_attrs["fillcolor"] == theme_config["node_color"]
        assert node_attrs["fontname"] == theme_config["font"]
        
        # Verify edge attributes
        edge_attrs = next(
            call.kwargs for call in instance.attr.call_args_list 
            if call.args and call.args[0] == "edge"
        )
        assert edge_attrs["color"] == theme_config["edge_color"]
        assert edge_attrs["fontname"] == theme_config["font"]

def test_theme_specific_colors():
    """Test that each theme has distinct colors."""
    visualizer = RuntimeVisualizer()
    themes = visualizer.themes
    
    # Each theme should have unique colors
    node_colors = {theme: config["node_color"] for theme, config in themes.items()}
    edge_colors = {theme: config["edge_color"] for theme, config in themes.items()}
    backgrounds = {theme: config["background"] for theme, config in themes.items()}
    
    # Dark theme should have light text
    dark_config = themes["dark"]
    assert dark_config["background"] == "#121212"  # Dark background
    assert dark_config["node_color"] == "#BB86FC"  # Light purple nodes
    assert dark_config["edge_color"] == "#03DAC6"  # Teal edges

def test_graphviz_initialization():
    """Test Graphviz initialization and error handling."""
    # Test missing Graphviz executable
    with patch('shutil.which', return_value=None):
        with pytest.raises(RuntimeError) as exc_info:
            RuntimeVisualizer()
        assert "Graphviz system executable (dot) is not installed" in str(exc_info.value)
        assert "Please install Graphviz from https://graphviz.org/download/" in str(exc_info.value)
    
    # Test Graphviz render failure
    visualizer = RuntimeVisualizer()
    trace_data = [TraceEvent("call", "main", "test", 1)]
    
    with patch('graphviz.Digraph.render') as mock_render:
        mock_render.side_effect = graphviz.ExecutableNotFound(['dot'])
        with pytest.raises(RuntimeError) as exc_info:
            visualizer.generate_diagram(trace_data)
        assert "Graphviz executable not found" in str(exc_info.value)

def test_output_path_handling(visualizer, mock_digraph, tmp_path):
    """Test output path handling."""
    # Test creating nested directories
    output_dir = tmp_path / "subdir" / "graphs"
    output_path = str(output_dir / "test_diagram")
    
    trace_data = [TraceEvent("call", "main", "test", 1)]
    visualizer.generate_diagram(trace_data, output_path)
    
    assert os.path.dirname(output_path) == str(output_dir)
    
    # Test handling invalid paths
    with pytest.raises(ValueError) as exc_info:
        visualizer.generate_diagram(trace_data, "")
    assert "Invalid output path" in str(exc_info.value)
    
    # Test handling paths with invalid characters
    with pytest.raises(ValueError) as exc_info:
        visualizer.generate_diagram(trace_data, "test<>:diagram")
    assert "Invalid characters in output path" in str(exc_info.value)

def test_node_label_formatting(visualizer):
    """Test node label formatting with different options."""
    # Test with all options enabled
    label = visualizer._create_node_label("test_func", "test_module", 42)
    assert "Module: test_module" in label
    assert "Function: test_func" in label
    assert "Line: 42" in label
    
    # Test with line numbers disabled
    visualizer.include_line_numbers = False
    label = visualizer._create_node_label("test_func", "test_module", 42)
    assert "Line: 42" not in label
    
    # Test with modules disabled
    visualizer.include_modules = False
    label = visualizer._create_node_label("test_func", "test_module", 42)
    assert "Module: test_module" not in label
