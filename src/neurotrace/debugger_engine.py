import logging
import os
import sys
import sysconfig
import threading
from dataclasses import dataclass
from typing import Dict, List, Optional, Union, Any

from .log_interceptor import LogInterceptor
from .ollama_ai_adapter import OllamaAIAdapter, OllamaConfig
from .runtime_visualizer import RuntimeVisualizer, TraceEvent

@dataclass
class DebuggerConfig:
    """Configuration for the DebuggerEngine."""
    enable_visualizer: bool = True
    ai_model: str = "phi4"
    log_level: str = "ERROR"  # Default to ERROR level to hide debug messages
    max_log_size: int = 10000
    ai_timeout: int = 60
    diagram_format: str = "png"
    diagram_theme: str = "default"
    verbose_mode: bool = False  # Control debug message visibility

    live_tracing: bool = False
    line_level_tracing: bool = False

def _is_stdlib_or_sitepkg(path: str) -> bool:
    abs_path = os.path.abspath(path)
    std_lib = sysconfig.get_paths()['stdlib']
    site_pkg = sysconfig.get_paths()['purelib']
    return abs_path.startswith(std_lib) or abs_path.startswith(site_pkg)

def _is_in_user_dir(script_dir: str, path: str) -> bool:
    abs_path = os.path.abspath(path)
    try:
        rel_path = os.path.relpath(abs_path, script_dir)
        return not rel_path.startswith('..')
    except ValueError:
        return False

class DebuggerEngine:
    """Central orchestrator for NeuroTrace debugging capabilities."""

    def __init__(
        self,
        config: Optional[DebuggerConfig] = None,
        ai_adapter_config: Optional[Dict] = None,
        log_interceptor_config: Optional[Dict] = None,
        script_path: Optional[str] = None
    ):
        self.config = config or DebuggerConfig()
        self._script_path = script_path
        self._is_debugging = False
        self._previous_tracer: Optional[Any] = None
        self._trace_data: List[TraceEvent] = []
        self._trace_lock = threading.Lock()
        self._logger = logging.getLogger(__name__)

        self._setup_logging()

        # Configure LogInterceptor
        log_config = {
            "max_logs": self.config.max_log_size,
            "verbose_mode": self.config.verbose_mode
        }
        if log_interceptor_config:
            if "max_logs" in log_interceptor_config:
                log_config["max_logs"] = log_interceptor_config["max_logs"]
            if "verbose_mode" in log_interceptor_config:
                log_config["verbose_mode"] = log_interceptor_config["verbose_mode"]
        self.log_interceptor = LogInterceptor(**log_config)

        # Configure AI Adapter
        ai_config = OllamaConfig(
            timeout=self.config.ai_timeout,
            model=self.config.ai_model
        )
        if ai_adapter_config and "config" in ai_adapter_config:
            # If a complete config is provided, use it
            ai_config = ai_adapter_config["config"]
        elif ai_adapter_config:
            # Otherwise update individual fields
            for k, v in ai_adapter_config.items():
                if hasattr(ai_config, k):
                    setattr(ai_config, k, v)
        self.ai_adapter = OllamaAIAdapter(config=ai_config)

        # Configure Visualizer
        self.visualizer = None
        if self.config.enable_visualizer:
            try:
                self.visualizer = RuntimeVisualizer(
                    output_format=self.config.diagram_format,
                    theme=self.config.diagram_theme
                )
            except RuntimeError as e:
                self._logger.error(f"Failed to initialize visualizer: {e}")
                self.config.enable_visualizer = False

    def _setup_logging(self) -> None:
        level = getattr(logging, self.config.log_level.upper(), logging.INFO)
        logging.basicConfig(level=level)

    def _is_user_code(self, frame) -> bool:
        """
        Return True if this frame belongs to user code:
         - Not in 'neurotrace' or other known libs
         - Not stdlib/site-packages
         - Inside the user script directory if script_path is known
        """
        frame_file = os.path.abspath(frame.f_code.co_filename)

        # Ignore <string> or <frozen importlib> frames
        if "<" in frame_file or ">" in frame_file:
            return False

        # If it's from 'neurotrace', skip it entirely
        lower_path = frame_file.replace("\\", "/").lower()
        if "neurotrace" in lower_path:
            return False

        # Skip standard library or site-packages
        if _is_stdlib_or_sitepkg(frame_file):
            return False

        # If we have a script_path, only consider frames in that directory
        if self._script_path:
            script_dir = os.path.dirname(os.path.abspath(self._script_path))
            return _is_in_user_dir(script_dir, frame_file)
        return False

    def _trace_callback(self, frame, event, arg):
        try:
            if not self._is_debugging:
                return None

            if event == "line" and not self.config.line_level_tracing:
                return self._trace_callback

            if event == "exception":
                exc_type, exc_value, _ = arg
                if self.config.verbose_mode:
                    self._logger.debug(f"Exception caught: {exc_type.__name__}: {exc_value}")
                elif self._is_user_code(frame):
                    self._logger.error(f"{exc_type.__name__}: {exc_value}")
                if self._is_user_code(frame):
                    error_evt = TraceEvent(
                        event_type="error",
                        function_name=frame.f_code.co_name,
                        module_name=frame.f_globals.get("__name__", "unknown"),
                        line_number=frame.f_lineno
                    )
                    self.add_trace_event(error_evt)
                return self._trace_callback

            if event in ("call", "return", "line") and self._is_user_code(frame):
                func_name = frame.f_code.co_name
                mod_name = frame.f_globals.get("__name__", "unknown")
                line_no = frame.f_lineno
                caller = None

                if event == "call":
                    caller_frame = frame.f_back
                    if caller_frame:
                        caller_func = caller_frame.f_code.co_name
                        caller_mod = caller_frame.f_globals.get("__name__", "unknown")
                        caller = f"{caller_mod}.{caller_func}"

                evt = TraceEvent(
                    event_type=event,
                    function_name=func_name,
                    module_name=mod_name,
                    line_number=line_no,
                    caller=caller
                )
                self.add_trace_event(evt)
                if self.config.verbose_mode:
                    self._logger.debug(f"Traced user code {event}: {mod_name}.{func_name}")
        except Exception as e:
            self._logger.error(f"Error in trace callback: {str(e)}")
            return None
        return self._trace_callback

    def start_debugging(self) -> None:
        if self._is_debugging:
            self._logger.warning("Debugging session already active")
            return
        try:
            self.log_interceptor.start()
            self._is_debugging = True
            self._trace_data.clear()

            if self.config.live_tracing:
                self._previous_tracer = sys.gettrace()
                sys.settrace(self._trace_callback)

            if self.config.verbose_mode:
                self._logger.info("Debugging session started")
        except Exception as e:
            self._logger.error(f"Failed to start debugging: {str(e)}")
            self._is_debugging = False
            raise

    def stop_debugging(self) -> None:
        if not self._is_debugging:
            self._logger.warning("No active debugging session")
            return
        try:
            if self.config.live_tracing:
                self._logger.debug("Restoring previous tracer")
                sys.settrace(self._previous_tracer)
                self._previous_tracer = None

            self.log_interceptor.stop()
            self._is_debugging = False
            if self.config.verbose_mode:
                self._logger.info("Debugging session stopped")
        except Exception as e:
            self._logger.error(f"Failed to stop debugging: {str(e)}")
            sys.settrace(None)
            self._previous_tracer = None
            self._is_debugging = False
            raise

    def analyze_logs(self, system_prompt: Optional[str] = None) -> Dict[str, Union[bool, str, Dict]]:
        if not self._is_debugging:
            return {
                "success": False,
                "error": "No active debugging session",
                "analysis": None
            }
        try:
            logs = self.log_interceptor.get_logs()
            if not logs:
                return {
                    "success": False,
                    "error": "No logs captured",
                    "analysis": None
                }
            log_text = "\n".join(logs)
            analysis = self.ai_adapter.get_ai_analysis(
                log_text,
                model=self.config.ai_model,
                system_prompt=system_prompt
            )
            
            if (analysis.get("success", False) and 
                isinstance(analysis.get("analysis"), dict) and 
                "response" in analysis["analysis"] and
                analysis["analysis"]["response"]):  # Only if there's actual content
                # Format the output nicely
                print("\033[32mAI Analysis Results:\033[0m")  # Green header
                # Print response with minimal spacing and strip any extra whitespace
                print(analysis["analysis"]["response"].strip())  # Response is already colored
                return {"success": True}  # Simple success response
            return {"success": False}  # Simple failure response
        except Exception as e:
            self._logger.error(f"Failed to analyze logs: {str(e)}")
            return {
                "success": False,
                "error": f"Analysis failed: {str(e)}",
                "analysis": None
            }

    def add_trace_event(self, event: TraceEvent) -> None:
        if not self._is_debugging or not self.config.enable_visualizer:
            return
        with self._trace_lock:
            self._trace_data.append(event)

    def generate_visual(self, output_path: str = "debug_trace") -> Optional[str]:
        if not self.config.enable_visualizer or not self.visualizer:
            self._logger.warning("Visualization is disabled")
            return None

        if not self._trace_data:
            self._logger.warning("No trace events captured - generating basic error diagram")
            error_evt = TraceEvent(
                event_type="error",
                function_name="main",
                module_name="__main__",
                line_number=0
            )
            self._trace_data.append(error_evt)

        try:
            with self._trace_lock:
                return self.visualizer.generate_diagram(self._trace_data, output_path)
        except Exception as e:
            self._logger.error(f"Failed to generate diagram: {str(e)}")
            return None

    def clear_logs(self) -> None:
        self.log_interceptor.clear()
        with self._trace_lock:
            self._trace_data.clear()

    def __enter__(self):
        self.start_debugging()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop_debugging()
