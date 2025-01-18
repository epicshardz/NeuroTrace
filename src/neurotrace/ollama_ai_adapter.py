import json
import logging
import time
from dataclasses import dataclass
from typing import Dict, Optional, Union
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# Disable urllib3 logging
logging.getLogger("urllib3").setLevel(logging.WARNING)

@dataclass(frozen=True)
class OllamaConfig:
    """Configuration for the Ollama AI adapter."""
    base_url: str = "http://localhost:11434"
    max_retries: int = 3
    timeout: int = 10
    max_chunk_size: int = 4096
    model: str = "phi4"

    def __post_init__(self):
        """Validate and normalize configuration."""
        # Remove trailing slash from base_url
        object.__setattr__(self, 'base_url', self.base_url.rstrip('/'))
        
        # Validate numeric fields
        if self.max_retries < 0:
            raise ValueError("max_retries must be non-negative")
        if self.timeout < 1:
            raise ValueError("timeout must be at least 1 second")
        if self.max_chunk_size < 1:
            raise ValueError("max_chunk_size must be positive")

class OllamaAIAdapter:
    """Adapter for communicating with a local Ollama AI server."""

    def __init__(
        self,
        config: Optional[OllamaConfig] = None
    ):
        """Initialize the Ollama AI adapter.

        Args:
            config: Configuration for the adapter. If not provided, default configuration will be used.
        """
        self.config = config or OllamaConfig()
        
        # Configure session with retry strategy
        self.session = requests.Session()
        retry_strategy = Retry(
            total=self.config.max_retries,
            backoff_factor=0.5,
            status_forcelist=[500, 502, 503, 504],
            allowed_methods=["POST"],  # Allow retries on POST requests
            raise_on_status=True
        )
        adapter = HTTPAdapter(max_retries=retry_strategy)
        self.session.mount("http://", adapter)
        self.session.mount("https://", adapter)

    def _chunk_log_data(self, log_data: str) -> list[str]:
        """Split log data into chunks if it exceeds max_chunk_size.
        
        Args:
            log_data: The log data to chunk
            
        Returns:
            List of log data chunks
        """
        if len(log_data) <= self.config.max_chunk_size:
            return [log_data]
        
        chunks = []
        for i in range(0, len(log_data), self.config.max_chunk_size):
            chunk = log_data[i:i + self.config.max_chunk_size]
            chunks.append(chunk)
        return chunks

    def _send_request(
        self,
        endpoint: str,
        payload: Dict
    ) -> Dict:
        """Send a request to the Ollama server.
        
        Args:
            endpoint: API endpoint to call
            payload: Request payload
            
        Returns:
            Response data as dictionary
            
        Raises:
            requests.exceptions.RequestException: If the request fails after retries
        """
        url = f"{self.config.base_url}{endpoint}"
        retries = 0
        last_error = None

        while retries <= self.config.max_retries:
            try:
                response = self.session.post(
                    url,
                    json=payload,
                    timeout=self.config.timeout,
                    stream=True
                )
                response.raise_for_status()
                
                # Handle streaming response
                full_response = ""
                for line in response.iter_lines():
                    if line:
                        # Decode the line and accumulate the response
                        json_response = json.loads(line.decode('utf-8'))
                        if 'response' in json_response:
                            full_response += json_response['response']
                
                return {"response": full_response}
            except requests.exceptions.RequestException as e:
                last_error = e
                logging.error(f"Failed to communicate with Ollama server (attempt {retries + 1}/{self.config.max_retries + 1}): {str(e)}")
                retries += 1
                if retries <= self.config.max_retries:
                    time.sleep(0.5 * (2 ** (retries - 1)))  # Exponential backoff
                
        raise last_error if last_error else requests.exceptions.RequestException("Max retries exceeded")

    def get_ai_analysis(
        self,
        log_data: str,
        model: Optional[str] = None,
        system_prompt: Optional[str] = None
    ) -> Dict[str, Union[str, bool, Dict]]:
        """Get AI analysis for the provided log data.
        
        Args:
            log_data: Log data to analyze
            model: Ollama model to use for analysis
            system_prompt: Optional system prompt to guide the analysis
            
        Returns:
            Dictionary containing:
                - success: Boolean indicating if analysis was successful
                - error: Error message if success is False
                - analysis: Analysis results if success is True
                
        Note:
            If log_data exceeds max_chunk_size, it will be split into chunks
            and analyzed separately, with results combined.
        """
        # Animated thinking indicator with bright green color and snake movement
        print("\033[92m", end="")  # Bright green color
        snake_frames = ["🐍   ", " 🐍  ", "  🐍 ", "   🐍", "  🐍 ", " 🐍  "]
        for _ in range(2):  # Two full animations
            for frame in snake_frames:
                print(f"\r{frame} Neurotrace AI is thinking...", end="", flush=True)
                time.sleep(0.2)
        print("\r\n", end="")  # Add newline after animation
        if not log_data.strip():
            return {
                "success": False,
                "error": "Empty log data provided",
                "analysis": None
            }

        try:
            chunks = self._chunk_log_data(log_data)
            combined_analysis = []
            
            for chunk in chunks:
                payload = {
                    "model": model or self.config.model,
                    "prompt": chunk,
                }
                if system_prompt:
                    payload["system"] = system_prompt

                response = self._send_request("/api/generate", payload)
                # Clean up and format the response with green color
                if 'response' in response:
                    cleaned_response = '\n'.join(
                        line for line in response['response'].split('\n')
                        if not any(x in line.lower() for x in ['debug:', 'info:', 'neurotrace:', 'debugging session'])
                    ).strip()
                    response['response'] = f"\033[32m{cleaned_response}\033[0m"  # Green color with reset
                combined_analysis.append(response)

            # If we only had one chunk, return its analysis directly
            if len(combined_analysis) == 1:
                return {
                    "success": True,
                    "error": None,
                    "analysis": combined_analysis[0]
                }
            
            # Otherwise, combine analyses from all chunks
            return {
                "success": True,
                "error": None,
                "analysis": {
                    "chunks": combined_analysis,
                    "total_chunks": len(combined_analysis)
                }
            }

        except requests.exceptions.RequestException as e:
            return {
                "success": False,
                "error": f"Failed to get AI analysis: {str(e)}",
                "analysis": None
            }
        except Exception as e:
            return {
                "success": False,
                "error": f"Unexpected error during analysis: {str(e)}",
                "analysis": None
            }

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.session.close()
