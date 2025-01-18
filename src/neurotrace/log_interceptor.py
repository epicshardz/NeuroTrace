import sys
import threading
import logging
from collections import deque
from contextlib import contextmanager
from typing import List

class LogInterceptor:
    """Intercepts and stores stdout, stderr, and logging output."""

    _active_interceptors = []
    _global_lock = threading.Lock()

    def __init__(self, max_logs: int = 1000, verbose_mode: bool = False):
        """Initialize the log interceptor with a maximum buffer size.
        
        Args:
            max_logs: Maximum number of log entries to store
            verbose_mode: If True, show DEBUG level logs (default: False)
        """
        self.max_logs = max_logs
        self.buffer = deque(maxlen=max_logs)
        self.verbose_mode = verbose_mode
        self.lock = threading.Lock()
        self._original_stdout = None
        self._original_stderr = None
        self._is_intercepting = False
        self._handler = None

        class InterceptedStream:
            """
            Wraps the original stream (stdout/stderr) so we can:
              1. Store log text into self.interceptor.buffer
              2. Forward the text to the real stream
            """
            def __init__(self, interceptor: "LogInterceptor"):
                self.interceptor = interceptor
                self.original_stream = None

            def set_original_stream(self, stream):
                self.original_stream = stream

            def write(self, text):
                """
                Write text (str or bytes) to both the log buffer and the original stream.
                We'll decode bytes -> str if needed.
                """
                if isinstance(text, bytes):
                    # Attempt UTF-8 decode
                    try:
                        text = text.decode("utf-8", errors="replace")
                    except:
                        text = repr(text)

                if self.interceptor._is_intercepting and self.original_stream:
                    self.interceptor._store_log(text)

                if self.original_stream:
                    self.original_stream.write(text)

            def flush(self):
                if self.original_stream:
                    self.original_stream.flush()

            def isatty(self):
                if self.original_stream and hasattr(self.original_stream, "isatty"):
                    return self.original_stream.isatty()
                return False

            def fileno(self):
                if self.original_stream and hasattr(self.original_stream, "fileno"):
                    return self.original_stream.fileno()
                raise OSError("No fileno available on InterceptedStream.")

            @property
            def closed(self):
                """Some libraries check stream.closed to see if they can write."""
                if self.original_stream and hasattr(self.original_stream, "closed"):
                    return self.original_stream.closed
                return False

        self._stdout_interceptor = InterceptedStream(self)
        self._stderr_interceptor = InterceptedStream(self)

        class BufferHandler(logging.Handler):
            """A logging handler that appends log records to the interceptor's buffer."""
            def __init__(self, interceptor: "LogInterceptor"):
                super().__init__()
                self.interceptor = interceptor

            def emit(self, record: logging.LogRecord):
                if self.interceptor._is_intercepting:
                    # Only store DEBUG logs if verbose mode is enabled
                    if record.levelno != logging.DEBUG or self.interceptor.verbose_mode:
                        msg = self.format(record)
                        # Filter out urllib3 DEBUG logs
                        if "urllib3.connectionpool" in record.name and record.levelno == logging.DEBUG:
                            return
                        # Only include level name for ERROR or higher
                        if record.levelno >= logging.ERROR:
                            self.interceptor._store_log(f"{record.levelname}: {msg}\n")
                        else:
                            self.interceptor._store_log(f"{msg}\n")

        self._handler = BufferHandler(self)
        self._handler.setFormatter(logging.Formatter('%(message)s'))
        self._handler.setLevel(logging.NOTSET)

    def _store_log(self, text: str) -> None:
        """Store a log entry in the buffer, thread-safe."""
        if text.strip():
            with self.lock:
                self.buffer.append(text)

    def start(self) -> None:
        """Begin intercepting logs."""
        if not self._is_intercepting:
            with self._global_lock:
                self._is_intercepting = True
                self._original_stdout = sys.stdout
                self._original_stderr = sys.stderr

                self._stdout_interceptor.set_original_stream(self._original_stdout)
                self._stderr_interceptor.set_original_stream(self._original_stderr)

                sys.stdout = self._stdout_interceptor
                sys.stderr = self._stderr_interceptor

                root_logger = logging.getLogger()
                root_logger.addHandler(self._handler)

                if root_logger.level > logging.NOTSET:
                    root_logger.setLevel(logging.NOTSET)

                self._active_interceptors.append(self)

    def stop(self) -> None:
        """Stop intercepting logs and restore original streams."""
        if self._is_intercepting:
            with self._global_lock:
                logging.getLogger().removeHandler(self._handler)
                self._active_interceptors.remove(self)

                if not self._active_interceptors:
                    sys.stdout = self._original_stdout
                    sys.stderr = self._original_stderr
                else:
                    prev = self._active_interceptors[-1]
                    sys.stdout = prev._stdout_interceptor
                    sys.stderr = prev._stderr_interceptor

                self._is_intercepting = False
                self._original_stdout = None
                self._original_stderr = None

    def get_logs(self) -> List[str]:
        """Return all captured logs."""
        with self.lock:
            return list(self.buffer)

    def clear(self) -> None:
        """Clear all captured logs."""
        with self.lock:
            self.buffer.clear()

    @contextmanager
    def capture(self):
        """
        Context manager to start/stop interception automatically.
        """
        try:
            self.start()
            yield self
        finally:
            self.stop()
