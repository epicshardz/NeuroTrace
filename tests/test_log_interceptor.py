import sys
import logging
import threading
import time
from neurotrace.log_interceptor import LogInterceptor

def test_capture_print():
    """Test that print statements are captured."""
    interceptor = LogInterceptor(max_logs=10)
    with interceptor.capture():
        print("Test message")
    logs = interceptor.get_logs()
    assert "Test message" in logs[0]

def test_capture_stderr():
    """Test that stderr output is captured."""
    interceptor = LogInterceptor(max_logs=10)
    with interceptor.capture():
        print("Error message", file=sys.stderr)
    logs = interceptor.get_logs()
    assert "Error message" in logs[0]

def test_capture_logging():
    """Test that logging messages are captured with proper formatting."""
    interceptor = LogInterceptor(max_logs=10)
    with interceptor.capture():
        logging.error("Test error")
        logging.warning("Test warning")
        logging.info("Test info")
        logging.debug("Test debug")  # Should be filtered out by default
    logs = interceptor.get_logs()
    assert any("ERROR: Test error" in log for log in logs)  # ERROR gets prefix
    assert any("Test warning" in log for log in logs)  # WARNING no prefix
    assert any("Test info" in log for log in logs)  # INFO no prefix
    assert not any("Test debug" in log for log in logs)  # DEBUG filtered out

def test_max_logs():
    """Test that buffer doesn't exceed max_logs."""
    max_logs = 5
    interceptor = LogInterceptor(max_logs=max_logs)
    with interceptor.capture():
        for i in range(10):
            print(f"Message {i}")
    logs = interceptor.get_logs()
    assert len(logs) == max_logs
    # Should only contain the last 5 messages (5-9)
    for i, msg in enumerate(logs):
        assert f"Message {i+5}" in msg

def test_thread_safety():
    """Test that logging from multiple threads works correctly."""
    interceptor = LogInterceptor(max_logs=100)
    
    def log_messages(thread_id, count):
        for i in range(count):
            print(f"Thread {thread_id} message {i}")
            logging.info(f"Thread {thread_id} log {i}")
            time.sleep(0.01)  # Small delay to increase chance of thread overlap
    
    with interceptor.capture():
        threads = [
            threading.Thread(target=log_messages, args=(i, 5))
            for i in range(3)
        ]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
    
    logs = interceptor.get_logs()
    # Should have captured 30 messages (3 threads * 5 messages * 2 types)
    assert len(logs) == 30
    
    # Verify print messages from all threads
    for thread_id in range(3):
        print_messages = [log for log in logs if f"Thread {thread_id} message" in log]
        assert len(print_messages) == 5
        
    # Verify log messages from all threads
    for thread_id in range(3):
        log_messages = [log for log in logs if f"Thread {thread_id} log" in log]
        assert len(log_messages) == 5

def test_log_level_filtering():
    """Test filtering of logs based on level."""
    interceptor = LogInterceptor(max_logs=10, verbose_mode=False)
    
    with interceptor.capture():
        logging.debug("Debug message")  # Should be filtered out
        logging.info("Info message")
        logging.warning("Warning message")
        logging.error("Error message")
        
        # urllib3 debug logs should be filtered
        urllib3_logger = logging.getLogger("urllib3.connectionpool")
        urllib3_logger.debug("Connection debug")
        urllib3_logger.info("Connection info")  # Should be captured
    
    logs = interceptor.get_logs()
    assert len(logs) == 4  # Info, Warning, Error, Connection info
    assert not any("Debug message" in log for log in logs)
    assert not any("Connection debug" in log for log in logs)
    assert any("Error: Error message" in log for log in logs)
    assert any("Connection info" in log for log in logs)

def test_verbose_mode():
    """Test verbose mode with debug logs."""
    interceptor = LogInterceptor(max_logs=10, verbose_mode=True)
    
    with interceptor.capture():
        logging.debug("Debug message")
        logging.info("Info message")
        
        # urllib3 debug logs should still be filtered
        urllib3_logger = logging.getLogger("urllib3.connectionpool")
        urllib3_logger.debug("Connection debug")
    
    logs = interceptor.get_logs()
    assert len(logs) == 2  # Debug and Info
    assert any("Debug message" in log for log in logs)
    assert not any("Connection debug" in log for log in logs)

def test_concurrent_nested_capture():
    """Test nested captures with concurrent operations."""
    outer = LogInterceptor()
    inner = LogInterceptor()
    
    def worker():
        with inner.capture():
            print("Inner thread message")
            time.sleep(0.01)
    
    with outer.capture():
        print("Outer message 1")
        threads = [threading.Thread(target=worker) for _ in range(3)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
        print("Outer message 2")
    
    outer_logs = outer.get_logs()
    inner_logs = inner.get_logs()
    
    assert len(outer_logs) == 5  # Outer 1 + 3 inner + Outer 2
    assert len(inner_logs) == 3  # One per thread

def test_clear_logs():
    """Test that clear() removes all logs."""
    interceptor = LogInterceptor()
    with interceptor.capture():
        print("Test message")
    assert len(interceptor.get_logs()) == 1
    interceptor.clear()
    assert len(interceptor.get_logs()) == 0

def test_nested_capture():
    """Test that nested captures work correctly."""
    outer = LogInterceptor()
    inner = LogInterceptor()
    
    with outer.capture():
        print("Outer message 1")
        with inner.capture():
            print("Inner message")
        print("Outer message 2")
    
    outer_logs = outer.get_logs()
    inner_logs = inner.get_logs()
    
    assert len(outer_logs) == 3  # Should capture all messages
    assert len(inner_logs) == 1  # Should only capture inner message
    assert "Inner message" in inner_logs[0]
    assert all(msg in [log.strip() for log in outer_logs] for msg in [
        "Outer message 1",
        "Inner message",
        "Outer message 2"
    ])

def test_no_empty_logs():
    """Test that empty or whitespace-only logs are not stored."""
    interceptor = LogInterceptor()
    with interceptor.capture():
        print("")
        print("   ")
        print("\n")
        print("Real message")
    
    logs = interceptor.get_logs()
    assert len(logs) == 1
    assert "Real message" in logs[0]
