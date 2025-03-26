from contextlib import contextmanager
import sys
import os
import io


@contextmanager
def suppress_output():
    """Suppresses stdout and stderr for both Unix and Windows systems."""
    # Save the original high-level streams
    saved_stderr = sys.stderr
    saved_stdout = sys.stdout

    try:
        # Try to save the current file descriptors
        original_stderr_fd = sys.stderr.fileno()
        original_stdout_fd = sys.stdout.fileno()
        saved_stderr_fd = os.dup(original_stderr_fd)
        saved_stdout_fd = os.dup(original_stdout_fd)

        with open(os.devnull, "wb") as devnull:
            # Redirect the lower-level file descriptors
            os.dup2(devnull.fileno(), original_stderr_fd)
            os.dup2(devnull.fileno(), original_stdout_fd)

        # Redirect the higher-level Python streams
        sys.stderr = open(os.devnull, "w")
        sys.stdout = open(os.devnull, "w")

    except (io.UnsupportedOperation, AttributeError):
        # If fileno is not supported, just replace the Python streams
        sys.stderr = open(os.devnull, "w")
        sys.stdout = open(os.devnull, "w")

    try:
        yield
    finally:
        # Restore the high-level Python streams
        sys.stderr.close()
        sys.stdout.close()
        sys.stderr = saved_stderr
        sys.stdout = saved_stdout

        # Restore the original file descriptors if they were saved
        if "saved_stderr_fd" in locals() and "saved_stdout_fd" in locals():
            os.dup2(saved_stderr_fd, original_stderr_fd)
            os.dup2(saved_stdout_fd, original_stdout_fd)
            os.close(saved_stderr_fd)
            os.close(saved_stdout_fd)
