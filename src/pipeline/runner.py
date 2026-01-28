"""
Command execution utilities for running external processes.
"""

import subprocess


def run_command(cmd: list, cwd: str = None) -> bool:
    """
    Execute a command with real-time output streaming.
    
    Args:
        cmd: Command and arguments as a list
        cwd: Working directory for the command
        
    Returns:
        True if command succeeded, False otherwise
    """
    print(f"Executing: {' '.join(cmd)}")
    try:
        process = subprocess.Popen(
            cmd,
            cwd=cwd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            universal_newlines=True
        )
        for line in process.stdout:
            print(line, end='')
        process.wait()
        if process.returncode != 0:
            print(f"Command failed with exit code {process.returncode}")
            return False
        return True
    except Exception as e:
        print(f"Error executing command: {e}")
        return False
