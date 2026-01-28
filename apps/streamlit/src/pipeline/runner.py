"""
Command execution utilities for running external processes.
"""

import subprocess

def run_command(cmd: list, cwd: str = None, output_callback=None) -> bool:
    print(f"Executing: {' '.join(cmd)}")
    if output_callback:
        output_callback(f"Executing: {' '.join(cmd)}\n")
        
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
            if output_callback:
                output_callback(line)
                
        process.wait()
        
        if process.returncode != 0:
            err_msg = f"Command failed with exit code {process.returncode}"
            print(err_msg)
            if output_callback:
                output_callback(f"{err_msg}\n")
            return False
            
        return True
    except Exception as e:
        err_msg = f"Error executing command: {e}"
        print(err_msg)
        if output_callback:
            output_callback(f"{err_msg}\n")
        return False
