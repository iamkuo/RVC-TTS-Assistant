#!/usr/bin/env python3
"""
Debug wrapper for TTS_ai_nostream.py
This will catch and display the full error message that's being hidden
"""

import sys
import traceback
import os
import subprocess

def main():
    try:
        # Change to the script directory
        script_dir = os.path.dirname(os.path.abspath(__file__))
        os.chdir(script_dir)
        
        # Use the Conda environment Python
        conda_python = r"d:\program\RVC-TTS-PIPELINE\.conda\python.exe"
        
        print(f"Using Conda Python: {conda_python}")
        print(f"Working directory: {os.getcwd()}")
        
        # Check if the Conda Python exists
        if not os.path.exists(conda_python):
            print(f"ERROR: Conda Python not found at {conda_python}")
            return
        
        # Run the TTS script using the Conda environment
        print("Attempting to run TTS_ai_nostream.py with Conda environment...")
        print("-" * 50)
        
        result = subprocess.run([conda_python, 'TTS_ai_nostream.py'], 
                              cwd=script_dir, 
                              capture_output=False, 
                              text=True)
        
        if result.returncode != 0:
            print(f"Script exited with error code: {result.returncode}")
        
    except FileNotFoundError as e:
        print(f"ERROR: Could not find TTS_ai_nostream.py")
        print(f"Current directory: {os.getcwd()}")
        print(f"Files in directory: {os.listdir('.')}")
        print(f"Full error: {e}")
        
    except Exception as e:
        print("ERROR: An unexpected error occurred:")
        print(f"Error type: {type(e).__name__}")
        print(f"Error message: {str(e)}")
        print("Full traceback:")
        traceback.print_exc()
        
    input("\nPress Enter to exit...")

if __name__ == "__main__":
    main()