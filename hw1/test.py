#!/usr/bin/env python3
"""
Simple test runner interface for students.
Usage: python test.py [--part 1|2|3|all]
"""

import subprocess
import sys
import os

def main():
    """Run the test runner with command line arguments."""
    
    # Check if test_cases.json exists
    if not os.path.exists('test_cases.json'):
        print("Error: test_cases.json not found.")
        print("This file should be provided with the assignment.")
        sys.exit(1)
    
    # Check if test_runner.py exists
    if not os.path.exists('test_runner.py'):
        print("Error: test_runner.py not found.")
        print("This file should be provided with the assignment.")
        sys.exit(1)
    
    # Pass all command line arguments to test_runner.py
    cmd = [sys.executable, 'test_runner.py'] + sys.argv[1:]
    
    try:
        subprocess.run(cmd)
    except KeyboardInterrupt:
        print("\nTest interrupted by user.")
        sys.exit(1)

if __name__ == "__main__":
    main()