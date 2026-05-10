"""
Redirect file - points to the current active app
This file exists to maintain compatibility with legacy references.
"""
import subprocess
import sys

if __name__ == "__main__":
    # Redirect to the actual app
    print("⚠️  Redirecting to streamlit_app_new.py...")
    print("    Update your scripts to use 'streamlit_app_new.py' directly")
    subprocess.run([sys.executable, "-m", "streamlit", "run", "streamlit_app_new.py"] + sys.argv[1:])