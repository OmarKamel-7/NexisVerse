# main.py
import os
import sys
import subprocess

from menu import run_menu
from apps.math_app import run_math_app
from apps.virtual_lab import run_virtual_lab
from apps.study_session import run_study_session
from apps.music_app import run_music_app


# ---------------------------
# Utility Functionsq
# ---------------------------

def launch_notebook():
    """Open the notebook file with the system's default editor."""
    notebook_path = os.path.abspath("data/notebook.txt")
    if sys.platform.startswith("win"):
        os.startfile(notebook_path)
    elif sys.platform.startswith("darwin"):  # macOS
        subprocess.Popen(["open", notebook_path])
    else:  # Linux / others
        subprocess.Popen(["xdg-open", notebook_path])


# ---------------------------
# App Dispatcher
# ---------------------------

def dispatch_app(choice: str):
    """Map menu return values to actual app functions."""
    apps = {
        "Math App": run_math_app,
        "Virtual Lab": run_virtual_lab,
        "Study Session": run_study_session,
        "Music": run_music_app,
        "Notebook": launch_notebook,
    }

    action = apps.get(choice)
    if action:
        print(f"🚀 Launching {choice}...")
        action()
        print("↩️ Returned to NexisVerse Menu.")
    else:
        print(f"⚠️ Unknown selection: {choice}")


# ---------------------------
# Main Loop
# ---------------------------

def main():
    print("\n👁️ Welcome to NexisVerse - Vision Pro Style AR Menu")
    print("✨ Use your hands, gestures, or arrows to navigate.\n")

    while True:
        choice = run_menu()  # Waits until user selects or exits

        if choice in (None, "Exit"):
            print("\n👋 Exiting NexisVerse. Goodbye!")
            break

        dispatch_app(choice)


if __name__ == "__main__":
    main()
