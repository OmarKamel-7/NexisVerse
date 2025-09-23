# main.py
from menu import run_menu
from apps.math_app import run_math_app
from apps.virtual_lab import run_virtual_lab
from apps.study_session import run_study_session
from apps.music_app import run_music_app

def main():
    print("HIZMOS-Style AR Menu - PC build (gesture only)")
    while True:
        choice = run_menu()  # blocks until user selects something or Exit

        if choice is None or choice == "Exit":
            print("👋 Exiting system.")
            break

        if choice == "Math App":
            print("Launching Math App...")
            run_math_app()

        elif choice == "Virtual Lab":
            print("Launching Virtual Lab...")
            run_virtual_lab()

        elif choice == "Study Session":
            print("Launching Study Session...")
            run_study_session()

        elif choice == "Music":
            # we call run_music_app without a link — it will prompt for your link
            run_music_app()

        elif choice == "Notebook":
            # Notebook opens external text file editor then returns
            import os, subprocess, sys
            notebook_path = "data/notebook.txt"
            if sys.platform.startswith("win"):
                os.startfile(os.path.abspath(notebook_path))
            elif sys.platform.startswith("darwin"):
                subprocess.Popen(["open", notebook_path])
            else:
                subprocess.Popen(["xdg-open", notebook_path])

        else:
            print("Unknown choice:", choice)

if __name__ == "__main__":
    main()
