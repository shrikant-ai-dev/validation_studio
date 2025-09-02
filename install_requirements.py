import subprocess
import sys
import os

def install_dependencies(requirements_path):
    """
    Installs Python packages from a given requirements.txt file.

    Args:
        requirements_path (str): The absolute path to the requirements.txt file.
    """
    if not os.path.exists(requirements_path):
        print(f"Error: Requirements file not found at '{requirements_path}'")
        sys.exit(1)

    print(f"Installing packages from '{os.path.basename(requirements_path)}'...")

    try:
        # Use sys.executable to ensure we use the pip for the current Python env
        command = [sys.executable, "-m", "pip", "install", "-r", requirements_path]
        subprocess.check_call(command)
        print("\nSuccessfully installed or updated all required packages.")
    except subprocess.CalledProcessError as e:
        print(f"\nAn error occurred while installing packages: {e}")
        print("Please check the package names in your requirements file and your network connection.")
        sys.exit(1)

if __name__ == "__main__":
    # Assume requirements.txt is in the same directory as this script.
    current_dir = os.path.dirname(os.path.abspath(__file__))
    req_file = os.path.join(current_dir, "requirements.txt")
    install_dependencies(req_file)