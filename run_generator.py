import os
from pathlib import Path
from dotenv import load_dotenv
from dynamic_validator import DynamicValidator

def main():
    """
    This is a lightweight script to ONLY generate the validation notebook..
    It does NOT require PySpark to be installed or a SparkSession to be running.
    """
    print("--- Running Lightweight Notebook Generator ---")
    
    # Load API Key from .env file
    load_dotenv()
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        print("ERROR: GEMINI_API_KEY environment variable is not set.")
        print("Please create a .env file with your GEMINI_API_KEY.")
        return

    # Define relative paths for portability
    project_root = Path(__file__).parent
    rules_path = project_root / "validation_rules.csv"
    output_dir = project_root / "output"
    output_notebook_path = output_dir / "adb_datavalidation.py"
    cache_file_path = output_dir / "expression_cache.json"

    # Initialize the validator WITHOUT a SparkSession, but with a persistent cache path.
    # This is possible because the notebook generation method does not use Spark.
    validator = DynamicValidator(api_key=api_key, cache_path=str(cache_file_path))

    # Generate the notebook
    print(f"Reading rules from: {rules_path}")
    print(f"Generating notebook to: {output_notebook_path}")
    validator.generate_databricks_notebook(str(rules_path), str(output_notebook_path))
    
    print("\n--- Generation Complete ---")

if __name__ == "__main__":
    main()