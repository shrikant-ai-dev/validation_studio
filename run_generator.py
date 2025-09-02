import os
from pathlib import Path
from dotenv import load_dotenv
from dynamic_validator import DynamicValidator
import pandas as pd

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

    # Ensure the output directory exists
    output_dir.mkdir(parents=True, exist_ok=True)

    # Read rules into a pandas DataFrame
    try:
        rules_df = pd.read_csv(rules_path)
    except FileNotFoundError:
        print(f"ERROR: Rules file not found at {rules_path}")
        return

    # Initialize the validator WITHOUT a SparkSession, but with a persistent cache path.
    # This is possible because the notebook generation method does not use Spark.
    validator = DynamicValidator(api_key=api_key, cache_path=str(cache_file_path))

    # Generate the notebook
    print(f"Reading rules from: {rules_path}")
    notebook_content, _ = validator.generate_databricks_notebook(rules_df)
    output_notebook_path.write_text(notebook_content, encoding='utf-8')
    
    print(f"\n--- Generation Complete --- \nNotebook saved to: {output_notebook_path}")

if __name__ == "__main__":
    main()