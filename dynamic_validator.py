import os
import logging
import pathlib
import re
import json
from typing import Dict, List, Any, Tuple

import pandas as pd
import google.generativeai as genai
from pyspark.sql import SparkSession, DataFrame
from pyspark.sql import functions as F
from pyspark.sql.utils import ParseException

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class DynamicValidator:
    """
    A generic utility to validate a PySpark DataFrame based on rules defined in a CSV.
    It uses a generative AI model to dynamically create validation logic from natural language.
    """

    def __init__(self, api_key: str, spark: SparkSession = None, model_name: str = "gemini-1.5-flash-latest", cache_path: str = None):
        """
        Initializes the Validator.

        Args:
            api_key (str): The Google Generative AI API key.
            spark (SparkSession, optional): The active SparkSession. Required for `validate` method. Defaults to None.
            model_name (str): The name of the generative AI model to use.
            cache_path (str, optional): Path to a file for persistent expression caching. Defaults to None.
        """
        self.spark = spark
        self.model = None
        self.cache_path = cache_path

        if api_key:
            genai.configure(api_key=api_key)
            self.model = genai.GenerativeModel(model_name)
            logging.info(f"Successfully configured generative model '{model_name}'.")
        else:
            logging.warning("No API key provided to DynamicValidator. AI features are disabled.")

        self._expression_cache: Dict[str, str] = self._load_cache()

    def _load_cache(self) -> Dict[str, str]:
        """Loads the expression cache from a file if cache_path is set."""
        if self.cache_path and os.path.exists(self.cache_path):
            try:
                with open(self.cache_path, 'r', encoding='utf-8') as f:
                    cache = json.load(f)
                    logging.info(f"Successfully loaded expression cache from {self.cache_path}.")
                    return cache
            except (IOError, json.JSONDecodeError) as e:
                logging.warning(f"Could not load cache from {self.cache_path}. Starting with an empty cache. Error: {e}")
        return {}

    def _save_cache(self):
        """Saves the expression cache to a file if cache_path is set."""
        if self.cache_path:
            try:
                # Ensure parent directory exists
                pathlib.Path(self.cache_path).parent.mkdir(parents=True, exist_ok=True)
                with open(self.cache_path, 'w', encoding='utf-8') as f:
                    json.dump(self._expression_cache, f, indent=4)
                    logging.info(f"Successfully saved expression cache to {self.cache_path}.")
            except IOError as e:
                logging.error(f"Could not save cache to {self.cache_path}. Error: {e}")

    def _get_prompt(self, column_name: str, data_type: str, description: str) -> str:
        """Creates a detailed, few-shot prompt for reliable PySpark expression generation."""
        # This detailed prompt structure significantly improves the reliability of the AI's output.
        return f"""
You are a world-class PySpark expert specializing in data validation. Your task is to convert a natural language validation rule into a single, clean PySpark SQL expression.

The expression must evaluate to `true` if the data is valid and `false` if it is invalid.

**Context:**
- Column Name: `{column_name}`
- Column Data Type: `{data_type}`
- Validation Rule: `{description}`

**Instructions:**
1.  The output must be ONLY the PySpark SQL expression. Do not include any explanation or surrounding text.
2.  The expression must handle NULL values appropriately for the business rule. Unless the rule is a "not null" check, the expression should evaluate to `true` for a NULL value.
3.  Use backticks around the column name, e.g., `{column_name}`.
4.  **IMPORTANT**: An automatic data type check will be performed separately to ensure the column can be cast to `{data_type}`. You can assume that for any non-null value, `TRY_CAST(`{column_name}` AS {data_type})` will succeed. Your expression should only implement the specific business logic.

**Example 1:**
- Column Name: `age`
- Column Data Type: `int`
- Validation Rule: `value must be greater than 18`
- Your Output: (`age` IS NULL) OR (`age` > 18)

**Example 2:**
- Column Name: `email`
- Column Data Type: `string`
- Validation Rule: `Should be a valid email address`
- Your Output: (`email` IS NULL) OR (`email` RLIKE '^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\\.[a-zA-Z]{{2,6}}$')

**Example 3:**
- Column Name: `name`
- Column Data Type: `string`
- Validation Rule: `Not Null`
- Your Output: `name` IS NOT NULL

**Example 4:**
- Column Name: `status`
- Column Data Type: `string`
- Validation Rule: `Should be one of ['ACTIVE', 'INACTIVE']`
- Your Output: (`status` IS NULL) OR (`status` IN ('ACTIVE', 'INACTIVE'))

**Example 5 (Numeric Check on String):**
- Column Name: `zip_code`
- Column Data Type: `int`
- Validation Rule: `Must be a number`
- Your Output: true

**Example 6 (Compound Rule):**
- Column Name: `id`
- Column Data Type: `int`
- Validation Rule: `Should be a number and not null`
- Your Output: `id` IS NOT NULL

**Your Task:**
- Column Name: `{column_name}`
- Column Data Type: `{data_type}`
- Validation Rule: `{description}`
- Your Output:
"""

    def _generate_spark_expression(self, column_name: str, data_type: str, description: str) -> str:
        """
        Generates a PySpark SQL expression from a natural language rule using an AI model.
        Uses a cache to avoid redundant API calls.
        """
        if not self.model:
            raise ConnectionError("Generative model is not configured. Please set GEMINI_API_KEY.")

        cache_key = f"{column_name}|{data_type}|{description}"
        if cache_key in self._expression_cache:
            logging.info(f"Cache hit for rule: '{description}' on column '{column_name}'.")
            return self._expression_cache[cache_key]

        logging.info(f"Cache miss. Generating expression for rule: '{description}' on column '{column_name}'.")
        prompt = self._get_prompt(column_name, data_type, description)
        
        try:
            response = self.model.generate_content(prompt)
            # Add a robust check to see if the model returned any content.
            # An empty list of candidates usually means the response was blocked.
            if not response.candidates:
                feedback = response.prompt_feedback
                block_reason = getattr(feedback, 'block_reason', 'Unknown')
                error_message = f"Generation blocked by API. Reason: {block_reason}."
                logging.warning(f"Generation failed for rule '{description}': {error_message}")
                # Raise a specific error to be caught by the calling function.
                raise ValueError(error_message)

            expression = response.text.strip()
        except Exception as e:
            # Catch the ValueError from above or any other API error.
            logging.error(f"Error calling Generative AI API for rule '{description}': {e}")
            # Re-raise the exception to be handled gracefully by the notebook generator.
            raise e

        self._expression_cache[cache_key] = expression
        self._save_cache()
        return expression

    def validate(self, df: DataFrame, rules_path: str) -> DataFrame:
        """
        Applies validation rules from a CSV file to a DataFrame.

        Args:
            df (DataFrame): The input DataFrame to validate.
            rules_path (str): The path to the CSV file containing validation rules.

        Returns:
            DataFrame: A DataFrame containing only the invalid rows, with an added
                       'validation_errors' column detailing the failures.
        """
        if not self.spark:
            raise ValueError("A SparkSession must be provided to the DynamicValidator to use the 'validate' method.")

        logging.info(f"Reading validation rules from {rules_path}...")
        rules_df = self.spark.read.csv(rules_path, header=True, inferSchema=True)
        rules: List[Dict[str, Any]] = [row.asDict() for row in rules_df.collect()]

        validated_df = df
        error_conditions = []

        for i, rule in enumerate(rules):
            col_name = rule['column_name']
            validation_col_name = f"__is_valid_{i}"

            try:
                spark_expr = self._generate_spark_expression(
                    column_name=col_name,
                    data_type=rule['data_type'],
                    description=rule['rule']
                )
                logging.info(f"Applying rule for '{col_name}': {rule['rule']} -> {spark_expr}")
                validated_df = validated_df.withColumn(validation_col_name, F.expr(spark_expr))
                
                # Add the failure condition to our list
                error_message = F.lit(f"FAILED RULE: For column '{col_name}', rule was: '{rule['rule']}'")
                error_conditions.append(F.when(F.col(validation_col_name) == False, error_message))

            except ParseException as e:
                logging.error(f"Failed to parse generated expression for rule: {rule}. Error: {e}")
                # Optionally, you could add a generic error message for this failure
            except Exception as e:
                logging.error(f"An unexpected error occurred while processing rule: {rule}. Error: {e}")

        # Aggregate all error messages into a single array column
        validated_df = validated_df.withColumn("validation_errors", F.array_remove(F.array(*error_conditions), None))

        # Filter for rows that have at least one validation error
        invalid_df = validated_df.filter(F.size(F.col("validation_errors")) > 0)

        # Clean up temporary validation columns
        temp_cols = [f"__is_valid_{i}" for i in range(len(rules))]
        return invalid_df.drop(*temp_cols)

    @staticmethod
    def _generate_sample_data_code(rules: List[Dict[str, Any]]) -> (str, str):
        """Generates python code for sample columns and data based on rule definitions."""
        schema_from_rules = {}
        for rule in rules:
            col_name = rule.get('column_name')
            data_type = rule.get('data_type')
            if col_name and col_name not in schema_from_rules:
                schema_from_rules[col_name] = data_type
        
        if not schema_from_rules:
            return "[]", "[]"

        notebook_columns = list(schema_from_rules.keys())
        notebook_columns_str = str(notebook_columns)

        sample_data_str_rows = []
        for i in range(4): # 4 semi-valid rows
            row_values = []
            for col in notebook_columns:
                dtype = schema_from_rules[col].lower()
                val = "None" # Default
                if dtype in ('int', 'bigint'):
                    val = (i + 1) * 10
                elif dtype == 'double':
                    val = (i + 1) * 1.1
                elif dtype == 'string':
                    if 'email' in col.lower():
                        # Make one email invalid for testing
                        val = f"'user{i}@example.com'" if i > 0 else "'invalid-email'"
                    elif 'status' in col.lower():
                        val = "'ACTIVE'" if i % 2 == 0 else "'INACTIVE'"
                    elif 'name' in col.lower():
                        val = f"'FirstName{i}'"
                    else:
                        val = f"'value_{i}'"
                elif dtype == 'date':
                    val = f"datetime.date(2024, 1, {i+1})"
                elif dtype == 'timestamp':
                    val = f"datetime.datetime(2024, 1, {i+1}, 12, 0, 0)"
                elif dtype == 'boolean':
                    val = "True" if i % 2 == 0 else "False"
                row_values.append(str(val))
            sample_data_str_rows.append(f"    ({', '.join(row_values)})")
        
        # Add one row with all nulls to test null handling
        null_row = ["None"] * len(notebook_columns)
        sample_data_str_rows.append(f"    ({', '.join(null_row)})")

        sample_data_str = ",\n".join(sample_data_str_rows)
        
        return notebook_columns_str, sample_data_str

    def generate_databricks_notebook(self, rules_df: pd.DataFrame) -> Tuple[str, Dict[str, List[Dict[str, str]]]]:
        """
        Generates a standalone Databricks notebook string from a DataFrame of rules.

        Args:
            rules_df (pd.DataFrame): A DataFrame containing the validation rules.
                                     Must have 'column_name', 'data_type', and 'rule' columns.

        Returns:
            Tuple[str, Dict]: A tuple containing:
                - The generated notebook code as a string.
                - A dictionary with summaries for type checks and business rules.
        """
        logging.info(f"Generating Databricks notebook from {len(rules_df)} rules...")
        if rules_df.empty:
            return "", {}
        rules = rules_df.to_dict('records')

        # Generate sample data code dynamically
        sample_columns_str, sample_data_str = self._generate_sample_data_code(rules)

        # --- Prepare lists for different validation steps ---
        type_check_steps = []
        type_check_cols = []
        validation_steps = []
        validation_check_cols = []
        type_check_summaries = []
        business_rule_summaries = []

        for i, rule in enumerate(rules):
            col_name = rule['column_name']
            data_type = rule['data_type']

            # --- 1. Generate Type Check Step (once per column) ---
            # Sanitize column name for a valid python variable name
            sanitized_col_for_type = re.sub(r'[^a-zA-Z0-9_]', '_', col_name).lower()
            type_check_col_name = f"{sanitized_col_for_type}_type_ok"
            
            if type_check_col_name not in type_check_cols:
                type_check_expr = f"(`{col_name}` IS NULL) OR (TRY_CAST(`{col_name}` AS {data_type}) IS NOT NULL)"
                type_step_code = f"""
# Type check for column: '{col_name}' | Expected type: {data_type}
validated_df = validated_df.withColumn(
    "{type_check_col_name}",
    F.expr("{type_check_expr}")
)"""
                type_check_steps.append(type_step_code)
                type_check_cols.append(type_check_col_name)
                type_check_summaries.append({
                    "Column Name": col_name,
                    "Expected Data Type": data_type,
                    "Generated PySpark Expression": type_check_expr
                })

            # --- 2. Generate Business Rule Step ---
            # Sanitize column and rule description for a valid python variable name
            sanitized_col = re.sub(r'[^a-zA-Z0-9_]', '_', col_name).lower()
            sanitized_rule = re.sub(r'[^a-zA-Z0-9_]', '_', rule['rule']).lower()[:30]
            validation_col_name = f"{sanitized_col}_{sanitized_rule}_{i}_ok"

            generated_expression = "Error: An unknown error occurred."
            try:
                spark_expr = self._generate_spark_expression(
                    column_name=col_name,
                    data_type=data_type,
                    description=rule['rule']
                )
                generated_expression = spark_expr
                # Escape quotes for embedding in the script string
                escaped_spark_expr = spark_expr.replace('"', '\\"')
                
                step_code = f"""
# Rule for column: '{col_name}' | Description: {rule['rule']}
validated_df = validated_df.withColumn( # Using original rule for comment is fine
    "{validation_col_name}",
    F.expr("{escaped_spark_expr}")
)"""
                validation_steps.append(step_code)
                validation_check_cols.append(validation_col_name)

            except Exception as e:
                # Use the specific error message from the exception for better feedback.
                error_msg = str(e)
                logging.error(f"Could not generate expression for rule: {rule['rule']}. Error: {error_msg}")
                generated_expression = f"Error: {error_msg}"
                validation_steps.append(f"\n# FAILED TO GENERATE RULE for column '{col_name}': {rule['rule']}\n# Error: {error_msg}\n")
            
            business_rule_summaries.append({
                "Column Name": col_name,
                "Data Type": data_type,
                "Rule": rule['rule'],
                "Generated PySpark Expression": generated_expression
            })

        # --- 3. Combine and format all generated code ---
        all_check_cols = type_check_cols + validation_check_cols
        indented_type_check_steps = "\n    ".join(type_check_steps)
        indented_business_rule_steps = "\n    ".join(validation_steps)
        all_check_cols_str = str(all_check_cols)

        notebook_template = f"""# Databricks notebook source
# MAGIC %md
# MAGIC # Auto-Generated Data Quality Validation Notebook
# MAGIC
# MAGIC This notebook was generated by the DynamicValidator utility based on rules from:

# COMMAND ----------

from pyspark.sql import functions as F
from pyspark.sql.types import *
import datetime

# COMMAND ----------

# MAGIC %md
# MAGIC ### 1. Load Source Data
# MAGIC **IMPORTANT**: Replace the sample data logic below with your actual data loading logic.
# MAGIC For example: `source_df = spark.read.table("my_catalog.my_schema.my_table")`

# COMMAND ----------

# For demonstration, we create a sample DataFrame based on your rules.
# --> REPLACE THIS WITH YOUR DATA LOADING LOGIC <--
data = [
{sample_data_str}
]
columns = {sample_columns_str}
source_df = spark.createDataFrame(data, columns)
print("Source DataFrame (Sample):")
display(source_df)

# COMMAND ----------

# MAGIC %md
# MAGIC ### 2. Sanitize Column Names
# MAGIC This step cleans the column names of the source DataFrame by trimming leading/trailing whitespace. This prevents errors when rules refer to a column like `'name'` but the source file contained a header like `' name '`.

# COMMAND ----------

df_with_clean_cols = source_df
for col_name in df_with_clean_cols.columns:
    # Trim leading/trailing whitespace from the column name
    new_col_name = col_name.strip()
    if new_col_name != col_name:
        df_with_clean_cols = df_with_clean_cols.withColumnRenamed(col_name, new_col_name)

# Use the sanitized DataFrame for all subsequent steps
source_df = df_with_clean_cols

print("Schema after column name sanitization:")
source_df.printSchema()

# COMMAND ----------

# MAGIC %md
# MAGIC ### 3. Trim Whitespace from String Columns
# MAGIC This step cleans the data within string columns by trimming leading/trailing whitespace. This prevents validation failures for values like `' ACTIVE '` when the rule expects `'ACTIVE'`.

# COMMAND ----------

# Identify string columns to trim
string_columns = [field.name for field in source_df.schema.fields if isinstance(field.dataType, StringType)]

df_with_trimmed_data = source_df
for col_name in string_columns:
    df_with_trimmed_data = df_with_trimmed_data.withColumn(col_name, F.trim(F.col(col_name)))

# Use the trimmed DataFrame for all subsequent steps
source_df = df_with_trimmed_data

print("Data preview after trimming whitespace:")
display(source_df.limit(5))

# COMMAND ----------

# MAGIC %md
# MAGIC ### 4. Applying Data Type Validation
# MAGIC This step checks if the data in each column can be cast to its specified data type. This is a fundamental check for data integrity.
# MAGIC A boolean column (ending in `_type_ok`) is added for each column defined in the rules.
# MAGIC - `true`: The record passes the validation.
# MAGIC - `false`: The record fails the validation.

# COMMAND ----------

# Start of dynamically generated type validation checks
validated_df = source_df
# MAGIC {indented_type_check_steps}

# COMMAND ----------

# MAGIC %md
# MAGIC ### 5. Applying Business Logic Validation Rules
# MAGIC Each business rule from the CSV is applied, adding a new boolean column.
# MAGIC - `true`: The record passes the validation.
# MAGIC - `false`: The record fails the validation.

# COMMAND ----------

# Start of dynamically generated business rule checks
{indented_business_rule_steps}

# COMMAND ----------

# MAGIC %md
# MAGIC ### 6. Summarizing Validation Failures
# MAGIC This section calculates and displays a summary of how many records failed each specific rule.

# COMMAND ----------

all_check_cols = {all_check_cols_str}

if all_check_cols:
    total_records = validated_df.count()
    agg_expressions = [F.sum(F.when(F.col(c) == False, 1).otherwise(0)).alias(c) for c in all_check_cols]
    failure_counts_df = validated_df.agg(*agg_expressions).first()
    
    summary_data = []
    if total_records > 0 and failure_counts_df:
        for col_name in all_check_cols:
            failure_count = failure_counts_df[col_name]
            if failure_count > 0:
                summary_data.append({{"Validation Rule": col_name, "Failed Records": failure_count, "Failure Rate (%)": round((failure_count / total_records) * 100, 2)}})
    
    if summary_data:
        failure_summary_df = spark.createDataFrame(summary_data)
        print("Validation Failure Summary:")
        display(failure_summary_df)
    else:
        print("No validation failures found.")
else:
    print("No validation rules were applied or generated.")

# COMMAND ----------

# MAGIC %md
# MAGIC ### 7. Isolating and Storing Good and Bad Records
# MAGIC This final step filters the DataFrame to separate rows that passed all checks from those that failed at least one. The results are then written to separate Delta tables.

# COMMAND ----------

# --- Configuration for Output Paths ---
# IMPORTANT: Update these paths to your desired output locations.
valid_records_path = "/Volumes/data_files/testdata/data/valid/"
invalid_records_path = "/Volumes/data_files/testdata/data/invalid/"
# ------------------------------------

if all_check_cols:
    # A record is invalid if any check is explicitly false.
    failure_condition = " OR ".join(["(" + c + " = false)" for c in all_check_cols])
    
    all_original_cols = source_df.columns
    
    # Isolate good and invalid records
    invalid_rows_df = validated_df.filter(failure_condition)
    good_rows_df = validated_df.filter("NOT (" + failure_condition + ")")
    
    # --- Write Valid Records ---
    valid_count = good_rows_df.count()
    print(f"✅ Found {{valid_count}} valid records.")
    if valid_count > 0:
        print(f"Writing valid records to: {{valid_records_path}}")
        good_rows_df.select(*all_original_cols).write.format("delta").mode("overwrite").save(valid_records_path)
        print("Write complete.")
        display(good_rows_df.select(*all_original_cols).limit(10)) # Display a sample
    
    # --- Write Invalid Records ---
    invalid_count = invalid_rows_df.count()
    print(f"\\n❌ Found {{invalid_count}} records with at least one validation failure.")
    if invalid_count > 0:
        print(f"Writing invalid records to: {{invalid_records_path}}")
        # Save invalid rows with the boolean check columns for easy debugging
        invalid_rows_df.select(*all_original_cols, *all_check_cols).write.format("delta").mode("overwrite").save(invalid_records_path)
        print("Write complete.")
        display(invalid_rows_df.select(*all_original_cols, *all_check_cols).limit(10)) # Display a sample
else:
    print("No validation checks were performed. All records are considered good.")
    # Optionally, write all records to the 'valid' path if no rules were applied
    print(f"Writing all records to: {{valid_records_path}}")
    source_df.write.format("delta").mode("overwrite").save(valid_records_path)
    print("Write complete.")
    display(source_df.limit(10))
"""
        summaries = {
            "type_checks": type_check_summaries,
            "business_rules": business_rule_summaries
        }
        logging.info("Successfully generated Databricks notebook content in memory.")
        return notebook_template, summaries