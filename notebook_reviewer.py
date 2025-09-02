import os
import logging
import google.generativeai as genai

class NotebookReviewer:
    """
    Uses a generative AI model to review a Databricks notebook against a set of
    pre-defined development standards.
    """
    def __init__(self, api_key: str, model_name: str = "gemini-1.5-flash-latest"):
        """
        Initializes the NotebookReviewer.

        Args:
            api_key (str): The Google Generative AI API key.
            model_name (str): The name of the generative AI model to use.
        """
        self.model = None
        if api_key:
            genai.configure(api_key=api_key)
            self.model = genai.GenerativeModel(model_name)
            logging.info(f"Successfully configured NotebookReviewer with model '{model_name}'.")
        else:
            logging.warning("No API key provided to NotebookReviewer. Reviewer is disabled.")

    def _get_review_prompt(self, notebook_code: str) -> str:
        return f"""
You are an expert Databricks code reviewer. Your task is to analyze the following Python notebook code and evaluate it against a set of development standards.

**Development Standards to Check:**

1.  **No Hardcoded Secrets:** The code must not contain hardcoded secrets like passwords or access tokens. It should use `dbutils.secrets.get()` instead.
2.  **Proper Parameterization:** The notebook should use `dbutils.widgets.get("parameter_name")` to fetch input parameters rather than hardcoding values like file paths.
3.  **Sufficient Documentation:** The notebook must contain explanatory markdown cells (`# MAGIC %md`) and inline code comments (`#`) to explain complex logic.
4.  **Performance Best Practices:** The code should avoid calling actions like `.collect()` or `display()` on a full DataFrame. If used for debugging, they should be on a limited subset (e.g., using `.limit(n)`).
5.  **Robust Error Handling:** Critical operations (like data loading or writing) should be enclosed in `try...except` blocks to handle potential failures gracefully.

**Your Task:**

Review the notebook code provided below. For each development standard, provide a "PASS" or "FAIL" rating and a brief, clear explanation for your rating. Use Markdown for formatting, and include code snippets where relevant to support your findings.

**Notebook Code to Review:**
```python
{notebook_code}
```

**Your Output (in Markdown format):**
"""

    def review_notebook(self, notebook_code: str) -> str:
        if not self.model:
            raise ConnectionError("Generative model is not configured. Please set GEMINI_API_KEY.")
        
        prompt = self._get_review_prompt(notebook_code)
        response = self.model.generate_content(prompt)
        return response.text.strip()