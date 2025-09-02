import streamlit as st
import pandas as pd
import os
from pathlib import Path
import base64
from dynamic_validator import DynamicValidator
from notebook_reviewer import NotebookReviewer

# --- Page Configuration ---
st.set_page_config(
    page_title="Data Validation Studio",
    page_icon="🛠️",
    layout="wide"
)

# --- Helper Functions ---

def add_sidebar_logo():
    """Adds a Cognizant logo to the sidebar using a self-contained SVG."""
    logo_svg = """
    <svg fill="#0033A0" height="128px" width="128px" version="1.1" id="Layer_1" xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink" viewBox="0 0 300.00 300.00" xml:space="preserve">
        <path d="M149.793,0C67.062,0,0,67.062,0,149.793c0,82.732,67.062,150.207,149.793,150.207s150.207-67.475,150.207-150.207 C300,67.062,232.525,0,149.793,0z M215.333,167.131c0,36.781-29.887,66.668-66.668,66.668s-66.668-29.887-66.668-66.668 s29.887-66.668,66.668-66.668h36.052c16.213,0,29.49,13.277,29.49,29.49v1.126C214.207,147.293,215.333,156.885,215.333,167.131z"></path>
        <path d="M148.667,132.771c-19.125,0-34.625,15.5-34.625,34.625s15.5,34.625,34.625,34.625s34.625-15.5,34.625-34.625h-34.625V132.771 z"></path>
    </svg>
    """
    b64_svg = base64.b64encode(logo_svg.encode('utf-8')).decode('utf-8')
    st.sidebar.markdown(
        f'<div style="text-align: center;"><img src="data:image/svg+xml;base64,{b64_svg}" alt="Cognizant Logo" style="width:100px; margin-bottom: 20px;"></div>',
        unsafe_allow_html=True,
    )

def set_page_background():
    """
    Adds a subtle, professional background pattern to the page.
    The SVG is encoded in base64 to be self-contained.
    """
    svg_background = """
    <svg xmlns="http://www.w3.org/2000/svg" width="100%" height="100%">
        <defs>
            <pattern id="p" width="100" height="100" patternUnits="userSpaceOnUse" patternTransform="rotate(45)">
                <path id="a" data-color="outline" fill="none" stroke="#dce4f2" stroke-width="1.5" d="M0 0l100 100M100 0L0 100"></path>
            </pattern>
        </defs>
        <rect fill="white" width="100%" height="100%"></rect>
        <rect fill="url(#p)" width="100%" height="100%"></rect>
    </svg>
    """
    encoded_svg = base64.b64encode(svg_background.encode('utf-8')).decode('utf-8')
    page_bg_img = f"""
    <style>
    [data-testid="stAppViewContainer"] > .main {{
        background-image: url("data:image/svg+xml;base64,{encoded_svg}");        
    }}
    [data-testid="stHeader"] {{
        background-color: rgba(0,0,0,0);
    }}
    /* Clean tab style */
    .stTabs [data-baseweb="tab-list"] {{
		gap: 8px;
	}}
    .stTabs [data-baseweb="tab"] {{
        background-color: transparent;
		border-radius: 0.5rem;
    }}
    </style>
    """
    st.markdown(page_bg_img, unsafe_allow_html=True)

def validate_rules_df(df):
    """Checks if the uploaded DataFrame has the required columns."""
    required_columns = {'column_name', 'data_type', 'rule'}
    if not required_columns.issubset(df.columns):
        st.error(f"Invalid CSV format. Please ensure the file contains the following columns: {', '.join(required_columns)}")
        return False
    return True

def process_uploaded_file():
    """Callback function to process the uploaded file.
    This is triggered when a file is uploaded via st.file_uploader.
    """
    uploaded_file = st.session_state.get("file_uploader")
    if uploaded_file is None:
        return
    try:
        df = pd.read_csv(uploaded_file)
        if validate_rules_df(df):
            st.session_state.rules_df = df
            st.session_state.generated_notebook = None # Clear old notebook
    except Exception as e:
        st.error(f"Error reading or parsing the uploaded CSV file: {e}")

def profile_and_suggest_rules(sample_df: pd.DataFrame):
    """
    Profiles a sample DataFrame and generates a list of suggested validation rules. This enhanced
    version checks for nulls, uniqueness, low cardinality (for categories), and email formats.
    """
    suggested_rules = []

    # Map pandas dtypes to Spark SQL types for suggestions
    dtype_map = {
        'int64': 'int',
        'float64': 'double',
        'object': 'string',
        'bool': 'boolean',
        'datetime64[ns]': 'timestamp',
    }

    email_regex = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,6}$'

    for col in sample_df.columns:
        col_name = col.strip()
        data_type = dtype_map.get(str(sample_df[col].dtype), 'string')
        series = sample_df[col].dropna()  # Use a series with no nulls for most checks

        # Suggestion 1: Not Null check
        if sample_df[col].isnull().sum() == 0:
            suggested_rules.append({'column_name': col_name, 'data_type': data_type, 'rule': 'Not Null'})

        # Suggestion 2: Uniqueness check (for potential primary keys)
        if len(series) > 0 and series.is_unique:
            suggested_rules.append({'column_name': col_name, 'data_type': data_type, 'rule': 'Must be unique'})

        # Suggestion 3: Check for low cardinality strings (for IN list)
        if data_type == 'string':
            unique_vals = series.unique()
            if 1 < len(unique_vals) <= 10:  # Good candidate for an IN list
                # Escape single quotes in values before creating the list string
                safe_unique_vals = [str(v).replace("'", "\\'") for v in unique_vals]
                rule_text = f"Should be one of {safe_unique_vals}"
                suggested_rules.append({'column_name': col_name, 'data_type': data_type, 'rule': rule_text})

            # Suggestion 4: Email format check
            if len(series) > 0:
                email_matches = series.astype(str).str.match(email_regex).sum()
                if email_matches / len(series) > 0.8:  # If >80% look like emails
                    suggested_rules.append({'column_name': col_name, 'data_type': data_type, 'rule': 'Should be a valid email address'})

    if suggested_rules:
        # Remove potential duplicate rules for the same column and reset index
        final_rules_df = pd.DataFrame(suggested_rules).drop_duplicates().reset_index(drop=True)
        st.session_state.rules_df = final_rules_df
        st.success(f"✅ Suggested {len(final_rules_df)} rules based on your data sample!")
    else:
        st.warning("Could not automatically suggest any rules for the provided data sample.")

def on_generator_mode_change():
    """
    Callback to handle mode switching for the generator tool.
    """
    st.session_state.generated_notebook = None
    st.session_state.generation_stats = None
    # If switching to 'Create from scratch', reset the DataFrame for a clean slate.
    if st.session_state.mode == "Create rules from scratch":
        st.session_state.rules_df = pd.DataFrame(columns=['column_name', 'data_type', 'rule']) 

def show_screenshot(path, caption):
    """Safely displays a screenshot if it exists, otherwise shows a placeholder."""
    # Assumes images are in an 'assets' subdirectory
    full_path = Path(__file__).parent / "assets" / path
    if full_path.exists():
        st.image(str(full_path), caption=caption)
    else:
        st.warning(f"Screenshot not found: `assets/{path}`. Please add this image to your `assets` directory.", icon="🖼️")

def render_validation_generator():
    """Renders the UI for the Data Validation Generator tool."""
    # --- Sidebar Controls for Generator ---
    with st.sidebar:
        st.header("Generator Controls")
        mode = st.radio(
            "Choose how to provide rules:",
            ("Suggest rules from a data sample", "Upload a CSV file", "Create rules from scratch"),
            key='mode',
            on_change=on_generator_mode_change
        )

        if st.session_state.mode == "Upload a CSV file":
            st.file_uploader(
                "Upload Validation Rules CSV",
                type=['csv'],
                help="The CSV must contain 'column_name', 'data_type', and 'rule' columns.",
                key="file_uploader",
                on_change=process_uploaded_file
            )

        elif st.session_state.mode == "Suggest rules from a data sample":
            sample_file = st.file_uploader(
                "Upload a sample of your data (CSV)",
                type=['csv'],
                help="The tool will analyze this data to suggest validation rules."
            )
            if sample_file:
                if st.button("🔬 Profile Data & Suggest Rules", use_container_width=True, type="primary"):
                    with st.spinner("Analyzing data and suggesting rules..."):
                        sample_df = pd.read_csv(sample_file)
                        profile_and_suggest_rules(sample_df)

    # --- Main Area with Tabs ---
    generator_tab, help_tab = st.tabs(["🚀 Generator", "❓ Help & Instructions"])

    with generator_tab:
        # --- Main Area for Rule Editing ---
        st.header("✍️ Validation Rules Editor")

        if st.session_state.rules_df.empty:
            if st.session_state.mode == "Upload a CSV file":
                st.info("⬆️ Upload a rules file using the sidebar to get started.")
            elif st.session_state.mode == "Suggest rules from a data sample":
                st.info("⬆️ Upload a sample of your data and click the 'Profile' button in the sidebar.")
            else:
                st.info("✍️ Add your first rule below to get started.")
        else:
            st.success("Your rules are loaded in the editor. You can now modify them directly. Click 'Generate Notebook' when you're ready!")

        # IMPORTANT: The data_editor must be rendered before the buttons that depend on its state.
        # Use st.data_editor to allow for interactive editing, adding, and deleting of rules
        edited_df = st.data_editor(
            st.session_state.rules_df,
            num_rows="dynamic",  # Allows adding/deleting rows
            use_container_width=True,
            on_change=lambda: st.session_state.update(generated_notebook=None), # Clear notebook on edit
            column_config={
                "column_name": st.column_config.TextColumn("Column Name", help="The name of the column to validate.", required=True),
                "data_type": st.column_config.SelectboxColumn("Data Type", help="The Spark data type of the column.", options=["string", "int", "bigint", "double", "date", "timestamp"], required=True),
                "rule": st.column_config.TextColumn("Validation Rule", help="The validation rule in natural language.", width="large", required=True),
            }
        )
        # Immediately update the session state with the edited data
        st.session_state.rules_df = edited_df

        # --- Action Buttons ---
        st.divider()

        col1, col2 = st.columns([2, 1])

        with col1:
            # --- Generate Button ---
            if st.button("🚀 Generate Notebook", use_container_width=True, type="primary", disabled=st.session_state.rules_df.empty):
                with st.spinner("Generating notebook... This may take a moment. 🧠"):
                    # Define paths
                    project_root = Path(__file__).parent
                    output_dir = project_root / "output"
                    # Use a more robust temporary file name
                    temp_rules_path = output_dir / "temp_rules_for_generation.csv"
                    output_notebook_path = output_dir / "adb_datavalidation.py"
                    cache_file_path = output_dir / "expression_cache.json"

                    # Save the current state of rules to a temporary file
                    output_dir.mkdir(parents=True, exist_ok=True)
                    st.session_state.rules_df.to_csv(temp_rules_path, index=False)

                    # Initialize validator and generate
                    try:
                        # Pass the API key from Streamlit secrets to the validator
                        api_key = st.secrets.get("GEMINI_API_KEY")
                        if not api_key:
                            st.error("GEMINI_API_KEY not found in secrets. Please add it.")
                            st.stop()
                        validator = DynamicValidator(api_key=api_key, cache_path=str(cache_file_path))
                        summaries = validator.generate_databricks_notebook(
                            rules_path=str(temp_rules_path),
                            output_notebook_path=str(output_notebook_path)
                        )
                        
                        # Read the generated notebook content to store in session state for download
                        with open(output_notebook_path, 'r', encoding='utf-8') as f:
                            st.session_state.generated_notebook = f.read()

                        # Store stats for the dashboard
                        st.session_state.generation_stats = {
                            "rule_count": len(st.session_state.rules_df),
                            "type_distribution": st.session_state.rules_df['data_type'].value_counts(),
                            "summaries": summaries
                        }

                    except Exception as e:
                        st.error(f"An error occurred during generation: {e}")
                        st.session_state.generated_notebook = None # Clear on failure
                        st.session_state.generation_stats = None
                    
                    # Clean up temporary rules file
                    if os.path.exists(temp_rules_path):
                        os.remove(temp_rules_path)

        with col2:
            # --- Download Buttons ---
            if not st.session_state.rules_df.empty:
                st.download_button(
                   label="💾 Download Rules as CSV",
                   data=st.session_state.rules_df.to_csv(index=False).encode('utf-8'),
                   file_name='validation_rules.csv',
                   mime='text/csv',
                   use_container_width=True
                )
            if st.session_state.generated_notebook:
                st.download_button(
                    label="📥 Download Generated Notebook",
                    data=st.session_state.generated_notebook,
                    file_name="adb_datavalidation.py",
                    mime="text/x-python",
                    use_container_width=True
                )

        # --- Generated Code Preview ---
        if st.session_state.generated_notebook:
            st.header("✨ Generation Results")

            stats = st.session_state.get("generation_stats", {})
            summaries = stats.get("summaries", {})

            tab1, tab2, tab3 = st.tabs(["📊 **Dashboard**", "📝 **Code Review**", "📄 **Full Notebook**"])

            with tab1:
                st.subheader("Generation Dashboard")
                if stats:
                    with st.container(border=True):
                        st.metric(label="Total Business Rules Processed", value=stats.get("rule_count", 0))
                        st.success("✅ Notebook generated successfully!")
                else:
                    st.info("Generate a notebook to see the dashboard.")

            with tab2:
                st.subheader("Generated Code Review")
                type_check_summaries = summaries.get("type_checks")
                business_rule_summaries = summaries.get("business_rules")

                if type_check_summaries:
                    st.markdown("##### ⚙️ Data Type Validation")
                    st.info("These expressions are auto-generated to ensure data conforms to the specified type.", icon="ℹ️")
                    st.dataframe(pd.DataFrame(type_check_summaries), use_container_width=True, hide_index=True)

                if business_rule_summaries:
                    st.markdown("##### 🧠 Business Logic Validation")
                    st.info("Review the AI-generated expressions for your custom business rules.", icon="ℹ️")
                    st.dataframe(pd.DataFrame(business_rule_summaries), use_container_width=True, hide_index=True)

            with tab3:
                st.subheader("Complete Notebook Code")
                st.info("This is a preview of the full notebook. Use the copy button in the top-right corner of the code block below.", icon="📋")
                st.code(st.session_state.generated_notebook, language='python', line_numbers=True)

    with help_tab:
        st.header("Step-by-Step Guide to Using the Tool")

        st.markdown("""
        Welcome! This guide will walk you through the process of generating a Databricks data quality notebook.
        """)

        with st.expander("Step 1: Set Your API Key", expanded=True):
            st.markdown("""
            Before you begin, ensure you have a `.env` file in the root directory of this project. This file must contain your `GEMINI_API_KEY`.
            
            **Example `.env` file:**
            ```
            GEMINI_API_KEY="your_api_key_here"
            ```
            The application will show an error message at the top if the key is not found.
            """)

        with st.expander("Step 2: Choose How to Provide Rules", expanded=True):
            st.markdown("""
            Use the **Controls** in the sidebar to select one of three methods for providing validation rules.
            """)
            show_screenshot("screenshot_modes.png", "Select one of the three modes in the sidebar.")
            
            st.markdown("---")
            st.markdown("##### Method A: Suggest Rules from a Data Sample (Recommended)")
            st.markdown("""
            1.  Select "Suggest rules from a data sample".
            2.  Upload a CSV file containing a sample of your actual data.
            3.  Click the **🔬 Profile Data & Suggest Rules** button.
            4.  The tool will analyze your data and automatically populate the Rules Editor with suggested data quality checks.
            """)
            show_screenshot("screenshot_profiler.png", "Upload a data sample and click the 'Profile' button.")

            st.markdown("---")
            st.markdown("##### Method B: Upload a Rules CSV")
            st.markdown("""
            1.  Select "Upload a CSV file".
            2.  Upload a CSV that contains your predefined rules.
            3.  The CSV **must** have three columns: `column_name`, `data_type`, and `rule`.
            """)

            st.markdown("---")
            st.markdown("##### Method C: Create Rules from Scratch")
            st.markdown("""
            1.  Select "Create rules from scratch".
            2.  The Rules Editor will be empty, allowing you to add new rules manually by clicking the `+` icon at the bottom of the table.
            """)

        with st.expander("Step 3: Review and Edit Rules", expanded=True):
            st.markdown("""
            Regardless of the method chosen, your rules will appear in the **Validation Rules Editor**. You can add, delete, or modify any rule directly in this table.
            """)
            show_screenshot("screenshot_editor.png", "The main editor where you can add, remove, or modify rules.")

        with st.expander("Step 4: Generate and Download", expanded=True):
            st.markdown("""
            1.  Once you are satisfied with your rules, click the **🚀 Generate Notebook** button.
            2.  After generation, you can download the rules CSV or the final Python notebook using the download buttons.
            """)
            show_screenshot("screenshot_generate.png", "The 'Generate' and 'Download' buttons are located below the editor.")

        with st.expander("Step 5: Review the Results", expanded=True):
            st.markdown("""
            After a successful generation, a results section will appear at the bottom of the page with three tabs:
            - **Dashboard**: Shows high-level statistics about the generation.
            - **Code Review**: Provides a table of your natural language rules and a corresponding PySpark expressions generated by the AI. This is useful for verification.
            - **Full Notebook**: A preview of the complete, generated Databricks notebook.
            """)
            show_screenshot("screenshot_results.png", "The results section provides a dashboard, code review, and a full preview.")

def render_notebook_reviewer():
    """Renders the UI for the Notebook Reviewer tool."""
    st.header("Notebook Code Reviewer 🧐")
    st.markdown("Upload a Databricks notebook (`.py` file) to review it against development best practices.")

    with st.sidebar:
        st.header("Reviewer Controls")
        uploaded_notebook = st.file_uploader(
            "Upload your notebook script",
            type=['py'],
            key="notebook_uploader"
        )

    if uploaded_notebook:
        notebook_code = uploaded_notebook.read().decode("utf-8")
        
        if st.button("🕵️ Review Notebook", use_container_width=True, type="primary"):
            try:
                with st.spinner("AI is reviewing your code... 🤖"):
                    api_key = st.secrets.get("GEMINI_API_KEY")
                    if not api_key:
                        st.error("GEMINI_API_KEY not found in secrets. Please add it.")
                        st.stop()
                    reviewer = NotebookReviewer(api_key=api_key)
                    review_result = reviewer.review_notebook(notebook_code)
                    st.session_state.review_result = review_result
            except Exception as e:
                st.error(f"An error occurred during review: {e}")
                if 'review_result' in st.session_state:
                    del st.session_state.review_result

        st.divider()

        # Display results
        if 'review_result' in st.session_state and st.session_state.review_result:
            st.subheader("📋 Review Results")
            st.markdown(st.session_state.review_result, unsafe_allow_html=True)
        
        with st.expander("View Uploaded Code", expanded=False):
            st.code(notebook_code, language='python', line_numbers=True)
    else:
        st.info("Upload a notebook file from the sidebar to begin the review process.")
        if 'review_result' in st.session_state:
            del st.session_state.review_result

def main():
    """Main function to run the Streamlit app."""
    set_page_background()

    st.title("Data Validation Studio 🛠️")

    # --- Load API Key ---
    # Check for the secret key. This is the standard way for Streamlit Community Cloud.
    if "GEMINI_API_KEY" not in st.secrets:
        st.error("🚨 GEMINI_API_KEY not found in Streamlit Secrets.")
        st.info("For local development, create a file at `.streamlit/secrets.toml` and add your key:\n\n`GEMINI_API_KEY = \"your_key_here\"`\n\nWhen deploying, add this in the app's advanced settings under 'Secrets'.")
        st.stop()

    # --- Session State Initialization ---
    if 'rules_df' not in st.session_state:
        st.session_state.rules_df = pd.DataFrame(columns=['column_name', 'data_type', 'rule'])
    if 'generated_notebook' not in st.session_state: 
        st.session_state.generated_notebook = None
    if 'mode' not in st.session_state:
        st.session_state.mode = "Suggest rules from a data sample"
    if 'generation_stats' not in st.session_state:
        st.session_state.generation_stats = None
    if 'review_result' not in st.session_state:
        st.session_state.review_result = None

    # --- Sidebar for Tool Selection ---
    with st.sidebar:
        add_sidebar_logo()
        st.header("Studio Tools")
        tool_selection = st.radio(
            "Select a tool:",
            ("Data Validation Generator", "Notebook Reviewer"),
            key="tool_selection"
        )
        st.divider()

    # --- Main Area with Tool Logic ---
    if st.session_state.tool_selection == "Data Validation Generator":
        # A more interactive and visual introduction for the generator
        st.markdown("This tool helps you generate a Databricks Python notebook for data quality validation. **Choose an option from the sidebar to get started.**")
        st.divider()
        col1, col2, col3 = st.columns(3, gap="large")
        with col1:
            with st.container(border=True):
                st.subheader("🔬 Suggest Rules")
                st.markdown("**Recommended**")
                st.write("Upload a sample of your data. The tool will automatically profile it and suggest relevant validation rules.")
        with col2:
            with st.container(border=True):
                st.subheader("⬆️ Upload Rules")
                st.write("Bring your own rules. Upload a CSV file containing your column names, data types, and rule descriptions.")
        with col3:
            with st.container(border=True):
                st.subheader("✍️ Create from Scratch")
                st.write("Start with a blank slate for maximum control. Use the interactive editor to build your validation rules one by one.")
        st.divider()
        render_validation_generator()
    elif st.session_state.tool_selection == "Notebook Reviewer":
        render_notebook_reviewer()

if __name__ == "__main__": # This ensures the app runs only when the script is executed directly
    main()