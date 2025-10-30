# Databricks notebook source
# MAGIC %md
# MAGIC ## Page 1

# COMMAND ----------

# MAGIC %md
# MAGIC # Pagewise Summary

# COMMAND ----------

import json
import os
import mlflow.deployments
from pyspark.sql import SparkSession
from pyspark.dbutils import DBUtils

# ---------------------------------------------------------------------------
# CONFIGURATION: YOU MUST EDIT THIS SECTION
# ---------------------------------------------------------------------------
# 1. SET THE PAGE TO PROCESS
#    (e.g., "page1", "page3", "page4")
PAGE_TO_PROCESS = "page1"

# 2. SET THE BASE PATH TO YOUR "pages" FOLDER
#    Example: "/Workspace/Users/your.email@company.com/LLM_Project/pages"
BASE_CONFIG_PATH = "/Workspace/Users/ayush.dongardive@pivotbio.com/LLM Pipeline/pages"

# 3. SET THE DATABRICKS MODEL SERVING ENDPOINT NAME
LLM_ENDPOINT_NAME = "databricks-llama-4-maverick"

# 4. PASTE THE HARDCODED DATA FOR THE PAGE HERE
#    This dictionary contains the data that would have come from the SQL query.
#    This example is pre-filled with the data for Page 1.
PAGE_DATA = {
    "experiment__name": "25-ZEAMX-US550-MN01-GETT",
    "experiment__planting_status": "Planted",
    "experiment__planting_date": "4/18/2025",
    "protocol__crop": "Corn",
    "location__county": "Cottonwood County",
    "location__state": "MN",
    "experiment__grower_full_name": "Harder, Darby",
    "experiment__connected_field_name": "Home-HARD-DARBYHARDER-MN",
    "experiment__fertilizer_total_n_lba_pos_ctl_avg": 170,
    "experiment__fertilizer_total_n_lba_neg_ctl_avg": 130,
    "experiment__by_trt_avg_n_lba_diff_of_full": -40,
    "experiment__fertilizer_n_lba_percent_diff_of_full": -23.5,
    "experiment__fertilizer_reduction_dates": "April 1, 2025",
    "experiment__fertilizer_products_with_reduction": "NH3: 82-0-0 (Anhydrous Ammonia)"
}
# ---------------------------------------------------------------------------

# Initialize Spark Session and DBUtils (standard for Databricks notebooks)
spark = SparkSession.builder.appName("SummarizationPipeline").getOrCreate()
dbutils = DBUtils(spark)

def load_config_files(page_folder_path, page_name):
    """
    Loads all necessary JSON configuration files for a given page.
    It gracefully handles the absence of the optional metrics file.
    """
    print("Loading configuration files...")
    try:
        # Define paths for all potential files
        prompt_path = os.path.join(page_folder_path, f"{page_name}_prompt.json")
        metadata_path = os.path.join(page_folder_path, f"{page_name}_column_metadata.json")
        metrics_path = os.path.join(page_folder_path, f"{page_name}_metrics_threshold.json")

        # Load mandatory files
        with open(prompt_path, 'r') as f:
            prompt_template = json.load(f)
        with open(metadata_path, 'r') as f:
            column_metadata = json.load(f)

        # Load optional metrics file
        metrics_thresholds = None
        if os.path.exists(metrics_path):
            with open(metrics_path, 'r') as f:
                metrics_thresholds = json.load(f)
                print(f"INFO: Successfully loaded optional metrics file from {metrics_path}")
        else:
            print(f"INFO: Optional file not found at {metrics_path}. Proceeding without it.")

        return prompt_template, column_metadata, metrics_thresholds
    except FileNotFoundError as e:
        print(f"ERROR: A mandatory configuration file was not found. Details: {e}")
        return None, None, None
    except json.JSONDecodeError as e:
        print(f"ERROR: Failed to parse a JSON configuration file. Please check for syntax errors. Details: {e}")
        return None, None, None

def assemble_llm_payload(prompt_template, page_data, column_metadata, metrics_thresholds):
    """
    Assembles the final JSON payload to be sent to the LLM.
    """
    print("Assembling the final LLM payload...")
    try:
        # Start with a deep copy of the template to avoid modifying the original
        final_payload = json.loads(json.dumps(prompt_template))

        # Navigate to the instruction block where data will be injected
        instruction_block = final_payload['instructions_and_output_structure']
        input_data_section = instruction_block['input_data']

        # Inject the hardcoded page data and metadata
        input_data_section['page_data'] = page_data
        input_data_section['column_metadata'] = column_metadata.get('column_metadata', [])

        # Inject metrics thresholds if they exist
        if metrics_thresholds:
            input_data_section['key_metrics_threshold_info'] = metrics_thresholds
        
        print("Payload assembled successfully.")
        return final_payload
    except KeyError as e:
        print(f"ERROR: The prompt template is missing a required key. Could not find key: {e}")
        return None

def call_llm(payload_dict, endpoint_name):
    """
    Calls the Databricks Model Serving endpoint using the mlflow.deployments client.
    """
    print("\n" + "="*70)
    print(f"Calling LLM Endpoint: {endpoint_name}...")

    try:
        # Convert the entire payload dictionary into a single JSON string for the prompt
        prompt = json.dumps(payload_dict, indent=2)

        # Initialize the deployment client
        client = mlflow.deployments.get_deploy_client("databricks")
        
        # Structure the final payload for the Databricks LLM serving API
        inputs_payload = {
            "messages": [
                {"role": "user", "content": prompt}
            ]
        }
        
        # Call the LLM deployment endpoint
        response = client.predict(
            endpoint=endpoint_name, 
            inputs=inputs_payload
        )
        
        # Extract the summary text from the response structure
        summary = response.get('choices', [{}])[0].get('message', {}).get('content', "Error: Could not extract content from LLM response.")
        
        # Clean up the summary (e.g., strip surrounding whitespace)
        summary = summary.strip()
        
        print("✅ LLM Summary Generated Successfully.")
        return summary

    except Exception as e:
        # Handle any connection or prediction errors gracefully
        error_message = f"Error generating summary from LLM endpoint: {str(e)}"
        print(f"❌ {error_message}")
        return error_message

def main():
    """
    Main function to orchestrate the summarization pipeline.
    """
    print(f"--- Starting Summary Generation for Page '{PAGE_TO_PROCESS}' ---")

    # Construct the full path to the page's configuration folder
    page_folder = os.path.join(BASE_CONFIG_PATH, PAGE_TO_PROCESS)

    # 1. Load all configuration files
    prompt_template, column_metadata, metrics_thresholds = load_config_files(page_folder, PAGE_TO_PROCESS)
    if not prompt_template:
        print("ERROR: Failed to load configuration. Aborting.")
        return

    # 2. Use the hardcoded page data
    page_data = PAGE_DATA
    print("Using hardcoded page data.")
    
    # 3. Assemble the final payload for the LLM
    llm_payload = assemble_llm_payload(prompt_template, page_data, column_metadata, metrics_thresholds)
    if not llm_payload:
        print("ERROR: Failed to assemble LLM payload. Aborting.")
        return

    # 4. Call the LLM to get the summary
    summary = call_llm(llm_payload, LLM_ENDPOINT_NAME)
    if not summary:
        print("ERROR: Failed to generate summary. Aborting.")
        return
        
    # 5. Display the final summary
    print("\n--- FINAL SUMMARY PREVIEW ---")
    print(summary)
    print("---------------------------\n")

# Run the main pipeline function
if __name__ == "__main__":
    main()
    

# COMMAND ----------

# MAGIC %md
# MAGIC ## Comparative Metric

# COMMAND ----------

df = spark.read.table("gold.fieldtrials_filtered.cur_flat_plots")
df.display() # To view the DataFrame

# COMMAND ----------

# MAGIC %md
# MAGIC Filtering on State and Crop

# COMMAND ----------

from pyspark.sql.functions import col
df_filtered = df.filter((col("location__state") == 'MN') & (col("protocol__crop") == 'Corn'))
display(df_filtered)


# COMMAND ----------

import json
from pyspark.sql.functions import avg, col

# --- 1. Aggregation (Your existing code) ---

# Note: The column 'experiment__by_trt_avg_n_lba_diff_of_full' 
# was intentionally excluded here as it caused the MAP<STRING, DOUBLE> error 
# in your previous prompt. Assuming you only want to convert the three successful columns.

avg_cols = [
    avg("experiment__fertilizer_total_n_lba_pos_ctl_avg").alias("avg_full_rate_n_applied"),
    avg("experiment__fertilizer_total_n_lba_neg_ctl_avg").alias("avg_reduced_rate_n_applied"),
    avg("experiment__fertilizer_n_lba_percent_diff_of_full").alias("avg_percent_of_n_reduced")
]

# Assuming 'df_filtered' is your filtered DataFrame
avg_df = df_filtered.select(*avg_cols)

# --- 2. Convert DataFrame to JSON String ---

# Collect the single row of results into a Python Row object
# Use .limit(1) as a safeguard, though aggregation should result in one row
avg_row = avg_df.limit(1).collect()[0]

# Convert the Row object to a standard Python dictionary
avg_dict = avg_row.asDict()

# Convert the Python dictionary to a JSON string
json_output = json.dumps(avg_dict, indent=4)

# --- 3. Display the result ---

print("✅ JSON Output of Averages:")
print(json_output)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Comparison summary

# COMMAND ----------

import json
import os
import mlflow.deployments
from pyspark.sql import SparkSession
from pyspark.dbutils import DBUtils

# Initialize Spark Session and DBUtils (standard for Databricks notebooks)
spark = SparkSession.builder.appName("SummarizationPipeline").getOrCreate()
dbutils = DBUtils(spark)

# ---------------------------------------------------------------------------
# GLOBAL LLM INTERACTION FUNCTIONS (Reusable across all pipelines)
# ---------------------------------------------------------------------------

def load_comparative_metrics(base_path, page_name):
    """
    Loads the JSON file containing the regional comparative metrics.
    Assumes file is named "{page_name}_comparative_metrics.json".
    """
    print("Loading comparative data file...")
    comparative_path = os.path.join(base_path, f"{page_name}_comparative_metrics.json")

    try:
        with open(comparative_path, 'r') as f:
            comparative_metrics = json.load(f)
            print(f"INFO: Successfully loaded comparative metrics from {comparative_path}")
            return comparative_metrics
    except FileNotFoundError as e:
        print(f"ERROR: Comparative metrics file not found at {comparative_path}. Aborting.")
        return None
    except json.JSONDecodeError as e:
        print(f"ERROR: Failed to parse the comparative metrics JSON. Details: {e}")
        return None

def assemble_llm_payload(prompt_template, trial_data, comparative_data):
    """
    Assembles the final LLM prompt string by injecting trial and comparative data.
    """
    print("Assembling the final LLM payload...")
    
    # Format the data into clean JSON strings for the LLM
    trial_data_str = json.dumps(trial_data, indent=2)
    comparative_data_str = json.dumps(comparative_data, indent=2)

    # Inject data into the prompt template using .format()
    final_prompt = prompt_template.format(
        trial_data=trial_data_str,
        comparative_data=comparative_data_str
    )
    
    # Structure the final payload for the Databricks LLM serving API
    llm_payload = {
        "messages": [
            {"role": "user", "content": final_prompt}
        ]
    }

    print("Payload assembled successfully.")
    return llm_payload

def call_llm(payload_dict, endpoint_name):
    """
    Calls the Databricks Model Serving endpoint.
    """
    print("\n" + "="*70)
    print(f"Calling LLM Endpoint: {endpoint_name}...")

    try:
        client = mlflow.deployments.get_deploy_client("databricks")
        
        response = client.predict(
            endpoint=endpoint_name, 
            inputs=payload_dict
        )
        
        # Extract the summary text
        summary = response.get('choices', [{}])[0].get('message', {}).get('content', "Error: Could not extract content from LLM response.")
        summary = summary.strip()
        
        print("✅ LLM Summary Generated Successfully.")
        return summary

    except Exception as e:
        error_message = f"Error generating summary from LLM endpoint: {str(e)}"
        print(f"❌ {error_message}")
        return error_message

# ---------------------------------------------------------------------------
# MAIN PIPELINE EXECUTION FUNCTION
# ---------------------------------------------------------------------------

def run_pipeline(
    page_name: str, 
    base_data_path: str, 
    llm_endpoint_name: str, 
    specific_trial_data: dict, 
    llm_prompt_template: str
):
    """
    Orchestrates the LLM summarization pipeline for a single trial.
    """
    print(f"--- Starting Summary Generation for Page '{page_name}' ---")

    # 1. Load the comparative data from the specified path/file
    comparative_data = load_comparative_metrics(base_data_path, page_name)
    if not comparative_data:
        print("ERROR: Failed to load comparative data. Aborting.")
        return

    # 2. Assemble the final LLM payload
    llm_payload = assemble_llm_payload(llm_prompt_template, specific_trial_data, comparative_data)
    
    # 3. Call the LLM to get the summary
    summary = call_llm(llm_payload, llm_endpoint_name)
    if summary and not summary.startswith("Error"):
        # 4. Display the final summary
        print("\n--- FINAL SUMMARY PREVIEW ---")
        print(summary)
        print("---------------------------\n")
    else:
        print("ERROR: Failed to generate summary. Aborting.")
        
    return summary

# ---------------------------------------------------------------------------
# EXECUTION BLOCK (How you would call the function)
# ---------------------------------------------------------------------------

# --- CONFIGURATION SECTION (Typically placed at the top of a notebook) ---

PAGE_TO_PROCESS = "page1"
BASE_DATA_PATH = "/Workspace/Users/ayush.dongardive@pivotbio.com/LLM Pipeline/Comparitive Metric"
LLM_ENDPOINT_NAME = "databricks-llama-4-maverick"

# Data for the specific trial page (comes from a hardcoded block or SQL query)
SPECIFIC_TRIAL_DATA = {
    "experiment__name": "25-ZEAMX-US550-MN01-GETT",
    "location__state": "MN",
    "experiment__fertilizer_total_n_lba_pos_ctl_avg": 170,
    "experiment__fertilizer_total_n_lba_neg_ctl_avg": 130,
    "experiment__by_trt_avg_n_lba_diff_of_full": -40,
    "experiment__fertilizer_n_lba_percent_diff_of_full": -23.5,
}

# The prompt (comes from an external source or is defined here)
# NOTE: The prompt structure MUST contain {trial_data} and {comparative_data} placeholders.
LLM_COMPARISON_PROMPT = """
{{
  "role": "You are an Analytical AI Assistant for the Pivot Bio Sales and Agronomy Team, specializing in comparing trial setup and strategy against regional benchmarks.",
  "company_context": "Pivot Bio is a U.S.-based agricultural biotechnology company that develops microbe-based nitrogen solutions to replace synthetic fertilizers. Their flagship product, Pivot Bio PROVEN® 40, replaces synthetic nitrogen, improving yield consistency and environmental outcomes. They run field-scale validation trials across the U.S. to demonstrate nitrogen-reduction performance under various conditions.",
  "task": "Analyze the provided data for this specific trial's N-management setup in the context of the regional averages. The goal is to generate a comprehensive comparative summary on how this specific trial's reduction strategy is positioned relative to its peers.",
  "instructions_and_output_structure": {{
    "input_data_format": "You will receive two primary JSON data blocks: 'specific_trial_data' (setup for the current trial) and 'comparative_metrics' (regional averages calculated across all similar trials).",
    "input_data": {{
      "specific_trial_data": "{trial_data}",  
      "comparative_metrics": "{comparative_data}"
    }},
    "output_format": {{
      "sections": [
        {{
          "title": "Strategy Comparison: Trial vs. Regional Peer Group",
          "instruction": "Compare the current trial's N-reduction strategy to the regional averages. State the trial's Full N Rate (GSP) and Reduced N Rate. Contrast the trial's Percent of N Reduced with the regional average Percent of N Reduced. Clearly state if the trial's strategy is more conservative or more aggressive than the peer group."
        }},
        {{
          "title": "Technical Positioning and Implication",
          "instruction": "Calculate the difference in lb/ac between the trial's N Reduction and the approximate regional average N Reduction (using the regional average on the trial's GSP). Explain the agronomic implication of this difference. If the trial is more conservative, imply higher yield stability; if more aggressive, imply higher potential N-use efficiency but greater risk."
        }},
        {{
          "title": "Sales and Agronomic Takeaway",
          "instruction": "Deliver a 2-3 sentence business interpretation for the sales team. Frame the trial's N-reduction strategy as a clear validation point for a Pivot Bio product claim. Emphasize what the comparison reveals about the grower's confidence (or lack thereof) relative to the regional peer group, providing a clear action point for follow-up."
        }}
      ],
      "tone": "Analytical, comparative, and highly business-aligned.",
      "length": "250-300 words."
    }}
}}
"""

# --- RUN THE PIPELINE ---
if __name__ == "__main__":
    run_pipeline(
        page_name=PAGE_TO_PROCESS,
        base_data_path=BASE_DATA_PATH,
        llm_endpoint_name=LLM_ENDPOINT_NAME,
        specific_trial_data=SPECIFIC_TRIAL_DATA,
        llm_prompt_template=LLM_COMPARISON_PROMPT
    )

# COMMAND ----------

# MAGIC %md
# MAGIC ### Page 3

# COMMAND ----------

import json
import os
import mlflow.deployments
from pyspark.sql import SparkSession
from pyspark.dbutils import DBUtils

# ---------------------------------------------------------------------------
# CONFIGURATION: YOU MUST EDIT THIS SECTION
# ---------------------------------------------------------------------------
# 1. SET THE PAGE TO PROCESS
#    (e.g., "page1", "page3", "page4")
PAGE_TO_PROCESS = "page3"

# 2. SET THE BASE PATH TO YOUR "pages" FOLDER
#    Example: "/Workspace/Users/your.email@company.com/LLM_Project/pages"
BASE_CONFIG_PATH = "/Workspace/Users/ayush.dongardive@pivotbio.com/LLM Pipeline/pages"

# 3. SET THE DATABRICKS MODEL SERVING ENDPOINT NAME
LLM_ENDPOINT_NAME = "databricks-llama-4-maverick"

# 4. PASTE THE HARDCODED DATA FOR THE PAGE HERE
#    This dictionary contains the data that would have come from the SQL query.
#    This example is pre-filled with the data for Page 1.
PAGE_DATA = {
    "Nitrate (lbs)": 59.6,
  "WEO C (lbs)": 172.6,
  "WEO N (lbs)": 60.2,
  "C:N Ratio": 5.2,
  "CEC (meq/100g)": 21.1,
  "OM (%)": 4.1,
  "pH": 5.4,
  "Buffer pH": 6.5,
  "Bray P (ppm)": 34.3,
  "Olsen P (ppm)": 19.1,
  "Potassium (K) (ppm)": 159.9,
  "Sulfur (S) (ppm)": 14.2,
  "Overall N Rating Score": 2.6,
  "Overall N Rating Text": "Low",
  "Overall P Rating Score": 4.0,
  "Overall P Rating Text": "High",
  "Overall K Rating Score": 3.0,
  "Overall K Rating Text": "Medium",
  "Overall S Rating Score": "N/A",
  "Overall S Rating Text": "No Rating"
}
# ---------------------------------------------------------------------------

# Initialize Spark Session and DBUtils (standard for Databricks notebooks)
spark = SparkSession.builder.appName("SummarizationPipeline").getOrCreate()
dbutils = DBUtils(spark)

def load_config_files(page_folder_path, page_name):
    """
    Loads all necessary JSON configuration files for a given page.
    It gracefully handles the absence of the optional metrics file.
    """
    print("Loading configuration files...")
    try:
        # Define paths for all potential files
        prompt_path = os.path.join(page_folder_path, f"{page_name}_prompt.json")
        metadata_path = os.path.join(page_folder_path, f"{page_name}_column_metadata.json")
        metrics_path = os.path.join(page_folder_path, f"{page_name}_metrics.json")

        # Load mandatory files
        with open(prompt_path, 'r') as f:
            prompt_template = json.load(f)
        with open(metadata_path, 'r') as f:
            column_metadata = json.load(f)

        # Load optional metrics file
        metrics_thresholds = None
        if os.path.exists(metrics_path):
            with open(metrics_path, 'r') as f:
                metrics_thresholds = json.load(f)
                print(f"INFO: Successfully loaded optional metrics file from {metrics_path}")
        else:
            print(f"INFO: Optional file not found at {metrics_path}. Proceeding without it.")

        return prompt_template, column_metadata, metrics_thresholds
    except FileNotFoundError as e:
        print(f"ERROR: A mandatory configuration file was not found. Details: {e}")
        return None, None, None
    except json.JSONDecodeError as e:
        print(f"ERROR: Failed to parse a JSON configuration file. Please check for syntax errors. Details: {e}")
        return None, None, None

def assemble_llm_payload(prompt_template, page_data, column_metadata, metrics_thresholds):
    """
    Assembles the final JSON payload to be sent to the LLM.
    """
    print("Assembling the final LLM payload...")
    try:
        # Start with a deep copy of the template to avoid modifying the original
        final_payload = json.loads(json.dumps(prompt_template))

        # Navigate to the instruction block where data will be injected
        instruction_block = final_payload['instructions_and_output_structure']
        input_data_section = instruction_block['input_data']

        # Inject the hardcoded page data and metadata
        input_data_section['page_data'] = page_data
        input_data_section['column_metadata'] = column_metadata.get('column_metadata', [])

        # Inject metrics thresholds if they exist
        if metrics_thresholds:
            input_data_section['key_metrics_threshold_info'] = metrics_thresholds
        
        print("Payload assembled successfully.")
        return final_payload
    except KeyError as e:
        print(f"ERROR: The prompt template is missing a required key. Could not find key: {e}")
        return None

def call_llm(payload_dict, endpoint_name):
    """
    Calls the Databricks Model Serving endpoint using the mlflow.deployments client.
    """
    print("\n" + "="*70)
    print(f"Calling LLM Endpoint: {endpoint_name}...")

    try:
        # Convert the entire payload dictionary into a single JSON string for the prompt
        prompt = json.dumps(payload_dict, indent=2)

        # Initialize the deployment client
        client = mlflow.deployments.get_deploy_client("databricks")
        
        # Structure the final payload for the Databricks LLM serving API
        inputs_payload = {
            "messages": [
                {"role": "user", "content": prompt}
            ]
        }
        
        # Call the LLM deployment endpoint
        response = client.predict(
            endpoint=endpoint_name, 
            inputs=inputs_payload
        )
        
        # Extract the summary text from the response structure
        summary = response.get('choices', [{}])[0].get('message', {}).get('content', "Error: Could not extract content from LLM response.")
        
        # Clean up the summary (e.g., strip surrounding whitespace)
        summary = summary.strip()
        
        print("✅ LLM Summary Generated Successfully.")
        return summary

    except Exception as e:
        # Handle any connection or prediction errors gracefully
        error_message = f"Error generating summary from LLM endpoint: {str(e)}"
        print(f"❌ {error_message}")
        return error_message

def main():
    """
    Main function to orchestrate the summarization pipeline.
    """
    print(f"--- Starting Summary Generation for Page '{PAGE_TO_PROCESS}' ---")

    # Construct the full path to the page's configuration folder
    page_folder = os.path.join(BASE_CONFIG_PATH, PAGE_TO_PROCESS)

    # 1. Load all configuration files
    prompt_template, column_metadata, metrics_thresholds = load_config_files(page_folder, PAGE_TO_PROCESS)
    if not prompt_template:
        print("ERROR: Failed to load configuration. Aborting.")
        return

    # 2. Use the hardcoded page data
    page_data = PAGE_DATA
    print("Using hardcoded page data.")
    
    # 3. Assemble the final payload for the LLM
    llm_payload = assemble_llm_payload(prompt_template, page_data, column_metadata, metrics_thresholds)
    if not llm_payload:
        print("ERROR: Failed to assemble LLM payload. Aborting.")
        return

    # 4. Call the LLM to get the summary
    summary = call_llm(llm_payload, LLM_ENDPOINT_NAME)
    if not summary:
        print("ERROR: Failed to generate summary. Aborting.")
        return
        
    # 5. Display the final summary
    print("\n--- FINAL SUMMARY PREVIEW ---")
    print(summary)
    print("---------------------------\n")

# Run the main pipeline function
if __name__ == "__main__":
    main()
    

# COMMAND ----------

# MAGIC %md
# MAGIC ### Comparative Summary

# COMMAND ----------

import json
import os
import mlflow.deployments
from pyspark.sql import SparkSession
# Initialize Spark session (Databricks automatically provides dbutils)
spark = SparkSession.builder.appName("SummarizationPipeline").getOrCreate()
# ---------------------------------------------------------------------------
# GLOBAL LLM INTERACTION FUNCTIONS (Reusable across all pipelines)
# ---------------------------------------------------------------------------
def load_comparative_metrics(base_path, page_name):
    """
    Loads the JSON file containing the regional comparative metrics.
    """
    print("Loading comparative data file...")
    comparative_path = os.path.join(base_path, f"{page_name}_comparative_metrics.json")

    try:
        with open(comparative_path, "r") as f:
            comparative_metrics = json.load(f)
            print(f"INFO: Successfully loaded comparative metrics from {comparative_path}")
            return comparative_metrics
    except FileNotFoundError:
        print(f"ERROR: Comparative metrics file not found at {comparative_path}. Aborting.")
        return None
    except json.JSONDecodeError as e:
        print(f"ERROR: Failed to parse comparative metrics JSON. Details: {e}")
        return None
def assemble_llm_payload(prompt_template, trial_data, comparative_data):
    """
    Assembles the final LLM prompt string by injecting trial and comparative data.
    """
    print("Assembling the final LLM payload...")
    trial_data_str = json.dumps(trial_data, indent=2)
    comparative_data_str = json.dumps(comparative_data, indent=2)
    final_prompt = prompt_template.format(
        trial_data=trial_data_str,
        comparative_data=comparative_data_str
    )
    llm_payload = {"messages": [{"role": "user", "content": final_prompt}]}
    print("Payload assembled successfully.")
    return llm_payload
def call_llm(payload_dict, endpoint_name):
    """
    Calls the Databricks Model Serving endpoint.
    """
    print("\n" + "=" * 70)
    print(f"Calling LLM Endpoint: {endpoint_name}...")
    try:
        client = mlflow.deployments.get_deploy_client("databricks")
        # Depending on the model type, you may need to use 'inputs' instead of 'messages'
        response = client.predict(endpoint=endpoint_name, inputs=payload_dict)
        summary = (
            response.get("choices", [{}])[0]
            .get("message", {})
            .get("content", "Error: Could not extract content from LLM response.")
            .strip()
        )
        print(":white_check_mark: LLM Summary Generated Successfully.")
        return summary
    except Exception as e:
        error_message = f"Error generating summary from LLM endpoint: {str(e)}"
        print(f":x: {error_message}")
        return error_message
# ---------------------------------------------------------------------------
# MAIN PIPELINE EXECUTION FUNCTION
# ---------------------------------------------------------------------------
def run_pipeline(page_name, base_data_path, llm_endpoint_name, specific_trial_data, llm_prompt_template):
    """
    Orchestrates the LLM summarization pipeline for a single trial.
    """
    print(f"--- Starting Summary Generation for Page '{page_name}' ---")
    comparative_data = load_comparative_metrics(base_data_path, page_name)
    if not comparative_data:
        print("ERROR: Failed to load comparative data. Aborting.")
        return
    llm_payload = assemble_llm_payload(llm_prompt_template, specific_trial_data, comparative_data)
    summary = call_llm(llm_payload, llm_endpoint_name)
    if summary and not summary.startswith("Error"):
        print("\n--- FINAL SUMMARY PREVIEW ---")
        print(summary)
        print("---------------------------\n")
    else:
        print("ERROR: Failed to generate summary. Aborting.")
    return summary
# ---------------------------------------------------------------------------
# CONFIGURATION SECTION
# ---------------------------------------------------------------------------
PAGE_TO_PROCESS = "page3"
BASE_DATA_PATH = "/Workspace/Users/ayush.dongardive@pivotbio.com/LLM Pipeline/Comparitive Metric"
LLM_ENDPOINT_NAME = "databricks-llama-4-maverick"
SPECIFIC_TRIAL_DATA = {
  "Nitrate (lbs)": 59.6,
  "WEO C (lbs)": 172.6,
  "WEO N (lbs)": 60.2,
  "C:N Ratio": 5.2,
  "CEC (meq/100g)": 21.1,
  "OM (%)": 4.1,
  "pH": 5.4,
  "Buffer pH": 6.5,
  "Bray P (ppm)": 34.3,
  "Olsen P (ppm)": 19.1,
  "Potassium (K) (ppm)": 159.9,
  "Sulfur (S) (ppm)": 14.2,
  "Overall N Rating Score": 2.6,
  "Overall N Rating Text": "Low",
  "Overall P Rating Score": 4.0,
  "Overall P Rating Text": "High",
  "Overall K Rating Score": 3.0,
  "Overall K Rating Text": "Medium",
  "Overall S Rating Score": "N/A",
  "Overall S Rating Text": "No Rating"
}
# Use triple single quotes to avoid conflicts with double quotes in JSON-style text
LLM_COMPARISON_PROMPT = '''
You are an Analytical AI Assistant for the Pivot Bio Sales and Agronomy Team.
You specialize in comparing early season soil health and nutrient availability against regional benchmarks (State Average data).
Company Context:
Pivot Bio is a U.S.-based agricultural biotechnology company that develops microbe-based nitrogen solutions to replace synthetic fertilizers.
Their flagship product, Pivot Bio PROVEN® 40, replaces synthetic nitrogen, improving yield consistency and environmental outcomes.
They run field-scale validation trials across the U.S. to demonstrate nitrogen-reduction performance.
Task:
Analyze the provided Early Season Soil Analysis Data (in 'specific_trial_data') and compare it against the regional averages (in 'comparative_metrics'). Generate a comprehensive comparative summary focusing on identifying the site's unique characteristics and major limiting factors relative to the state.
Context Summary:
"The soil analysis shows the trial site has high native fertility with high CEC and high OM, but the overall N rating is low. All individual soil metrics (Nitrate, P, K, S) are technically rated 'High' except for a lower-than-optimal pH (5.4), which could limit nutrient availability, particularly for N and P."
Input Data:
- Specific Trial Data: {trial_data}
- Comparative Metrics: {comparative_data}
Output:
Write an analytical, comparative, business-aligned summary (250–300 words) with three sections:
1. Soil Health and Fertility Comparison:
Instruction: Compare the trial's key metrics (CEC, OM, pH, Nitrate) to the regional averages. **Provide an insight** on whether the trial site is generally more or less fertile, and identify the single largest structural or chemical difference compared to the state average (e.g., pH difference, CEC difference).
2. Potential Nutrient Limitation Analysis:
Instruction: **Provide an insight** by comparing the trial's specific nutrient levels (Nitrate, P, K, S) to the regional averages. Based on the **trial's low pH** and the **overall low N rating**, identify the single biggest potential early-season soil-based constraint for the crop (e.g., Nitrogen supply, Phosphorus availability, etc.).
3. Sales & Agronomic Takeaway:
Instruction: **Provide an insight** into the sales narrative. Conclude that the site's naturally high fertility (high CEC/OM) is a strong argument for PROVEN® 40 (since less synthetic N is needed), but the acidic pH is a critical factor that requires attention, shifting the grower's immediate focus to soil amendment rather than just N management.
'''
# ---------------------------------------------------------------------------
# RUN PIPELINE
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    run_pipeline(
        page_name=PAGE_TO_PROCESS,
        base_data_path=BASE_DATA_PATH,
        llm_endpoint_name=LLM_ENDPOINT_NAME,
        specific_trial_data=SPECIFIC_TRIAL_DATA,
        llm_prompt_template=LLM_COMPARISON_PROMPT
    )


# COMMAND ----------

# MAGIC %md
# MAGIC ## Page 4

# COMMAND ----------

import json
import os
import mlflow.deployments
from pyspark.sql import SparkSession
from pyspark.dbutils import DBUtils

# ---------------------------------------------------------------------------
# CONFIGURATION: YOU MUST EDIT THIS SECTION
# ---------------------------------------------------------------------------
# 1. SET THE PAGE TO PROCESS
#    (e.g., "page1", "page3", "page4")
PAGE_TO_PROCESS = "page4"

# 2. SET THE BASE PATH TO YOUR "pages" FOLDER
#    Example: "/Workspace/Users/your.email@company.com/LLM_Project/pages"
BASE_CONFIG_PATH = "/Workspace/Users/ayush.dongardive@pivotbio.com/LLM Pipeline/pages"

# 3. SET THE DATABRICKS MODEL SERVING ENDPOINT NAME
LLM_ENDPOINT_NAME = "databricks-llama-4-maverick"

# 4. PASTE THE HARDCODED DATA FOR THE PAGE HERE
#    This dictionary contains the data that would have come from the SQL query.
#    This example is pre-filled with the data for Page 1.
PAGE_DATA = {
    "N%":" 4.40",
    "P%":"0.41",
    "S%":"0.29",
    "K%":"1.47"
}
# ---------------------------------------------------------------------------

# Initialize Spark Session and DBUtils (standard for Databricks notebooks)
spark = SparkSession.builder.appName("SummarizationPipeline").getOrCreate()
dbutils = DBUtils(spark)

def load_config_files(page_folder_path, page_name):
    """
    Loads all necessary JSON configuration files for a given page.
    It gracefully handles the absence of the optional metrics file.
    """
    print("Loading configuration files...")
    try:
        # Define paths for all potential files
        prompt_path = os.path.join(page_folder_path, f"{page_name}_prompt.json")
        metadata_path = os.path.join(page_folder_path, f"{page_name}_column_metadata.json")
        metrics_path = os.path.join(page_folder_path, f"{page_name}_metrics.json")

        # Load mandatory files
        with open(prompt_path, 'r') as f:
            prompt_template = json.load(f)
        with open(metadata_path, 'r') as f:
            column_metadata = json.load(f)

        # Load optional metrics file
        metrics_thresholds = None
        if os.path.exists(metrics_path):
            with open(metrics_path, 'r') as f:
                metrics_thresholds = json.load(f)
                print(f"INFO: Successfully loaded optional metrics file from {metrics_path}")
        else:
            print(f"INFO: Optional file not found at {metrics_path}. Proceeding without it.")

        return prompt_template, column_metadata, metrics_thresholds
    except FileNotFoundError as e:
        print(f"ERROR: A mandatory configuration file was not found. Details: {e}")
        return None, None, None
    except json.JSONDecodeError as e:
        print(f"ERROR: Failed to parse a JSON configuration file. Please check for syntax errors. Details: {e}")
        return None, None, None

def assemble_llm_payload(prompt_template, page_data, column_metadata, metrics_thresholds):
    """
    Assembles the final JSON payload to be sent to the LLM.
    """
    print("Assembling the final LLM payload...")
    try:
        # Start with a deep copy of the template to avoid modifying the original
        final_payload = json.loads(json.dumps(prompt_template))

        # Navigate to the instruction block where data will be injected
        instruction_block = final_payload['instructions_and_output_structure']
        input_data_section = instruction_block['input_data']

        # Inject the hardcoded page data and metadata
        input_data_section['page_data'] = page_data
        input_data_section['column_metadata'] = column_metadata.get('column_metadata', [])

        # Inject metrics thresholds if they exist
        if metrics_thresholds:
            input_data_section['key_metrics_threshold_info'] = metrics_thresholds
        
        print("Payload assembled successfully.")
        return final_payload
    except KeyError as e:
        print(f"ERROR: The prompt template is missing a required key. Could not find key: {e}")
        return None

def call_llm(payload_dict, endpoint_name):
    """
    Calls the Databricks Model Serving endpoint using the mlflow.deployments client.
    """
    print("\n" + "="*70)
    print(f"Calling LLM Endpoint: {endpoint_name}...")

    try:
        # Convert the entire payload dictionary into a single JSON string for the prompt
        prompt = json.dumps(payload_dict, indent=2)

        # Initialize the deployment client
        client = mlflow.deployments.get_deploy_client("databricks")
        
        # Structure the final payload for the Databricks LLM serving API
        inputs_payload = {
            "messages": [
                {"role": "user", "content": prompt}
            ]
        }
        
        # Call the LLM deployment endpoint
        response = client.predict(
            endpoint=endpoint_name, 
            inputs=inputs_payload
        )
        
        # Extract the summary text from the response structure
        summary = response.get('choices', [{}])[0].get('message', {}).get('content', "Error: Could not extract content from LLM response.")
        
        # Clean up the summary (e.g., strip surrounding whitespace)
        summary = summary.strip()
        
        print("✅ LLM Summary Generated Successfully.")
        return summary

    except Exception as e:
        # Handle any connection or prediction errors gracefully
        error_message = f"Error generating summary from LLM endpoint: {str(e)}"
        print(f"❌ {error_message}")
        return error_message

def main():
    """
    Main function to orchestrate the summarization pipeline.
    """
    print(f"--- Starting Summary Generation for Page '{PAGE_TO_PROCESS}' ---")

    # Construct the full path to the page's configuration folder
    page_folder = os.path.join(BASE_CONFIG_PATH, PAGE_TO_PROCESS)

    # 1. Load all configuration files
    prompt_template, column_metadata, metrics_thresholds = load_config_files(page_folder, PAGE_TO_PROCESS)
    if not prompt_template:
        print("ERROR: Failed to load configuration. Aborting.")
        return

    # 2. Use the hardcoded page data
    page_data = PAGE_DATA
    print("Using hardcoded page data.")
    
    # 3. Assemble the final payload for the LLM
    llm_payload = assemble_llm_payload(prompt_template, page_data, column_metadata, metrics_thresholds)
    if not llm_payload:
        print("ERROR: Failed to assemble LLM payload. Aborting.")
        return

    # 4. Call the LLM to get the summary
    summary = call_llm(llm_payload, LLM_ENDPOINT_NAME)
    if not summary:
        print("ERROR: Failed to generate summary. Aborting.")
        return
        
    # 5. Display the final summary
    print("\n--- FINAL SUMMARY PREVIEW ---")
    print(summary)
    print("---------------------------\n")

# Run the main pipeline function
if __name__ == "__main__":
    main()
    

# COMMAND ----------

# MAGIC %md
# MAGIC Comparative Summary

# COMMAND ----------

import json
import os
import mlflow.deployments
from pyspark.sql import SparkSession
# Initialize Spark session (Databricks automatically provides dbutils)
spark = SparkSession.builder.appName("SummarizationPipeline").getOrCreate()
# ---------------------------------------------------------------------------
# GLOBAL LLM INTERACTION FUNCTIONS (Reusable across all pipelines)
# ---------------------------------------------------------------------------
def load_comparative_metrics(base_path, page_name):
    """
    Loads the JSON file containing the regional comparative metrics.
    """
    print("Loading comparative data file...")
    comparative_path = os.path.join(base_path, f"{page_name}_comparative_metrics.json")

    try:
        with open(comparative_path, "r") as f:
            comparative_metrics = json.load(f)
            print(f"INFO: Successfully loaded comparative metrics from {comparative_path}")
            return comparative_metrics
    except FileNotFoundError:
        print(f"ERROR: Comparative metrics file not found at {comparative_path}. Aborting.")
        return None
    except json.JSONDecodeError as e:
        print(f"ERROR: Failed to parse comparative metrics JSON. Details: {e}")
        return None
def assemble_llm_payload(prompt_template, trial_data, comparative_data):
    """
    Assembles the final LLM prompt string by injecting trial and comparative data.
    """
    print("Assembling the final LLM payload...")
    trial_data_str = json.dumps(trial_data, indent=2)
    comparative_data_str = json.dumps(comparative_data, indent=2)
    final_prompt = prompt_template.format(
        trial_data=trial_data_str,
        comparative_data=comparative_data_str
    )
    llm_payload = {"messages": [{"role": "user", "content": final_prompt}]}
    print("Payload assembled successfully.")
    return llm_payload
def call_llm(payload_dict, endpoint_name):
    """
    Calls the Databricks Model Serving endpoint.
    """
    print("\n" + "=" * 70)
    print(f"Calling LLM Endpoint: {endpoint_name}...")
    try:
        client = mlflow.deployments.get_deploy_client("databricks")
        # Depending on the model type, you may need to use 'inputs' instead of 'messages'
        response = client.predict(endpoint=endpoint_name, inputs=payload_dict)
        summary = (
            response.get("choices", [{}])[0]
            .get("message", {})
            .get("content", "Error: Could not extract content from LLM response.")
            .strip()
        )
        print(":white_check_mark: LLM Summary Generated Successfully.")
        return summary
    except Exception as e:
        error_message = f"Error generating summary from LLM endpoint: {str(e)}"
        print(f":x: {error_message}")
        return error_message
# ---------------------------------------------------------------------------
# MAIN PIPELINE EXECUTION FUNCTION
# ---------------------------------------------------------------------------
def run_pipeline(page_name, base_data_path, llm_endpoint_name, specific_trial_data, llm_prompt_template):
    """
    Orchestrates the LLM summarization pipeline for a single trial.
    """
    print(f"--- Starting Summary Generation for Page '{page_name}' ---")
    comparative_data = load_comparative_metrics(base_data_path, page_name)
    if not comparative_data:
        print("ERROR: Failed to load comparative data. Aborting.")
        return
    llm_payload = assemble_llm_payload(llm_prompt_template, specific_trial_data, comparative_data)
    summary = call_llm(llm_payload, llm_endpoint_name)
    if summary and not summary.startswith("Error"):
        print("\n--- FINAL SUMMARY PREVIEW ---")
        print(summary)
        print("---------------------------\n")
    else:
        print("ERROR: Failed to generate summary. Aborting.")
    return summary
# ---------------------------------------------------------------------------
# CONFIGURATION SECTION
# ---------------------------------------------------------------------------
PAGE_TO_PROCESS = "page4"
BASE_DATA_PATH = "/Workspace/Users/ayush.dongardive@pivotbio.com/LLM Pipeline/Comparitive Metric"
LLM_ENDPOINT_NAME = "databricks-llama-4-maverick"
SPECIFIC_TRIAL_DATA = {
    "N%": "4.40",
    "P%": "0.41",
    "S%": "0.29",
    "K%": "1.47"
}
# Use triple single quotes to avoid conflicts with double quotes in JSON-style text
LLM_COMPARISON_PROMPT = '''
You are an Analytical AI Assistant for the Pivot Bio Sales and Agronomy Team.
You specialize in comparing trial setup and mid-season nutrient status against regional benchmarks.
Company Context:
Pivot Bio is a U.S.-based agricultural biotechnology company that develops microbe-based nitrogen solutions to replace synthetic fertilizers.
Their flagship product, Pivot Bio PROVEN® 40, replaces synthetic nitrogen, improving yield consistency and environmental outcomes.
They run field-scale validation trials across the U.S. to demonstrate nitrogen-reduction performance.
Task:
Analyze the provided data for this specific trial, comparing its N-management strategy and mid-season nutrient status (N, P, K, S)
against regional averages and nutrient targets. Use the provided context summary as foundation.
Context Summary:
"Ground Truth
The tissue analysis at the V16 growth stage shows average ratings for N (4.40%), P (0.41%), K (1.47%), and S (0.29%).
Nitrogen levels were consistent across all treatments.
Technical Understanding
- N (4.40%) is above the 3.50–4.20% threshold → sufficient/high.
- P (0.41%) is within the 0.30–0.50% threshold → sufficient.
- S (0.29%) is at upper threshold → sufficient.
- K (1.47%) is below 1.75–2.50% threshold → deficient."
Sales Takeaway:
High N levels in reduced-rate Pivot Bio treatments show product effectiveness mid-season.
K deficiency offers a consultative opportunity for agronomists to recommend K adjustments."
Input Data:
- Specific Trial Data: {trial_data}
- Comparative Metrics: {comparative_data}
Output:
Write an analytical, comparative, business-aligned summary (250–300 words) with three sections:
1. Strategy Comparison: Compare trial’s N-Reduction vs regional averages.
2. Tissue Nutrient Analysis: Identify nutrients as high, near target, or deficient.
3. Sales & Agronomic Takeaway: Frame results for sales interpretation.
'''
# ---------------------------------------------------------------------------
# RUN PIPELINE
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    run_pipeline(
        page_name=PAGE_TO_PROCESS,
        base_data_path=BASE_DATA_PATH,
        llm_endpoint_name=LLM_ENDPOINT_NAME,
        specific_trial_data=SPECIFIC_TRIAL_DATA,
        llm_prompt_template=LLM_COMPARISON_PROMPT
    )


# COMMAND ----------

# MAGIC %md
# MAGIC ### Page 5

# COMMAND ----------

import json
import os
import mlflow.deployments
from pyspark.sql import SparkSession
from pyspark.dbutils import DBUtils

# ---------------------------------------------------------------------------
# CONFIGURATION: YOU MUST EDIT THIS SECTION
# ---------------------------------------------------------------------------
# 1. SET THE PAGE TO PROCESS
#    (e.g., "page1", "page3", "page4")
PAGE_TO_PROCESS = "page5"

# 2. SET THE BASE PATH TO YOUR "pages" FOLDER
#    Example: "/Workspace/Users/your.email@company.com/LLM_Project/pages"
BASE_CONFIG_PATH = "/Workspace/Users/ayush.dongardive@pivotbio.com/LLM Pipeline/pages"

# 3. SET THE DATABRICKS MODEL SERVING ENDPOINT NAME
LLM_ENDPOINT_NAME = "databricks-llama-4-maverick"

# 4. PASTE THE HARDCODED DATA FOR THE PAGE HERE
#    This dictionary contains the data that would have come from the SQL query.
#    This example is pre-filled with the data for Page 1.
PAGE_DATA = {
  "Trial ID": "25-ZEAMX-US550-MN01-GETT",
  "N_Reduction_Product_Reduced": "NH3: 82-0-0 (Anhydrous Ammonia)",
  "NTC: 100%N_N_Total_lb_ac": 170,
  "NTC: 100%N_N_Diff_from_Full": 0,
  "NTC: 100%N_N_Percent_Diff": 0,
  "NTC: 100%N_P_Total_lb_ac": 86,
  "NTC: 100%N_K_Total_lb_ac": 80,
  "NTC: 100%N_S_Total_lb_ac": 15,
  "NTC: 100%N -BMP_N_Total_lb_ac": 130,
  "NTC: 100%N -BMP_N_Diff_from_Full": -40,
  "NTC: 100%N -BMP_N_Percent_Diff": -24,
  "NTC: 100%N -BMP_P_Total_lb_ac": 86,
  "NTC: 100%N -BMP_K_Total_lb_ac": 80,
  "NTC: 100%N -BMP_S_Total_lb_ac": 15,
  "PBDryB: 100%N -BMP_N_Total_lb_ac": 130,
  "PBDryB: 100%N -BMP_N_Diff_from_Full": -40,
  "PBDryB: 100%N -BMP_N_Percent_Diff": -24,
  "PBDryB: 100%N -BMP_P_Total_lb_ac": 86,
  "PBDryB: 100%N -BMP_K_Total_lb_ac": 80,
  "PBDryB: 100%N -BMP_S_Total_lb_ac": 15,
  "PBDryS: 100%N -BMP_N_Total_lb_ac": 130,
  "PBDryS: 100%N -BMP_N_Diff_from_Full": -40,
  "PBDryS: 100%N -BMP_N_Percent_Diff": -24,
  "PBDryS: 100%N -BMP_P_Total_lb_ac": 86,
  "PBDryS: 100%N -BMP_K_Total_lb_ac": 80,
  "PBDryS: 100%N -BMP_S_Total_lb_ac": 15,
  "PBOS: 100%N_N_Total_lb_ac": 170,
  "PBOS: 100%N_N_Diff_from_Full": 0,
  "PBOS: 100%N_N_Percent_Diff": 0,
  "PBOS: 100%N_P_Total_lb_ac": 86,
  "PBOS: 100%N_K_Total_lb_ac": 80,
  "PBOS: 100%N_S_Total_lb_ac": 15,
  "PBOS: 100%N -BMP_N_Total_lb_ac": 130,
  "PBOS: 100%N -BMP_N_Diff_from_Full": -40,
  "PBOS: 100%N -BMP_N_Percent_Diff": -24,
  "PBOS: 100%N -BMP_P_Total_lb_ac": 86,
  "PBOS: 100%N -BMP_K_Total_lb_ac": 80,
  "PBOS: 100%N -BMP_S_Total_lb_ac": 15,
  "Max_NH3_N_lb_ac_applied": 90,
  "Min_NH3_N_lb_ac_applied": 50,
  "NH3_N_lb_ac_Reduction": -40,
  "Max_DAP_N_lb_ac_applied": 18,
  "Max_MicroEssentials_N_lb_ac_applied": 12,
  "Max_Urea_N_lb_ac_applied": 46,
  "Max_AMS_N_lb_ac_applied": 4
  
}
# ---------------------------------------------------------------------------

# Initialize Spark Session and DBUtils (standard for Databricks notebooks)
spark = SparkSession.builder.appName("SummarizationPipeline").getOrCreate()
dbutils = DBUtils(spark)

def load_config_files(page_folder_path, page_name):
    """
    Loads all necessary JSON configuration files for a given page.
    It gracefully handles the absence of the optional metrics file.
    """
    print("Loading configuration files...")
    try:
        # Define paths for all potential files
        prompt_path = os.path.join(page_folder_path, f"{page_name}_prompt.json")
        metadata_path = os.path.join(page_folder_path, f"{page_name}_column_metadata.json")
        metrics_path = os.path.join(page_folder_path, f"{page_name}_metrics.json")

        # Load mandatory files
        with open(prompt_path, 'r') as f:
            prompt_template = json.load(f)
        with open(metadata_path, 'r') as f:
            column_metadata = json.load(f)

        # Load optional metrics file
        metrics_thresholds = None
        if os.path.exists(metrics_path):
            with open(metrics_path, 'r') as f:
                metrics_thresholds = json.load(f)
                print(f"INFO: Successfully loaded optional metrics file from {metrics_path}")
        else:
            print(f"INFO: Optional file not found at {metrics_path}. Proceeding without it.")

        return prompt_template, column_metadata, metrics_thresholds
    except FileNotFoundError as e:
        print(f"ERROR: A mandatory configuration file was not found. Details: {e}")
        return None, None, None
    except json.JSONDecodeError as e:
        print(f"ERROR: Failed to parse a JSON configuration file. Please check for syntax errors. Details: {e}")
        return None, None, None

def assemble_llm_payload(prompt_template, page_data, column_metadata, metrics_thresholds):
    """
    Assembles the final JSON payload to be sent to the LLM.
    """
    print("Assembling the final LLM payload...")
    try:
        # Start with a deep copy of the template to avoid modifying the original
        final_payload = json.loads(json.dumps(prompt_template))

        # Navigate to the instruction block where data will be injected
        instruction_block = final_payload['instructions_and_output_structure']
        input_data_section = instruction_block['input_data']

        # Inject the hardcoded page data and metadata
        input_data_section['page_data'] = page_data
        input_data_section['column_metadata'] = column_metadata.get('column_metadata', [])

        # Inject metrics thresholds if they exist
        if metrics_thresholds:
            input_data_section['key_metrics_threshold_info'] = metrics_thresholds
        
        print("Payload assembled successfully.")
        return final_payload
    except KeyError as e:
        print(f"ERROR: The prompt template is missing a required key. Could not find key: {e}")
        return None

def call_llm(payload_dict, endpoint_name):
    """
    Calls the Databricks Model Serving endpoint using the mlflow.deployments client.
    """
    print("\n" + "="*70)
    print(f"Calling LLM Endpoint: {endpoint_name}...")

    try:
        # Convert the entire payload dictionary into a single JSON string for the prompt
        prompt = json.dumps(payload_dict, indent=2)

        # Initialize the deployment client
        client = mlflow.deployments.get_deploy_client("databricks")
        
        # Structure the final payload for the Databricks LLM serving API
        inputs_payload = {
            "messages": [
                {"role": "user", "content": prompt}
            ]
        }
        
        # Call the LLM deployment endpoint
        response = client.predict(
            endpoint=endpoint_name, 
            inputs=inputs_payload
        )
        
        # Extract the summary text from the response structure
        summary = response.get('choices', [{}])[0].get('message', {}).get('content', "Error: Could not extract content from LLM response.")
        
        # Clean up the summary (e.g., strip surrounding whitespace)
        summary = summary.strip()
        
        print("✅ LLM Summary Generated Successfully.")
        return summary

    except Exception as e:
        # Handle any connection or prediction errors gracefully
        error_message = f"Error generating summary from LLM endpoint: {str(e)}"
        print(f"❌ {error_message}")
        return error_message

def main():
    """
    Main function to orchestrate the summarization pipeline.
    """
    print(f"--- Starting Summary Generation for Page '{PAGE_TO_PROCESS}' ---")

    # Construct the full path to the page's configuration folder
    page_folder = os.path.join(BASE_CONFIG_PATH, PAGE_TO_PROCESS)

    # 1. Load all configuration files
    prompt_template, column_metadata, metrics_thresholds = load_config_files(page_folder, PAGE_TO_PROCESS)
    if not prompt_template:
        print("ERROR: Failed to load configuration. Aborting.")
        return

    # 2. Use the hardcoded page data
    page_data = PAGE_DATA
    print("Using hardcoded page data.")
    
    # 3. Assemble the final payload for the LLM
    llm_payload = assemble_llm_payload(prompt_template, page_data, column_metadata, metrics_thresholds)
    if not llm_payload:
        print("ERROR: Failed to assemble LLM payload. Aborting.")
        return

    # 4. Call the LLM to get the summary
    summary = call_llm(llm_payload, LLM_ENDPOINT_NAME)
    if not summary:
        print("ERROR: Failed to generate summary. Aborting.")
        return
        
    # 5. Display the final summary
    print("\n--- FINAL SUMMARY PREVIEW ---")
    print(summary)
    print("---------------------------\n")

# Run the main pipeline function
if __name__ == "__main__":
    main()
    

# COMMAND ----------

# MAGIC %md
# MAGIC Comparative Summary

# COMMAND ----------

import json
import os
import mlflow.deployments
from pyspark.sql import SparkSession
# Initialize Spark session (Databricks automatically provides dbutils)
spark = SparkSession.builder.appName("SummarizationPipeline").getOrCreate()
# ---------------------------------------------------------------------------
# GLOBAL LLM INTERACTION FUNCTIONS (Reusable across all pipelines)
# ---------------------------------------------------------------------------
def load_comparative_metrics(base_path, page_name):
    """
    Loads the JSON file containing the regional comparative metrics.
    """
    print("Loading comparative data file...")
    comparative_path = os.path.join(base_path, f"{page_name}_comparative_metrics.json")

    try:
        with open(comparative_path, "r") as f:
            comparative_metrics = json.load(f)
            print(f"INFO: Successfully loaded comparative metrics from {comparative_path}")
            return comparative_metrics
    except FileNotFoundError:
        print(f"ERROR: Comparative metrics file not found at {comparative_path}. Aborting.")
        return None
    except json.JSONDecodeError as e:
        print(f"ERROR: Failed to parse comparative metrics JSON. Details: {e}")
        return None
def assemble_llm_payload(prompt_template, trial_data, comparative_data):
    """
    Assembles the final LLM prompt string by injecting trial and comparative data.
    """
    print("Assembling the final LLM payload...")
    trial_data_str = json.dumps(trial_data, indent=2)
    comparative_data_str = json.dumps(comparative_data, indent=2)
    final_prompt = prompt_template.format(
        trial_data=trial_data_str,
        comparative_data=comparative_data_str
    )
    llm_payload = {"messages": [{"role": "user", "content": final_prompt}]}
    print("Payload assembled successfully.")
    return llm_payload
def call_llm(payload_dict, endpoint_name):
    """
    Calls the Databricks Model Serving endpoint.
    """
    print("\n" + "=" * 70)
    print(f"Calling LLM Endpoint: {endpoint_name}...")
    try:
        client = mlflow.deployments.get_deploy_client("databricks")
        # Depending on the model type, you may need to use 'inputs' instead of 'messages'
        response = client.predict(endpoint=endpoint_name, inputs=payload_dict)
        summary = (
            response.get("choices", [{}])[0]
            .get("message", {})
            .get("content", "Error: Could not extract content from LLM response.")
            .strip()
        )
        print(":white_check_mark: LLM Summary Generated Successfully.")
        return summary
    except Exception as e:
        error_message = f"Error generating summary from LLM endpoint: {str(e)}"
        print(f":x: {error_message}")
        return error_message
# ---------------------------------------------------------------------------
# MAIN PIPELINE EXECUTION FUNCTION
# ---------------------------------------------------------------------------
def run_pipeline(page_name, base_data_path, llm_endpoint_name, specific_trial_data, llm_prompt_template):
    """
    Orchestrates the LLM summarization pipeline for a single trial.
    """
    print(f"--- Starting Summary Generation for Page '{page_name}' ---")
    comparative_data = load_comparative_metrics(base_data_path, page_name)
    if not comparative_data:
        print("ERROR: Failed to load comparative data. Aborting.")
        return
    llm_payload = assemble_llm_payload(llm_prompt_template, specific_trial_data, comparative_data)
    summary = call_llm(llm_payload, llm_endpoint_name)
    if summary and not summary.startswith("Error"):
        print("\n--- FINAL SUMMARY PREVIEW ---")
        print(summary)
        print("---------------------------\n")
    else:
        print("ERROR: Failed to generate summary. Aborting.")
    return summary
# ---------------------------------------------------------------------------
# CONFIGURATION SECTION
# ---------------------------------------------------------------------------
PAGE_TO_PROCESS = "page5"
BASE_DATA_PATH = "/Workspace/Users/ayush.dongardive@pivotbio.com/LLM Pipeline/Comparitive Metric"
LLM_ENDPOINT_NAME = "databricks-llama-4-maverick"
SPECIFIC_TRIAL_DATA = {
    "Trial ID": "25-ZEAMX-US550-MN01-GETT",
  "N_Reduction_Product_Reduced": "NH3: 82-0-0 (Anhydrous Ammonia)",
  "NTC: 100%N_N_Total_lb_ac": 170,
  "NTC: 100%N_N_Diff_from_Full": 0,
  "NTC: 100%N_N_Percent_Diff": 0,
  "NTC: 100%N_P_Total_lb_ac": 86,
  "NTC: 100%N_K_Total_lb_ac": 80,
  "NTC: 100%N_S_Total_lb_ac": 15,
  "NTC: 100%N -BMP_N_Total_lb_ac": 130,
  "NTC: 100%N -BMP_N_Diff_from_Full": -40,
  "NTC: 100%N -BMP_N_Percent_Diff": -24,
  "NTC: 100%N -BMP_P_Total_lb_ac": 86,
  "NTC: 100%N -BMP_K_Total_lb_ac": 80,
  "NTC: 100%N -BMP_S_Total_lb_ac": 15,
  "PBDryB: 100%N -BMP_N_Total_lb_ac": 130,
  "PBDryB: 100%N -BMP_N_Diff_from_Full": -40,
  "PBDryB: 100%N -BMP_N_Percent_Diff": -24,
  "PBDryB: 100%N -BMP_P_Total_lb_ac": 86,
  "PBDryB: 100%N -BMP_K_Total_lb_ac": 80,
  "PBDryB: 100%N -BMP_S_Total_lb_ac": 15,
  "PBDryS: 100%N -BMP_N_Total_lb_ac": 130,
  "PBDryS: 100%N -BMP_N_Diff_from_Full": -40,
  "PBDryS: 100%N -BMP_N_Percent_Diff": -24,
  "PBDryS: 100%N -BMP_P_Total_lb_ac": 86,
  "PBDryS: 100%N -BMP_K_Total_lb_ac": 80,
  "PBDryS: 100%N -BMP_S_Total_lb_ac": 15,
  "PBOS: 100%N_N_Total_lb_ac": 170,
  "PBOS: 100%N_N_Diff_from_Full": 0,
  "PBOS: 100%N_N_Percent_Diff": 0,
  "PBOS: 100%N_P_Total_lb_ac": 86,
  "PBOS: 100%N_K_Total_lb_ac": 80,
  "PBOS: 100%N_S_Total_lb_ac": 15,
  "PBOS: 100%N -BMP_N_Total_lb_ac": 130,
  "PBOS: 100%N -BMP_N_Diff_from_Full": -40,
  "PBOS: 100%N -BMP_N_Percent_Diff": -24,
  "PBOS: 100%N -BMP_P_Total_lb_ac": 86,
  "PBOS: 100%N -BMP_K_Total_lb_ac": 80,
  "PBOS: 100%N -BMP_S_Total_lb_ac": 15,
  "Max_NH3_N_lb_ac_applied": 90,
  "Min_NH3_N_lb_ac_applied": 50,
  "NH3_N_lb_ac_Reduction": -40,
  "Max_DAP_N_lb_ac_applied": 18,
  "Max_MicroEssentials_N_lb_ac_applied": 12,
  "Max_Urea_N_lb_ac_applied": 46,
  "Max_AMS_N_lb_ac_applied": 4
}
# Use triple single quotes to avoid conflicts with double quotes in JSON-style text
LLM_COMPARISON_PROMPT = '''
You are an Analytical AI Assistant for the Pivot Bio Sales and Agronomy Team.
You specialize in comparing trial setup and final nutrient programs against regional benchmarks (State Average data).
Company Context:
Pivot Bio is a U.S.-based agricultural biotechnology company that develops microbe-based nitrogen solutions to replace synthetic fertilizers.
Their flagship product, Pivot Bio PROVEN® 40, replaces synthetic nitrogen, improving yield consistency and environmental outcomes.
They run field-scale validation trials across the U.S. to demonstrate nitrogen-reduction performance.
Task:
Analyze the Fertilizer Application data to detail the trial's nutrient program, the specifics of the nitrogen reduction, and its sales relevance. All necessary quantitative values must be drawn directly from the provided 'page_data' and 'Regional_Comparative_Metrics' in the input.
Input Data:
- Specific Trial Data: {trial_data}
- Comparative Metrics: {comparative_data}
Output:
"title": "Ground Truth: Trial vs. Regional N-Strategy",
"instruction": "Summarize the trial's N-strategy (Full N rate, Reduced N rate, and percentage reduction) and compare it directly to the regional averages (Avg_Full_N_Rate_GSP_lb_ac, Avg_N_Reduction_lb_ac, Avg_N_Reduction_Percent). State the exact N reduction amount (in lbs/ac) and its percentage for the trial, and identify the primary N fertilizer product (Anhydrous Ammonia) and the application timings (Fall, Spring) for complete context."
"title": "Technical Understanding: Isolation and Baseline Comparison",
"instruction": "Explain what the data demonstrates technically. State the total lbs/ac applied for the P, K, and S baselines for the trial, and compare these totals (P, K, S) against the regional averages. Emphasize that the trial's uniformity successfully isolates the nitrogen reduction as the key variable being tested, making any subsequent performance differences directly attributable to the nitrogen source."
"title": "Sales Takeaway: Positioning the Trial",
"instruction": "Provide a business insight. State whether the trial's N reduction is more or less aggressive than the regional average, and confirm that the consistent baseline for other nutrients ensures a fair, apples-to-apples comparison. Conclude with a strong statement: the trial's aggressive reduction (40 lb/ac) is a strong validation point for the PROVEN® 40 value proposition, showing the grower's confidence in microbial nitrogen."
"tone": "Analytical, factual, and business-aligned.",
"length": "180-220 words."
'''
# ---------------------------------------------------------------------------
# RUN PIPELINE
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    run_pipeline(
        page_name=PAGE_TO_PROCESS,
        base_data_path=BASE_DATA_PATH,
        llm_endpoint_name=LLM_ENDPOINT_NAME,
        specific_trial_data=SPECIFIC_TRIAL_DATA,
        llm_prompt_template=LLM_COMPARISON_PROMPT
    )


# COMMAND ----------

# MAGIC %md
# MAGIC ### Page 6

# COMMAND ----------

import json
import os
import mlflow.deployments
from pyspark.sql import SparkSession
from pyspark.dbutils import DBUtils

# ---------------------------------------------------------------------------
# CONFIGURATION: YOU MUST EDIT THIS SECTION
# ---------------------------------------------------------------------------
# 1. SET THE PAGE TO PROCESS
#    (e.g., "page1", "page3", "page4")
PAGE_TO_PROCESS = "page6"

# 2. SET THE BASE PATH TO YOUR "pages" FOLDER
#    Example: "/Workspace/Users/your.email@company.com/LLM_Project/pages"
BASE_CONFIG_PATH = "/Workspace/Users/ayush.dongardive@pivotbio.com/LLM Pipeline/pages"

# 3. SET THE DATABRICKS MODEL SERVING ENDPOINT NAME
LLM_ENDPOINT_NAME = "databricks-llama-4-maverick"

# 4. PASTE THE HARDCODED DATA FOR THE PAGE HERE
#    This dictionary contains the data that would have come from the SQL query.
#    This example is pre-filled with the data for Page 1.
PAGE_DATA = {

  
  "Trial ID": "25-ZEAMX-US550-MN01-GETT",
  "Lab": "Next Level Ag, LLC",
  "Analysis Package": "Indicator Completa",
  "Results Received": "June 3, 2025",
  "Sample Date": "May 30, 2025",
  "Sample Depth": "0\"-6\"",
  "Number of Samples": 4,
  "Soil_Map_Canisteo_clay_loam": "0 to 2 percent slopes",
  "Soil_Map_Clarion_loam": "2 to 5 percent slopes",
  "Soil_Map_Nicollet_clay_loam": "1 to 3 percent slopes",
  "Soil_Map_Webster_clay_loam": "0 to 2 percent slopes",
  "Sample_1_ID": "033125-2182428",
  "Sample_1_Total_Nitrate_lbs": 58.1,
  "Sample_1_WEO_C_lbs": 184.3,
  "Sample_1_Org_N_WEO_N_lbs": 51.8,
  "Sample_1_C_N_Ratio": 6.4,
  "Sample_2_ID": "033125-46164177",
  "Sample_2_Total_Nitrate_lbs": 58.5,
  "Sample_2_WEO_C_lbs": 184.6,
  "Sample_2_Org_N_WEO_N_lbs": 60.5,
  "Sample_2_C_N_Ratio": 5.5,
  "Sample_3_ID": "033125-5815105",
  "Sample_3_Total_Nitrate_lbs": 63.9,
  "Sample_3_WEO_C_lbs": 139.9,
  "Sample_3_Org_N_WEO_N_lbs": 60.7,
  "Sample_3_C_N_Ratio": 4.2,
  "Sample_4_ID": "033125-8655163",
  "Sample_4_Total_Nitrate_lbs": 57.8,
  "Sample_4_WEO_C_lbs": 181.6,
  "Sample_4_Org_N_WEO_N_lbs": 67.6,
  "Sample_4_C_N_Ratio": 4.8

  
}
# ---------------------------------------------------------------------------

# Initialize Spark Session and DBUtils (standard for Databricks notebooks)
spark = SparkSession.builder.appName("SummarizationPipeline").getOrCreate()
dbutils = DBUtils(spark)

def load_config_files(page_folder_path, page_name):
    """
    Loads all necessary JSON configuration files for a given page.
    It gracefully handles the absence of the optional metrics file.
    """
    print("Loading configuration files...")
    try:
        # Define paths for all potential files
        prompt_path = os.path.join(page_folder_path, f"{page_name}_prompt.json")
        metadata_path = os.path.join(page_folder_path, f"{page_name}_column_metadata.json")
        metrics_path = os.path.join(page_folder_path, f"{page_name}_metrics_threshold.json")

        # Load mandatory files
        with open(prompt_path, 'r') as f:
            prompt_template = json.load(f)
        with open(metadata_path, 'r') as f:
            column_metadata = json.load(f)

        # Load optional metrics file
        metrics_thresholds = None
        if os.path.exists(metrics_path):
            with open(metrics_path, 'r') as f:
                metrics_thresholds = json.load(f)
                print(f"INFO: Successfully loaded optional metrics file from {metrics_path}")
        else:
            print(f"INFO: Optional file not found at {metrics_path}. Proceeding without it.")

        return prompt_template, column_metadata, metrics_thresholds
    except FileNotFoundError as e:
        print(f"ERROR: A mandatory configuration file was not found. Details: {e}")
        return None, None, None
    except json.JSONDecodeError as e:
        print(f"ERROR: Failed to parse a JSON configuration file. Please check for syntax errors. Details: {e}")
        return None, None, None

def assemble_llm_payload(prompt_template, page_data, column_metadata, metrics_thresholds):
    """
    Assembles the final JSON payload to be sent to the LLM.
    """
    print("Assembling the final LLM payload...")
    try:
        # Start with a deep copy of the template to avoid modifying the original
        final_payload = json.loads(json.dumps(prompt_template))

        # Navigate to the instruction block where data will be injected
        instruction_block = final_payload['instructions_and_output_structure']
        input_data_section = instruction_block['input_data']

        # Inject the hardcoded page data and metadata
        input_data_section['page_data'] = page_data
        input_data_section['column_metadata'] = column_metadata.get('column_metadata', [])

        # Inject metrics thresholds if they exist
        if metrics_thresholds:
            input_data_section['key_metrics_threshold_info'] = metrics_thresholds
        
        print("Payload assembled successfully.")
        return final_payload
    except KeyError as e:
        print(f"ERROR: The prompt template is missing a required key. Could not find key: {e}")
        return None

def call_llm(payload_dict, endpoint_name):
    """
    Calls the Databricks Model Serving endpoint using the mlflow.deployments client.
    """
    print("\n" + "="*70)
    print(f"Calling LLM Endpoint: {endpoint_name}...")

    try:
        # Convert the entire payload dictionary into a single JSON string for the prompt
        prompt = json.dumps(payload_dict, indent=2)

        # Initialize the deployment client
        client = mlflow.deployments.get_deploy_client("databricks")
        
        # Structure the final payload for the Databricks LLM serving API
        inputs_payload = {
            "messages": [
                {"role": "user", "content": prompt}
            ]
        }
        
        # Call the LLM deployment endpoint
        response = client.predict(
            endpoint=endpoint_name, 
            inputs=inputs_payload
        )
        
        # Extract the summary text from the response structure
        summary = response.get('choices', [{}])[0].get('message', {}).get('content', "Error: Could not extract content from LLM response.")
        
        # Clean up the summary (e.g., strip surrounding whitespace)
        summary = summary.strip()
        
        print("✅ LLM Summary Generated Successfully.")
        return summary

    except Exception as e:
        # Handle any connection or prediction errors gracefully
        error_message = f"Error generating summary from LLM endpoint: {str(e)}"
        print(f"❌ {error_message}")
        return error_message

def main():
    """
    Main function to orchestrate the summarization pipeline.
    """
    print(f"--- Starting Summary Generation for Page '{PAGE_TO_PROCESS}' ---")

    # Construct the full path to the page's configuration folder
    page_folder = os.path.join(BASE_CONFIG_PATH, PAGE_TO_PROCESS)

    # 1. Load all configuration files
    prompt_template, column_metadata, metrics_thresholds = load_config_files(page_folder, PAGE_TO_PROCESS)
    if not prompt_template:
        print("ERROR: Failed to load configuration. Aborting.")
        return

    # 2. Use the hardcoded page data
    page_data = PAGE_DATA
    print("Using hardcoded page data.")
    
    # 3. Assemble the final payload for the LLM
    llm_payload = assemble_llm_payload(prompt_template, page_data, column_metadata, metrics_thresholds)
    if not llm_payload:
        print("ERROR: Failed to assemble LLM payload. Aborting.")
        return

    # 4. Call the LLM to get the summary
    summary = call_llm(llm_payload, LLM_ENDPOINT_NAME)
    if not summary:
        print("ERROR: Failed to generate summary. Aborting.")
        return
        
    # 5. Display the final summary
    print("\n--- FINAL SUMMARY PREVIEW ---")
    print(summary)
    print("---------------------------\n")

# Run the main pipeline function
if __name__ == "__main__":
    main()
    

# COMMAND ----------

# MAGIC %md
# MAGIC Comparative Summary

# COMMAND ----------

import json
import os
import mlflow.deployments
from pyspark.sql import SparkSession
# Initialize Spark session (Databricks automatically provides dbutils)
spark = SparkSession.builder.appName("SummarizationPipeline").getOrCreate()
# ---------------------------------------------------------------------------
# GLOBAL LLM INTERACTION FUNCTIONS (Reusable across all pipelines)
# ---------------------------------------------------------------------------
def load_comparative_metrics(base_path, page_name):
    """
    Loads the JSON file containing the regional comparative metrics.
    """
    print("Loading comparative data file...")
    comparative_path = os.path.join(base_path, f"{page_name}_comparative_metrics.json")

    try:
        with open(comparative_path, "r") as f:
            comparative_metrics = json.load(f)
            print(f"INFO: Successfully loaded comparative metrics from {comparative_path}")
            return comparative_metrics
    except FileNotFoundError:
        print(f"ERROR: Comparative metrics file not found at {comparative_path}. Aborting.")
        return None
    except json.JSONDecodeError as e:
        print(f"ERROR: Failed to parse comparative metrics JSON. Details: {e}")
        return None
def assemble_llm_payload(prompt_template, trial_data, comparative_data):
    """
    Assembles the final LLM prompt string by injecting trial and comparative data.
    """
    print("Assembling the final LLM payload...")
    trial_data_str = json.dumps(trial_data, indent=2)
    comparative_data_str = json.dumps(comparative_data, indent=2)
    final_prompt = prompt_template.format(
        trial_data=trial_data_str,
        comparative_data=comparative_data_str
    )
    llm_payload = {"messages": [{"role": "user", "content": final_prompt}]}
    print("Payload assembled successfully.")
    return llm_payload
def call_llm(payload_dict, endpoint_name):
    """
    Calls the Databricks Model Serving endpoint.
    """
    print("\n" + "=" * 70)
    print(f"Calling LLM Endpoint: {endpoint_name}...")
    try:
        client = mlflow.deployments.get_deploy_client("databricks")
        # Depending on the model type, you may need to use 'inputs' instead of 'messages'
        response = client.predict(endpoint=endpoint_name, inputs=payload_dict)
        summary = (
            response.get("choices", [{}])[0]
            .get("message", {})
            .get("content", "Error: Could not extract content from LLM response.")
            .strip()
        )
        print(":white_check_mark: LLM Summary Generated Successfully.")
        return summary
    except Exception as e:
        error_message = f"Error generating summary from LLM endpoint: {str(e)}"
        print(f":x: {error_message}")
        return error_message
# ---------------------------------------------------------------------------
# MAIN PIPELINE EXECUTION FUNCTION
# ---------------------------------------------------------------------------
def run_pipeline(page_name, base_data_path, llm_endpoint_name, specific_trial_data, llm_prompt_template):
    """
    Orchestrates the LLM summarization pipeline for a single trial.
    """
    print(f"--- Starting Summary Generation for Page '{page_name}' ---")
    comparative_data = load_comparative_metrics(base_data_path, page_name)
    if not comparative_data:
        print("ERROR: Failed to load comparative data. Aborting.")
        return
    llm_payload = assemble_llm_payload(llm_prompt_template, specific_trial_data, comparative_data)
    summary = call_llm(llm_payload, llm_endpoint_name)
    if summary and not summary.startswith("Error"):
        print("\n--- FINAL SUMMARY PREVIEW ---")
        print(summary)
        print("---------------------------\n")
    else:
        print("ERROR: Failed to generate summary. Aborting.")
    return summary
# ---------------------------------------------------------------------------
# CONFIGURATION SECTION
# ---------------------------------------------------------------------------
PAGE_TO_PROCESS = "page6"
BASE_DATA_PATH = "/Workspace/Users/ayush.dongardive@pivotbio.com/LLM Pipeline/Comparitive Metric"
LLM_ENDPOINT_NAME = "databricks-llama-4-maverick"
SPECIFIC_TRIAL_DATA = {
 "Number of Samples": 4,
  "Soil_Map_Canisteo_clay_loam": "0 to 2 percent slopes",
  "Soil_Map_Clarion_loam": "2 to 5 percent slopes",
  "Soil_Map_Nicollet_clay_loam": "1 to 3 percent slopes",
  "Soil_Map_Webster_clay_loam": "0 to 2 percent slopes",
  "Sample_1_ID": "033125-2182428",
  "Sample_1_Total_Nitrate_lbs": 58.1,
  "Sample_1_WEO_C_lbs": 184.3,
  "Sample_1_Org_N_WEO_N_lbs": 51.8,
  "Sample_1_C_N_Ratio": 6.4,
  "Sample_2_ID": "033125-46164177",
  "Sample_2_Total_Nitrate_lbs": 58.5,
  "Sample_2_WEO_C_lbs": 184.6,
  "Sample_2_Org_N_WEO_N_lbs": 60.5,
  "Sample_2_C_N_Ratio": 5.5,
  "Sample_3_ID": "033125-5815105",
  "Sample_3_Total_Nitrate_lbs": 63.9,
  "Sample_3_WEO_C_lbs": 139.9,
  "Sample_3_Org_N_WEO_N_lbs": 60.7,
  "Sample_3_C_N_Ratio": 4.2,
  "Sample_4_ID": "033125-8655163",
  "Sample_4_Total_Nitrate_lbs": 57.8,
  "Sample_4_WEO_C_lbs": 181.6,
  "Sample_4_Org_N_WEO_N_lbs": 67.6,
  "Sample_4_C_N_Ratio": 4.8

  
}
# Use triple single quotes to avoid conflicts with double quotes in JSON-style text
LLM_COMPARISON_PROMPT = '''
You are an Analytical AI Assistant for the Pivot Bio Sales and Agronomy Team.
You specialize in comparing **early season soil health and nutrient availability** against regional benchmarks (State Average data) and analyzing internal field variability.
Company Context:
Pivot Bio is a U.S.-based agricultural biotechnology company that develops microbe-based nitrogen solutions to replace synthetic fertilizers.
Their flagship product, Pivot Bio PROVEN® 40, replaces synthetic nitrogen, improving yield consistency and environmental outcomes.
They run field-scale validation trials across the U.S. to demonstrate nitrogen-reduction performance.
Task:
Analyze the provided Early Season Soil Analysis Data, focusing on the available Nitrogen (Total Nitrate) and Carbon dynamics (WEO C, C:N Ratio). Compare the trial's average soil characteristics against the regional averages (in 'comparative_metrics') and quantify the internal variability between the individual samples.
Input Data:
- Specific Trial Data: {trial_data}
- Comparative Metrics: {comparative_data}
Output:
"title": "Regional Comparison: Available N and Carbon Dynamics",
"instruction": "Calculate the average Total Nitrate and WEO C for the four specific samples (1-4) in the specific trial data. Compare these trial averages directly against the regional averages (Regional_Avg_Total_Nitrate_lbs and Regional_Avg_WEO_C_lbs). State whether the trial site has a higher or lower early-season N and organic carbon base than the region, and by what quantified margin (lbs)."
"title": "Internal Field Variability Analysis",
"instruction": "Assess the uniformity of the field by quantifying the difference between the single highest and single lowest reading for Total Nitrate (lbs) and C:N Ratio (unitless) across the four individual samples. Use the C:N Ratio variability to draw a conclusion on the uniformity of N immobilization risk across the field."
"title": "Sales & Agronomic Takeaway",
"instruction": "Deliver a 2-3 sentence business insight. Combine the findings: Frame the trial's Total Nitrate level relative to the regional average (e.g., 'high residual N') to affirm the grower's confidence in reducing synthetic N. Then, use the C:N Ratio variability insight to suggest that the grower could benefit from variable rate application of Pivot Bio, but that the overall uniformity is adequate for a successful plot trial."
"tone": "Analytical, comparative, and highly focused on business impact.",
"length": "180-250 words."
'''
# ---------------------------------------------------------------------------
# RUN PIPELINE
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    run_pipeline(
        page_name=PAGE_TO_PROCESS,
        base_data_path=BASE_DATA_PATH,
        llm_endpoint_name=LLM_ENDPOINT_NAME,
        specific_trial_data=SPECIFIC_TRIAL_DATA,
        llm_prompt_template=LLM_COMPARISON_PROMPT
    )


# COMMAND ----------

# MAGIC %md
# MAGIC ### Page 7
# MAGIC ### 

# COMMAND ----------

import json
import os
import mlflow.deployments
from pyspark.sql import SparkSession
from pyspark.dbutils import DBUtils

# ---------------------------------------------------------------------------
# CONFIGURATION: YOU MUST EDIT THIS SECTION
# ---------------------------------------------------------------------------
# 1. SET THE PAGE TO PROCESS
#    (e.g., "page1", "page3", "page4")
PAGE_TO_PROCESS = "page7"

# 2. SET THE BASE PATH TO YOUR "pages" FOLDER
#    Example: "/Workspace/Users/your.email@company.com/LLM_Project/pages"
BASE_CONFIG_PATH = "/Workspace/Users/ayush.dongardive@pivotbio.com/LLM Pipeline/pages"

# 3. SET THE DATABRICKS MODEL SERVING ENDPOINT NAME
LLM_ENDPOINT_NAME = "databricks-llama-4-maverick"

# 4. PASTE THE HARDCODED DATA FOR THE PAGE HERE
#    This dictionary contains the data that would have come from the SQL query.
#    This example is pre-filled with the data for Page 1.
PAGE_DATA = {

  "NTC: 100%N": "30889",
  "NTC: 100%N -BMP": "31264",
  "PBDryB: 100%N -BMP": "30952",
  "PBDryS: 100%N -BMP": "30451",
  "PBOS: 100%N": "31139",
  "PBOS: 100%N -BMP": "31202"
  
}
# ---------------------------------------------------------------------------

# Initialize Spark Session and DBUtils (standard for Databricks notebooks)
spark = SparkSession.builder.appName("SummarizationPipeline").getOrCreate()
dbutils = DBUtils(spark)

def load_config_files(page_folder_path, page_name):
    """
    Loads all necessary JSON configuration files for a given page.
    It gracefully handles the absence of the optional metrics file.
    """
    print("Loading configuration files...")
    try:
        # Define paths for all potential files
        prompt_path = os.path.join(page_folder_path, f"{page_name}_prompt.json")
        metadata_path = os.path.join(page_folder_path, f"{page_name}_column_metadata.json")
        metrics_path = os.path.join(page_folder_path, f"{page_name}_metrics_threshold.json")

        # Load mandatory files
        with open(prompt_path, 'r') as f:
            prompt_template = json.load(f)
        with open(metadata_path, 'r') as f:
            column_metadata = json.load(f)

        # Load optional metrics file
        metrics_thresholds = None
        if os.path.exists(metrics_path):
            with open(metrics_path, 'r') as f:
                metrics_thresholds = json.load(f)
                print(f"INFO: Successfully loaded optional metrics file from {metrics_path}")
        else:
            print(f"INFO: Optional file not found at {metrics_path}. Proceeding without it.")

        return prompt_template, column_metadata, metrics_thresholds
    except FileNotFoundError as e:
        print(f"ERROR: A mandatory configuration file was not found. Details: {e}")
        return None, None, None
    except json.JSONDecodeError as e:
        print(f"ERROR: Failed to parse a JSON configuration file. Please check for syntax errors. Details: {e}")
        return None, None, None

def assemble_llm_payload(prompt_template, page_data, column_metadata, metrics_thresholds):
    """
    Assembles the final JSON payload to be sent to the LLM.
    """
    print("Assembling the final LLM payload...")
    try:
        # Start with a deep copy of the template to avoid modifying the original
        final_payload = json.loads(json.dumps(prompt_template))

        # Navigate to the instruction block where data will be injected
        instruction_block = final_payload['instructions_and_output_structure']
        input_data_section = instruction_block['input_data']

        # Inject the hardcoded page data and metadata
        input_data_section['page_data'] = page_data
        input_data_section['column_metadata'] = column_metadata.get('column_metadata', [])

        # Inject metrics thresholds if they exist
        if metrics_thresholds:
            input_data_section['key_metrics_threshold_info'] = metrics_thresholds
        
        print("Payload assembled successfully.")
        return final_payload
    except KeyError as e:
        print(f"ERROR: The prompt template is missing a required key. Could not find key: {e}")
        return None

def call_llm(payload_dict, endpoint_name):
    """
    Calls the Databricks Model Serving endpoint using the mlflow.deployments client.
    """
    print("\n" + "="*70)
    print(f"Calling LLM Endpoint: {endpoint_name}...")

    try:
        # Convert the entire payload dictionary into a single JSON string for the prompt
        prompt = json.dumps(payload_dict, indent=2)

        # Initialize the deployment client
        client = mlflow.deployments.get_deploy_client("databricks")
        
        # Structure the final payload for the Databricks LLM serving API
        inputs_payload = {
            "messages": [
                {"role": "user", "content": prompt}
            ]
        }
        
        # Call the LLM deployment endpoint
        response = client.predict(
            endpoint=endpoint_name, 
            inputs=inputs_payload
        )
        
        # Extract the summary text from the response structure
        summary = response.get('choices', [{}])[0].get('message', {}).get('content', "Error: Could not extract content from LLM response.")
        
        # Clean up the summary (e.g., strip surrounding whitespace)
        summary = summary.strip()
        
        print("✅ LLM Summary Generated Successfully.")
        return summary

    except Exception as e:
        # Handle any connection or prediction errors gracefully
        error_message = f"Error generating summary from LLM endpoint: {str(e)}"
        print(f"❌ {error_message}")
        return error_message

def main():
    """
    Main function to orchestrate the summarization pipeline.
    """
    print(f"--- Starting Summary Generation for Page '{PAGE_TO_PROCESS}' ---")

    # Construct the full path to the page's configuration folder
    page_folder = os.path.join(BASE_CONFIG_PATH, PAGE_TO_PROCESS)

    # 1. Load all configuration files
    prompt_template, column_metadata, metrics_thresholds = load_config_files(page_folder, PAGE_TO_PROCESS)
    if not prompt_template:
        print("ERROR: Failed to load configuration. Aborting.")
        return

    # 2. Use the hardcoded page data
    page_data = PAGE_DATA
    print("Using hardcoded page data.")
    
    # 3. Assemble the final payload for the LLM
    llm_payload = assemble_llm_payload(prompt_template, page_data, column_metadata, metrics_thresholds)
    if not llm_payload:
        print("ERROR: Failed to assemble LLM payload. Aborting.")
        return

    # 4. Call the LLM to get the summary
    summary = call_llm(llm_payload, LLM_ENDPOINT_NAME)
    if not summary:
        print("ERROR: Failed to generate summary. Aborting.")
        return
        
    # 5. Display the final summary
    print("\n--- FINAL SUMMARY PREVIEW ---")
    print(summary)
    print("---------------------------\n")

# Run the main pipeline function
if __name__ == "__main__":
    main()
    

# COMMAND ----------

# MAGIC %md
# MAGIC Comparative Summary

# COMMAND ----------

import json
import os
import mlflow.deployments
from pyspark.sql import SparkSession
# Initialize Spark session (Databricks automatically provides dbutils)
spark = SparkSession.builder.appName("SummarizationPipeline").getOrCreate()
# ---------------------------------------------------------------------------
# GLOBAL LLM INTERACTION FUNCTIONS (Reusable across all pipelines)
# ---------------------------------------------------------------------------
def load_comparative_metrics(base_path, page_name):
    """
    Loads the JSON file containing the regional comparative metrics.
    """
    print("Loading comparative data file...")
    comparative_path = os.path.join(base_path, f"{page_name}_comparative_metrics.json")

    try:
        with open(comparative_path, "r") as f:
            comparative_metrics = json.load(f)
            print(f"INFO: Successfully loaded comparative metrics from {comparative_path}")
            return comparative_metrics
    except FileNotFoundError:
        print(f"ERROR: Comparative metrics file not found at {comparative_path}. Aborting.")
        return None
    except json.JSONDecodeError as e:
        print(f"ERROR: Failed to parse comparative metrics JSON. Details: {e}")
        return None
def assemble_llm_payload(prompt_template, trial_data, comparative_data):
    """
    Assembles the final LLM prompt string by injecting trial and comparative data.
    """
    print("Assembling the final LLM payload...")
    trial_data_str = json.dumps(trial_data, indent=2)
    comparative_data_str = json.dumps(comparative_data, indent=2)
    final_prompt = prompt_template.format(
        trial_data=trial_data_str,
        comparative_data=comparative_data_str
    )
    llm_payload = {"messages": [{"role": "user", "content": final_prompt}]}
    print("Payload assembled successfully.")
    return llm_payload
def call_llm(payload_dict, endpoint_name):
    """
    Calls the Databricks Model Serving endpoint.
    """
    print("\n" + "=" * 70)
    print(f"Calling LLM Endpoint: {endpoint_name}...")
    try:
        client = mlflow.deployments.get_deploy_client("databricks")
        # Depending on the model type, you may need to use 'inputs' instead of 'messages'
        response = client.predict(endpoint=endpoint_name, inputs=payload_dict)
        summary = (
            response.get("choices", [{}])[0]
            .get("message", {})
            .get("content", "Error: Could not extract content from LLM response.")
            .strip()
        )
        print(":white_check_mark: LLM Summary Generated Successfully.")
        return summary
    except Exception as e:
        error_message = f"Error generating summary from LLM endpoint: {str(e)}"
        print(f":x: {error_message}")
        return error_message
# ---------------------------------------------------------------------------
# MAIN PIPELINE EXECUTION FUNCTION
# ---------------------------------------------------------------------------
def run_pipeline(page_name, base_data_path, llm_endpoint_name, specific_trial_data, llm_prompt_template):
    """
    Orchestrates the LLM summarization pipeline for a single trial.
    """
    print(f"--- Starting Summary Generation for Page '{page_name}' ---")
    comparative_data = load_comparative_metrics(base_data_path, page_name)
    if not comparative_data:
        print("ERROR: Failed to load comparative data. Aborting.")
        return
    llm_payload = assemble_llm_payload(llm_prompt_template, specific_trial_data, comparative_data)
    summary = call_llm(llm_payload, llm_endpoint_name)
    if summary and not summary.startswith("Error"):
        print("\n--- FINAL SUMMARY PREVIEW ---")
        print(summary)
        print("---------------------------\n")
    else:
        print("ERROR: Failed to generate summary. Aborting.")
    return summary
# ---------------------------------------------------------------------------
# CONFIGURATION SECTION
# ---------------------------------------------------------------------------
PAGE_TO_PROCESS = "page7"
BASE_DATA_PATH = "/Workspace/Users/ayush.dongardive@pivotbio.com/LLM Pipeline/Comparitive Metric"
LLM_ENDPOINT_NAME = "databricks-llama-4-maverick"
SPECIFIC_TRIAL_DATA = {
    "NTC: 100%N": "30889",
  "NTC: 100%N -BMP": "31264",
  "PBDryB: 100%N -BMP": "30952",
  "PBDryS: 100%N -BMP": "30451",
  "PBOS: 100%N": "31139",
  "PBOS: 100%N -BMP": "31202"
  
}
# Use triple single quotes to avoid conflicts with double quotes in JSON-style text
LLM_COMPARISON_PROMPT = '''
You are an Analytical AI Assistant for the Pivot Bio Sales and Agronomy Team.
You specialize in comparing trial setup and mid-season nutrient status against regional benchmarks.
Company Context:
Pivot Bio is a U.S.-based agricultural biotechnology company that develops microbe-based nitrogen solutions to replace synthetic fertilizers.
Their flagship product, Pivot Bio PROVEN® 40, replaces synthetic nitrogen, improving yield consistency and environmental outcomes.
They run field-scale validation trials across the U.S. to demonstrate nitrogen-reduction performance.
Task:
Analyze the provided data for this specific trial, comparing its stand count and N-management strategy against regional averages and established targets. Generate a comprehensive comparative summary focusing on stand count uniformity and its implications for yield potential.
Context Summary:
"Initial stand count analysis shows high uniformity across all treatments, but the trial average of 30,983 plants/ac is below the target of 35,500. The variability among treatments is less than 3 of the average stand. Stand issues are not treatment-related but field-wide"
Input Data:
- Specific Trial Data: {trial_data}
- Comparative Metrics: {comparative_data}
Output:
"title": "Stand Count Uniformity and Comparison",
"instruction": "Compare the trial's average stand (30,983) and the stand counts of individual treatments against the regional average (32,500) and the trial target (35,500). Calculate the percent difference of the trial average from the target. Analyze the difference between the highest and lowest stand count treatments to assess uniformity. State if the Pivot Bio products (OS, DryB, DryS) caused any emergence issues compared to the NTC.
"title": "Strategy Context and Potential Impact",
"instruction": " Use the Stand Count results to determine if the low stand count (compared to the target) might be the primary limiting factor for yield, potentially overshadowing the N-treatment effects later in the season. "
"title": "Sales and Agronomic Takeaway",
"instruction": "Deliver a 2-3 sentence business interpretation. Frame the highly uniform stand counts as a positive, confirming that Pivot Bio products did not inhibit germination. Then, emphasize that the sub-optimal overall plant population is the immediate challenge, suggesting that the focus should be on managing the remaining population for optimal yield (e.g., via irrigation/fertility) rather than diagnosing a product-related stand issue."

      "tone": "Analytical, comparative, and highly business-aligned.",
      "length": "250-300 words."

'''
# ---------------------------------------------------------------------------
# RUN PIPELINE
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    run_pipeline(
        page_name=PAGE_TO_PROCESS,
        base_data_path=BASE_DATA_PATH,
        llm_endpoint_name=LLM_ENDPOINT_NAME,
        specific_trial_data=SPECIFIC_TRIAL_DATA,
        llm_prompt_template=LLM_COMPARISON_PROMPT
    )


# COMMAND ----------

# MAGIC %md
# MAGIC ### Page 9

# COMMAND ----------

import json
import os
import mlflow.deployments
from pyspark.sql import SparkSession
from pyspark.dbutils import DBUtils

# ---------------------------------------------------------------------------
# CONFIGURATION: YOU MUST EDIT THIS SECTION
# ---------------------------------------------------------------------------
# 1. SET THE PAGE TO PROCESS
#    (e.g., "page1", "page3", "page4")
PAGE_TO_PROCESS = "page9"

# 2. SET THE BASE PATH TO YOUR "pages" FOLDER
#    Example: "/Workspace/Users/your.email@company.com/LLM_Project/pages"
BASE_CONFIG_PATH = "/Workspace/Users/ayush.dongardive@pivotbio.com/LLM Pipeline/pages"

# 3. SET THE DATABRICKS MODEL SERVING ENDPOINT NAME
LLM_ENDPOINT_NAME = "databricks-llama-4-maverick"

# 4. PASTE THE HARDCODED DATA FOR THE PAGE HERE
#    This dictionary contains the data that would have come from the SQL query.
#    This example is pre-filled with the data for Page 1.
PAGE_DATA = {
  
  "NTC: 100%N_Fresh_Weight_Diff_g": "0",
  "NTC: 100%N_Percent_Diff": "0",
  "NTC: 100%N_-BMP_Fresh_Weight_Diff_g": "14",
  "NTC: 100%N_-BMP_Percent_Diff": "4",
  "PBDryA: 100%N_-BMP_Fresh_Weight_Diff_g": "11",
  "PBDryA: 100%N_-BMP_Percent_Diff": "3",
  "PBDryB: 100%N_-BMP_Fresh_Weight_Diff_g": "19",
  "PBDryB: 100%N_-BMP_Percent_Diff": "6",
  "PBOS: 100%N_Fresh_Weight_Diff_g": "16",
  "PBOS: 100%N_Percent_Diff": "5",
  "PBOS: 100%N_-BMP_Percent_Diff": "5"

  
}
# ---------------------------------------------------------------------------

# Initialize Spark Session and DBUtils (standard for Databricks notebooks)
spark = SparkSession.builder.appName("SummarizationPipeline").getOrCreate()
dbutils = DBUtils(spark)

def load_config_files(page_folder_path, page_name):
    """
    Loads all necessary JSON configuration files for a given page.
    It gracefully handles the absence of the optional metrics file.
    """
    print("Loading configuration files...")
    try:
        # Define paths for all potential files
        prompt_path = os.path.join(page_folder_path, f"{page_name}_prompt.json")
        metadata_path = os.path.join(page_folder_path, f"{page_name}_column_metadata.json")
        metrics_path = os.path.join(page_folder_path, f"{page_name}_metrics.json")

        # Load mandatory files
        with open(prompt_path, 'r') as f:
            prompt_template = json.load(f)
        with open(metadata_path, 'r') as f:
            column_metadata = json.load(f)

        # Load optional metrics file
        metrics_thresholds = None
        if os.path.exists(metrics_path):
            with open(metrics_path, 'r') as f:
                metrics_thresholds = json.load(f)
                print(f"INFO: Successfully loaded optional metrics file from {metrics_path}")
        else:
            print(f"INFO: Optional file not found at {metrics_path}. Proceeding without it.")

        return prompt_template, column_metadata, metrics_thresholds
    except FileNotFoundError as e:
        print(f"ERROR: A mandatory configuration file was not found. Details: {e}")
        return None, None, None
    except json.JSONDecodeError as e:
        print(f"ERROR: Failed to parse a JSON configuration file. Please check for syntax errors. Details: {e}")
        return None, None, None

def assemble_llm_payload(prompt_template, page_data, column_metadata, metrics_thresholds):
    """
    Assembles the final JSON payload to be sent to the LLM.
    """
    print("Assembling the final LLM payload...")
    try:
        # Start with a deep copy of the template to avoid modifying the original
        final_payload = json.loads(json.dumps(prompt_template))

        # Navigate to the instruction block where data will be injected
        instruction_block = final_payload['instructions_and_output_structure']
        input_data_section = instruction_block['input_data']

        # Inject the hardcoded page data and metadata
        input_data_section['page_data'] = page_data
        input_data_section['column_metadata'] = column_metadata.get('column_metadata', [])

        # Inject metrics thresholds if they exist
        if metrics_thresholds:
            input_data_section['key_metrics_threshold_info'] = metrics_thresholds
        
        print("Payload assembled successfully.")
        return final_payload
    except KeyError as e:
        print(f"ERROR: The prompt template is missing a required key. Could not find key: {e}")
        return None

def call_llm(payload_dict, endpoint_name):
    """
    Calls the Databricks Model Serving endpoint using the mlflow.deployments client.
    """
    print("\n" + "="*70)
    print(f"Calling LLM Endpoint: {endpoint_name}...")

    try:
        # Convert the entire payload dictionary into a single JSON string for the prompt
        prompt = json.dumps(payload_dict, indent=2)

        # Initialize the deployment client
        client = mlflow.deployments.get_deploy_client("databricks")
        
        # Structure the final payload for the Databricks LLM serving API
        inputs_payload = {
            "messages": [
                {"role": "user", "content": prompt}
            ]
        }
        
        # Call the LLM deployment endpoint
        response = client.predict(
            endpoint=endpoint_name, 
            inputs=inputs_payload
        )
        
        # Extract the summary text from the response structure
        summary = response.get('choices', [{}])[0].get('message', {}).get('content', "Error: Could not extract content from LLM response.")
        
        # Clean up the summary (e.g., strip surrounding whitespace)
        summary = summary.strip()
        
        print("✅ LLM Summary Generated Successfully.")
        return summary

    except Exception as e:
        # Handle any connection or prediction errors gracefully
        error_message = f"Error generating summary from LLM endpoint: {str(e)}"
        print(f"❌ {error_message}")
        return error_message

def main():
    """
    Main function to orchestrate the summarization pipeline.
    """
    print(f"--- Starting Summary Generation for Page '{PAGE_TO_PROCESS}' ---")

    # Construct the full path to the page's configuration folder
    page_folder = os.path.join(BASE_CONFIG_PATH, PAGE_TO_PROCESS)

    # 1. Load all configuration files
    prompt_template, column_metadata, metrics_thresholds = load_config_files(page_folder, PAGE_TO_PROCESS)
    if not prompt_template:
        print("ERROR: Failed to load configuration. Aborting.")
        return

    # 2. Use the hardcoded page data
    page_data = PAGE_DATA
    print("Using hardcoded page data.")
    
    # 3. Assemble the final payload for the LLM
    llm_payload = assemble_llm_payload(prompt_template, page_data, column_metadata, metrics_thresholds)
    if not llm_payload:
        print("ERROR: Failed to assemble LLM payload. Aborting.")
        return

    # 4. Call the LLM to get the summary
    summary = call_llm(llm_payload, LLM_ENDPOINT_NAME)
    if not summary:
        print("ERROR: Failed to generate summary. Aborting.")
        return
        
    # 5. Display the final summary
    print("\n--- FINAL SUMMARY PREVIEW ---")
    print(summary)
    print("---------------------------\n")

# Run the main pipeline function
if __name__ == "__main__":
    main()
    

# COMMAND ----------

# MAGIC %md
# MAGIC Comparative Summary

# COMMAND ----------

import json
import os
import mlflow.deployments
from pyspark.sql import SparkSession
# Initialize Spark session (Databricks automatically provides dbutils)
spark = SparkSession.builder.appName("SummarizationPipeline").getOrCreate()
# ---------------------------------------------------------------------------
# GLOBAL LLM INTERACTION FUNCTIONS (Reusable across all pipelines)
# ---------------------------------------------------------------------------
def load_comparative_metrics(base_path, page_name):
    """
    Loads the JSON file containing the regional comparative metrics.
    """
    print("Loading comparative data file...")
    comparative_path = os.path.join(base_path, f"{page_name}_comparative_metrics.json")

    try:
        with open(comparative_path, "r") as f:
            comparative_metrics = json.load(f)
            print(f"INFO: Successfully loaded comparative metrics from {comparative_path}")
            return comparative_metrics
    except FileNotFoundError:
        print(f"ERROR: Comparative metrics file not found at {comparative_path}. Aborting.")
        return None
    except json.JSONDecodeError as e:
        print(f"ERROR: Failed to parse comparative metrics JSON. Details: {e}")
        return None
def assemble_llm_payload(prompt_template, trial_data, comparative_data):
    """
    Assembles the final LLM prompt string by injecting trial and comparative data.
    """
    print("Assembling the final LLM payload...")
    trial_data_str = json.dumps(trial_data, indent=2)
    comparative_data_str = json.dumps(comparative_data, indent=2)
    final_prompt = prompt_template.format(
        trial_data=trial_data_str,
        comparative_data=comparative_data_str
    )
    llm_payload = {"messages": [{"role": "user", "content": final_prompt}]}
    print("Payload assembled successfully.")
    return llm_payload
def call_llm(payload_dict, endpoint_name):
    """
    Calls the Databricks Model Serving endpoint.
    """
    print("\n" + "=" * 70)
    print(f"Calling LLM Endpoint: {endpoint_name}...")
    try:
        client = mlflow.deployments.get_deploy_client("databricks")
        # Depending on the model type, you may need to use 'inputs' instead of 'messages'
        response = client.predict(endpoint=endpoint_name, inputs=payload_dict)
        summary = (
            response.get("choices", [{}])[0]
            .get("message", {})
            .get("content", "Error: Could not extract content from LLM response.")
            .strip()
        )
        print(":white_check_mark: LLM Summary Generated Successfully.")
        return summary
    except Exception as e:
        error_message = f"Error generating summary from LLM endpoint: {str(e)}"
        print(f":x: {error_message}")
        return error_message
# ---------------------------------------------------------------------------
# MAIN PIPELINE EXECUTION FUNCTION
# ---------------------------------------------------------------------------
def run_pipeline(page_name, base_data_path, llm_endpoint_name, specific_trial_data, llm_prompt_template):
    """
    Orchestrates the LLM summarization pipeline for a single trial.
    """
    print(f"--- Starting Summary Generation for Page '{page_name}' ---")
    comparative_data = load_comparative_metrics(base_data_path, page_name)
    if not comparative_data:
        print("ERROR: Failed to load comparative data. Aborting.")
        return
    llm_payload = assemble_llm_payload(llm_prompt_template, specific_trial_data, comparative_data)
    summary = call_llm(llm_payload, llm_endpoint_name)
    if summary and not summary.startswith("Error"):
        print("\n--- FINAL SUMMARY PREVIEW ---")
        print(summary)
        print("---------------------------\n")
    else:
        print("ERROR: Failed to generate summary. Aborting.")
    return summary
# ---------------------------------------------------------------------------
# CONFIGURATION SECTION
# ---------------------------------------------------------------------------
PAGE_TO_PROCESS = "page9"
BASE_DATA_PATH = "/Workspace/Users/ayush.dongardive@pivotbio.com/LLM Pipeline/Comparitive Metric"
LLM_ENDPOINT_NAME = "databricks-llama-4-maverick"
SPECIFIC_TRIAL_DATA = {
    
  "NTC: 100%N_Fresh_Weight_Diff_g": "0",
  "NTC: 100%N_Percent_Diff": "0",
  "NTC: 100%N_-BMP_Fresh_Weight_Diff_g": "14",
  "NTC: 100%N_-BMP_Percent_Diff": "4",
  "PBDryA: 100%N_-BMP_Fresh_Weight_Diff_g": "11",
  "PBDryA: 100%N_-BMP_Percent_Diff": "3",
  "PBDryB: 100%N_-BMP_Fresh_Weight_Diff_g": "19",
  "PBDryB: 100%N_-BMP_Percent_Diff": "6",
  "PBOS: 100%N_Fresh_Weight_Diff_g": "16",
  "PBOS: 100%N_Percent_Diff": "5",
  "PBOS: 100%N_-BMP_Percent_Diff": "5"

}
# Use triple single quotes to avoid conflicts with double quotes in JSON-style text


## LLM Comparison Prompt for Yield Monitor Data (Text Format)

LLM_COMPARISON_PROMPT = '''
You are an Analytical AI Assistant for the Pivot Bio Sales and Agronomy Team.
You specialize in comparing trial setup, early-season plant vigor (fresh weight), and performance against regional benchmarks.
Company Context:
Pivot Bio is a U.S.-based agricultural biotechnology company that develops microbe-based nitrogen solutions to replace synthetic fertilizers. Their flagship product, Pivot Bio PROVEN® 40, replaces synthetic nitrogen, improving yield consistency and environmental outcomes. They run field-scale validation trials across the U.S. to demonstrate nitrogen-reduction performance.
Task:
Analyze the provided Above Ground Fresh Weight data, comparing the trial's treatment-specific vigor against the regional averages. Generate a comprehensive summary quantifying the product's effect on early growth relative to the regional peer group.
Input Data:
- Specific Trial Data: {trial_data}
- Comparative Metrics: {comparative_data}
Output:
"title": "Ground Truth: Trial Vigor vs. Regional Status",
"instruction": "Summarize the trial's maximum fresh weight difference (in grams and percentage) and identify the top performing treatment. Compare the trial's maximum fresh weight difference directly to the corresponding regional average fresh weight difference and quantify the margin (better or worse)."
"title": "Technical Understanding: Vigor Efficacy and Input Correlation",
"instruction": "Analyze the vigor performance of the Reduced N treatments with Pivot Bio product (e.g., PBDryB: -BMP) against the NTC: 100%N -BMP baseline. Quantify the margin by which the best Pivot Bio treatment beat the NTC: 100%N -BMP. Use the regional comparison (Trial Max Vigor vs. Regional Avg Max Vigor) to provide insight into whether this grower's field demonstrated better or worse early-season vigor support than the average field in the state."
"title": "Sales Takeaway: Early Vigor Validation and Regional Context",
"instruction": "Deliver a 2-3 sentence business interpretation. Conclude that the positive Fresh Weight Difference provides a powerful, early-season visual and numerical proof point that the product is effectively supplying nitrogen and driving superior physical vigor. Emphasize how the trial's superior/inferior performance relative to the regional average should shape the pitch regarding the product's value in driving growth in this specific area."
"tone": "Analytical, factual, and highly focused on regional business impact.",
"length": "180-250 words."
'''

# ---------------------------------------------------------------------------
# RUN PIPELINE
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    run_pipeline(
        page_name=PAGE_TO_PROCESS,
        base_data_path=BASE_DATA_PATH,
        llm_endpoint_name=LLM_ENDPOINT_NAME,
        specific_trial_data=SPECIFIC_TRIAL_DATA,
        llm_prompt_template=LLM_COMPARISON_PROMPT
    )


# COMMAND ----------

# MAGIC %md
# MAGIC ### Page 10 Tissue Sample

# COMMAND ----------

import json
import os
import mlflow.deployments
from pyspark.sql import SparkSession
from pyspark.dbutils import DBUtils

# ---------------------------------------------------------------------------
# CONFIGURATION: YOU MUST EDIT THIS SECTION
# ---------------------------------------------------------------------------
# 1. SET THE PAGE TO PROCESS
#    (e.g., "page1", "page3", "page4")
PAGE_TO_PROCESS = "page10"

# 2. SET THE BASE PATH TO YOUR "pages" FOLDER
#    Example: "/Workspace/Users/your.email@company.com/LLM_Project/pages"
BASE_CONFIG_PATH = "/Workspace/Users/ayush.dongardive@pivotbio.com/LLM Pipeline/pages"

# 3. SET THE DATABRICKS MODEL SERVING ENDPOINT NAME
LLM_ENDPOINT_NAME = "databricks-llama-4-maverick"

# 4. PASTE THE HARDCODED DATA FOR THE PAGE HERE
#    This dictionary contains the data that would have come from the SQL query.
#    This example is pre-filled with the data for Page 1.
PAGE_DATA = {

  "N_Unit": "N (%)",
  "N_Site_Average": 4.4,
  "NTC: 100%N_N_Percent": 4.46,
  "PBOS: 100%N_N_Percent": 4.32,
  "PBDryB: 100%N -BMP_N_Percent": 4.36,
  "PBDryS: 100%N -BMP_N_Percent": 4.47,
  "P_Unit": "P (%)",
  "P_Site_Average": 0.41,
  "NTC: 100%N_P_Percent": 0.41,
  "PBDryB: 100%N -BMP_P_Percent": 0.40,
  "PBDryS: 100%N -BMP_P_Percent": 0.44,
  "PBOS: 100%N_P_Percent": 0.39
  
}
# ---------------------------------------------------------------------------

# Initialize Spark Session and DBUtils (standard for Databricks notebooks)
spark = SparkSession.builder.appName("SummarizationPipeline").getOrCreate()
dbutils = DBUtils(spark)

def load_config_files(page_folder_path, page_name):
    """
    Loads all necessary JSON configuration files for a given page.
    It gracefully handles the absence of the optional metrics file.
    """
    print("Loading configuration files...")
    try:
        # Define paths for all potential files
        prompt_path = os.path.join(page_folder_path, f"{page_name}_prompt.json")
        metadata_path = os.path.join(page_folder_path, f"{page_name}_column_metadata.json")
        metrics_path = os.path.join(page_folder_path, f"{page_name}_metrics.json")

        # Load mandatory files
        with open(prompt_path, 'r') as f:
            prompt_template = json.load(f)
        with open(metadata_path, 'r') as f:
            column_metadata = json.load(f)

        # Load optional metrics file
        metrics_thresholds = None
        if os.path.exists(metrics_path):
            with open(metrics_path, 'r') as f:
                metrics_thresholds = json.load(f)
                print(f"INFO: Successfully loaded optional metrics file from {metrics_path}")
        else:
            print(f"INFO: Optional file not found at {metrics_path}. Proceeding without it.")

        return prompt_template, column_metadata, metrics_thresholds
    except FileNotFoundError as e:
        print(f"ERROR: A mandatory configuration file was not found. Details: {e}")
        return None, None, None
    except json.JSONDecodeError as e:
        print(f"ERROR: Failed to parse a JSON configuration file. Please check for syntax errors. Details: {e}")
        return None, None, None

def assemble_llm_payload(prompt_template, page_data, column_metadata, metrics_thresholds):
    """
    Assembles the final JSON payload to be sent to the LLM.
    """
    print("Assembling the final LLM payload...")
    try:
        # Start with a deep copy of the template to avoid modifying the original
        final_payload = json.loads(json.dumps(prompt_template))

        # Navigate to the instruction block where data will be injected
        instruction_block = final_payload['instructions_and_output_structure']
        input_data_section = instruction_block['input_data']

        # Inject the hardcoded page data and metadata
        input_data_section['page_data'] = page_data
        input_data_section['column_metadata'] = column_metadata.get('column_metadata', [])

        # Inject metrics thresholds if they exist
        if metrics_thresholds:
            input_data_section['key_metrics_threshold_info'] = metrics_thresholds
        
        print("Payload assembled successfully.")
        return final_payload
    except KeyError as e:
        print(f"ERROR: The prompt template is missing a required key. Could not find key: {e}")
        return None

def call_llm(payload_dict, endpoint_name):
    """
    Calls the Databricks Model Serving endpoint using the mlflow.deployments client.
    """
    print("\n" + "="*70)
    print(f"Calling LLM Endpoint: {endpoint_name}...")

    try:
        # Convert the entire payload dictionary into a single JSON string for the prompt
        prompt = json.dumps(payload_dict, indent=2)

        # Initialize the deployment client
        client = mlflow.deployments.get_deploy_client("databricks")
        
        # Structure the final payload for the Databricks LLM serving API
        inputs_payload = {
            "messages": [
                {"role": "user", "content": prompt}
            ]
        }
        
        # Call the LLM deployment endpoint
        response = client.predict(
            endpoint=endpoint_name, 
            inputs=inputs_payload
        )
        
        # Extract the summary text from the response structure
        summary = response.get('choices', [{}])[0].get('message', {}).get('content', "Error: Could not extract content from LLM response.")
        
        # Clean up the summary (e.g., strip surrounding whitespace)
        summary = summary.strip()
        
        print("✅ LLM Summary Generated Successfully.")
        return summary

    except Exception as e:
        # Handle any connection or prediction errors gracefully
        error_message = f"Error generating summary from LLM endpoint: {str(e)}"
        print(f"❌ {error_message}")
        return error_message

def main():
    """
    Main function to orchestrate the summarization pipeline.
    """
    print(f"--- Starting Summary Generation for Page '{PAGE_TO_PROCESS}' ---")

    # Construct the full path to the page's configuration folder
    page_folder = os.path.join(BASE_CONFIG_PATH, PAGE_TO_PROCESS)

    # 1. Load all configuration files
    prompt_template, column_metadata, metrics_thresholds = load_config_files(page_folder, PAGE_TO_PROCESS)
    if not prompt_template:
        print("ERROR: Failed to load configuration. Aborting.")
        return

    # 2. Use the hardcoded page data
    page_data = PAGE_DATA
    print("Using hardcoded page data.")
    
    # 3. Assemble the final payload for the LLM
    llm_payload = assemble_llm_payload(prompt_template, page_data, column_metadata, metrics_thresholds)
    if not llm_payload:
        print("ERROR: Failed to assemble LLM payload. Aborting.")
        return

    # 4. Call the LLM to get the summary
    summary = call_llm(llm_payload, LLM_ENDPOINT_NAME)
    if not summary:
        print("ERROR: Failed to generate summary. Aborting.")
        return
        
    # 5. Display the final summary
    print("\n--- FINAL SUMMARY PREVIEW ---")
    print(summary)
    print("---------------------------\n")

# Run the main pipeline function
if __name__ == "__main__":
    main()
    

# COMMAND ----------

# MAGIC %md
# MAGIC Cpmparative Summary

# COMMAND ----------

import json
import os
import mlflow.deployments
from pyspark.sql import SparkSession
# Initialize Spark session (Databricks automatically provides dbutils)
spark = SparkSession.builder.appName("SummarizationPipeline").getOrCreate()
# ---------------------------------------------------------------------------
# GLOBAL LLM INTERACTION FUNCTIONS (Reusable across all pipelines)
# ---------------------------------------------------------------------------
def load_comparative_metrics(base_path, page_name):
    """
    Loads the JSON file containing the regional comparative metrics.
    """
    print("Loading comparative data file...")
    comparative_path = os.path.join(base_path, f"{page_name}_comparative_metrics.json")

    try:
        with open(comparative_path, "r") as f:
            comparative_metrics = json.load(f)
            print(f"INFO: Successfully loaded comparative metrics from {comparative_path}")
            return comparative_metrics
    except FileNotFoundError:
        print(f"ERROR: Comparative metrics file not found at {comparative_path}. Aborting.")
        return None
    except json.JSONDecodeError as e:
        print(f"ERROR: Failed to parse comparative metrics JSON. Details: {e}")
        return None
def assemble_llm_payload(prompt_template, trial_data, comparative_data):
    """
    Assembles the final LLM prompt string by injecting trial and comparative data.
    """
    print("Assembling the final LLM payload...")
    trial_data_str = json.dumps(trial_data, indent=2)
    comparative_data_str = json.dumps(comparative_data, indent=2)
    final_prompt = prompt_template.format(
        trial_data=trial_data_str,
        comparative_data=comparative_data_str
    )
    llm_payload = {"messages": [{"role": "user", "content": final_prompt}]}
    print("Payload assembled successfully.")
    return llm_payload
def call_llm(payload_dict, endpoint_name):
    """
    Calls the Databricks Model Serving endpoint.
    """
    print("\n" + "=" * 70)
    print(f"Calling LLM Endpoint: {endpoint_name}...")
    try:
        client = mlflow.deployments.get_deploy_client("databricks")
        # Depending on the model type, you may need to use 'inputs' instead of 'messages'
        response = client.predict(endpoint=endpoint_name, inputs=payload_dict)
        summary = (
            response.get("choices", [{}])[0]
            .get("message", {})
            .get("content", "Error: Could not extract content from LLM response.")
            .strip()
        )
        print(":white_check_mark: LLM Summary Generated Successfully.")
        return summary
    except Exception as e:
        error_message = f"Error generating summary from LLM endpoint: {str(e)}"
        print(f":x: {error_message}")
        return error_message
# ---------------------------------------------------------------------------
# MAIN PIPELINE EXECUTION FUNCTION
# ---------------------------------------------------------------------------
def run_pipeline(page_name, base_data_path, llm_endpoint_name, specific_trial_data, llm_prompt_template):
    """
    Orchestrates the LLM summarization pipeline for a single trial.
    """
    print(f"--- Starting Summary Generation for Page '{page_name}' ---")
    comparative_data = load_comparative_metrics(base_data_path, page_name)
    if not comparative_data:
        print("ERROR: Failed to load comparative data. Aborting.")
        return
    llm_payload = assemble_llm_payload(llm_prompt_template, specific_trial_data, comparative_data)
    summary = call_llm(llm_payload, llm_endpoint_name)
    if summary and not summary.startswith("Error"):
        print("\n--- FINAL SUMMARY PREVIEW ---")
        print(summary)
        print("---------------------------\n")
    else:
        print("ERROR: Failed to generate summary. Aborting.")
    return summary
# ---------------------------------------------------------------------------
# CONFIGURATION SECTION
# ---------------------------------------------------------------------------
PAGE_TO_PROCESS = "page10"
BASE_DATA_PATH = "/Workspace/Users/ayush.dongardive@pivotbio.com/LLM Pipeline/Comparitive Metric"
LLM_ENDPOINT_NAME = "databricks-llama-4-maverick"
SPECIFIC_TRIAL_DATA = {
    
  "N_Unit": "N (%)",
  "N_Site_Average": 4.4,
  "NTC: 100%N_N_Percent": 4.46,
  "PBOS: 100%N_N_Percent": 4.32,
  "PBDryB: 100%N -BMP_N_Percent": 4.36,
  "PBDryS: 100%N -BMP_N_Percent": 4.47,
  "P_Unit": "P (%)",
  "P_Site_Average": 0.41,
  "NTC: 100%N_P_Percent": 0.41,
  "PBDryB: 100%N -BMP_P_Percent": 0.40,
  "PBDryS: 100%N -BMP_P_Percent": 0.44,
  "PBOS: 100%N_P_Percent": 0.39

}
# Use triple single quotes to avoid conflicts with double quotes in JSON-style text


## LLM Comparison Prompt for Yield Monitor Data (Text Format)

LLM_COMPARISON_PROMPT = '''
You are an Analytical AI Assistant for the Pivot Bio Sales and Agronomy Team, specializing in nutrient analysis and regional performance benchmarking.
Company Context:
Pivot Bio is a U.S.-based agricultural biotechnology company that develops microbe-based nitrogen solutions to replace synthetic fertilizers. Their flagship product, Pivot Bio PROVEN® 40, replaces synthetic nitrogen, improving yield consistency and environmental outcomes. They run field-scale validation trials across the U.S. to demonstrate nitrogen-reduction performance.
Task:
Analyze the V11 Tissue Sample data, comparing the trial's nutrient levels and uniformity against the State Average metrics. Generate insights on how the N-reduction strategy performed relative to the region and its impact on P uptake stability.
Input Data:
- Specific Trial Data: {trial_data}
- Comparative Metrics: {comparative_data}
Output:
"title": "Ground Truth: Trial vs. Regional Status",
"instruction": "Summarize the trial's site-average N  and P  and compare these values directly to the regional average N  and P  (from the comparative metrics). Quantify the percentage difference for both N and P between the trial and the regional average. State the highest tissue N recorded in the trial and its treatment."
"title": "Technical Understanding: N Efficacy and Input Correlation",
"instruction": "Explain the nutrient status by comparing the trial's site average N  against its threshold (High/Sufficient/Deficient). Crucially, analyze the performance of the Reduced N plots (PBDryB/S: -BMP) by comparing their N  directly to the NTC: 100 N plot. Use the regional comparison (Trial N vs. Regional Avg N) to provide insight into whether this grower's soil supported the reduction better or worse than the average field in the state."
"title": "Sales Takeaway: Value Proposition & Regional Context",
"instruction": "Provide a business insight. Conclude that the trial's N  status relative to the regional average shows the grower is either performing exceptionally well or had higher residual N. Frame the strong N status in the reduced-N plots as a direct, quantifiable validation of Pivot Bio's efficacy. Emphasize that the stability of P  (its lack of variability) demonstrates a stable fertility foundation, allowing the sales conversation to be entirely focused on the demonstrated success of microbial N replacement and the safe reduction of synthetic input."
"tone": "Analytical, factual, and highly focused on regional business impact.",
"length": "180-250 words."
'''

# ---------------------------------------------------------------------------
# RUN PIPELINE
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    run_pipeline(
        page_name=PAGE_TO_PROCESS,
        base_data_path=BASE_DATA_PATH,
        llm_endpoint_name=LLM_ENDPOINT_NAME,
        specific_trial_data=SPECIFIC_TRIAL_DATA,
        llm_prompt_template=LLM_COMPARISON_PROMPT
    )


# COMMAND ----------

# MAGIC %md
# MAGIC ### Page 16 Harvest Yeild

# COMMAND ----------

import json
import os
import mlflow.deployments
from pyspark.sql import SparkSession
from pyspark.dbutils import DBUtils

# ---------------------------------------------------------------------------
# CONFIGURATION: YOU MUST EDIT THIS SECTION
# ---------------------------------------------------------------------------
# 1. SET THE PAGE TO PROCESS
#    (e.g., "page1", "page3", "page4")
PAGE_TO_PROCESS = "page16"

# 2. SET THE BASE PATH TO YOUR "pages" FOLDER
#    Example: "/Workspace/Users/your.email@company.com/LLM_Project/pages"
BASE_CONFIG_PATH = "/Workspace/Users/ayush.dongardive@pivotbio.com/LLM Pipeline/pages"

# 3. SET THE DATABRICKS MODEL SERVING ENDPOINT NAME
LLM_ENDPOINT_NAME = "databricks-llama-4-maverick"

# 4. PASTE THE HARDCODED DATA FOR THE PAGE HERE
#    This dictionary contains the data that would have come from the SQL query.
#    This example is pre-filled with the data for Page 1.
PAGE_DATA = {

  "NTC: N 100%_yeild": 275.0,
  "NTC: N 100%_moisture": 27.1,
  "NTC: N Reduced_yeild": 275.0,
  "NTC: N Reduced_moisture": 26.3,
  "PVN G3: N Reduced_yeild": 274.0,
  "PVN G3: N Reduced_moisture": 26.6
  
}
# ---------------------------------------------------------------------------

# Initialize Spark Session and DBUtils (standard for Databricks notebooks)
spark = SparkSession.builder.appName("SummarizationPipeline").getOrCreate()
dbutils = DBUtils(spark)

def load_config_files(page_folder_path, page_name):
    """
    Loads all necessary JSON configuration files for a given page.
    It gracefully handles the absence of the optional metrics file.
    """
    print("Loading configuration files...")
    try:
        # Define paths for all potential files
        prompt_path = os.path.join(page_folder_path, f"{page_name}_prompt.json")
        metadata_path = os.path.join(page_folder_path, f"{page_name}_column_metadata.json")
        metrics_path = os.path.join(page_folder_path, f"{page_name}_metrics_threshold.json")

        # Load mandatory files
        with open(prompt_path, 'r') as f:
            prompt_template = json.load(f)
        with open(metadata_path, 'r') as f:
            column_metadata = json.load(f)

        # Load optional metrics file
        metrics_thresholds = None
        if os.path.exists(metrics_path):
            with open(metrics_path, 'r') as f:
                metrics_thresholds = json.load(f)
                print(f"INFO: Successfully loaded optional metrics file from {metrics_path}")
        else:
            print(f"INFO: Optional file not found at {metrics_path}. Proceeding without it.")

        return prompt_template, column_metadata, metrics_thresholds
    except FileNotFoundError as e:
        print(f"ERROR: A mandatory configuration file was not found. Details: {e}")
        return None, None, None
    except json.JSONDecodeError as e:
        print(f"ERROR: Failed to parse a JSON configuration file. Please check for syntax errors. Details: {e}")
        return None, None, None

def assemble_llm_payload(prompt_template, page_data, column_metadata, metrics_thresholds):
    """
    Assembles the final JSON payload to be sent to the LLM.
    """
    print("Assembling the final LLM payload...")
    try:
        # Start with a deep copy of the template to avoid modifying the original
        final_payload = json.loads(json.dumps(prompt_template))

        # Navigate to the instruction block where data will be injected
        instruction_block = final_payload['instructions_and_output_structure']
        input_data_section = instruction_block['input_data']

        # Inject the hardcoded page data and metadata
        input_data_section['page_data'] = page_data
        input_data_section['column_metadata'] = column_metadata.get('column_metadata', [])

        # Inject metrics thresholds if they exist
        if metrics_thresholds:
            input_data_section['key_metrics_threshold_info'] = metrics_thresholds
        
        print("Payload assembled successfully.")
        return final_payload
    except KeyError as e:
        print(f"ERROR: The prompt template is missing a required key. Could not find key: {e}")
        return None

def call_llm(payload_dict, endpoint_name):
    """
    Calls the Databricks Model Serving endpoint using the mlflow.deployments client.
    """
    print("\n" + "="*70)
    print(f"Calling LLM Endpoint: {endpoint_name}...")

    try:
        # Convert the entire payload dictionary into a single JSON string for the prompt
        prompt = json.dumps(payload_dict, indent=2)

        # Initialize the deployment client
        client = mlflow.deployments.get_deploy_client("databricks")
        
        # Structure the final payload for the Databricks LLM serving API
        inputs_payload = {
            "messages": [
                {"role": "user", "content": prompt}
            ]
        }
        
        # Call the LLM deployment endpoint
        response = client.predict(
            endpoint=endpoint_name, 
            inputs=inputs_payload
        )
        
        # Extract the summary text from the response structure
        summary = response.get('choices', [{}])[0].get('message', {}).get('content', "Error: Could not extract content from LLM response.")
        
        # Clean up the summary (e.g., strip surrounding whitespace)
        summary = summary.strip()
        
        print("✅ LLM Summary Generated Successfully.")
        return summary

    except Exception as e:
        # Handle any connection or prediction errors gracefully
        error_message = f"Error generating summary from LLM endpoint: {str(e)}"
        print(f"❌ {error_message}")
        return error_message

def main():
    """
    Main function to orchestrate the summarization pipeline.
    """
    print(f"--- Starting Summary Generation for Page '{PAGE_TO_PROCESS}' ---")

    # Construct the full path to the page's configuration folder
    page_folder = os.path.join(BASE_CONFIG_PATH, PAGE_TO_PROCESS)

    # 1. Load all configuration files
    prompt_template, column_metadata, metrics_thresholds = load_config_files(page_folder, PAGE_TO_PROCESS)
    if not prompt_template:
        print("ERROR: Failed to load configuration. Aborting.")
        return

    # 2. Use the hardcoded page data
    page_data = PAGE_DATA
    print("Using hardcoded page data.")
    
    # 3. Assemble the final payload for the LLM
    llm_payload = assemble_llm_payload(prompt_template, page_data, column_metadata, metrics_thresholds)
    if not llm_payload:
        print("ERROR: Failed to assemble LLM payload. Aborting.")
        return

    # 4. Call the LLM to get the summary
    summary = call_llm(llm_payload, LLM_ENDPOINT_NAME)
    if not summary:
        print("ERROR: Failed to generate summary. Aborting.")
        return
        
    # 5. Display the final summary
    print("\n--- FINAL SUMMARY PREVIEW ---")
    print(summary)
    print("---------------------------\n")

# Run the main pipeline function
if __name__ == "__main__":
    main()
    

# COMMAND ----------

# MAGIC %md
# MAGIC ### Comparison Summary

# COMMAND ----------

import json
import os
import mlflow.deployments
from pyspark.sql import SparkSession
# Initialize Spark session (Databricks automatically provides dbutils)
spark = SparkSession.builder.appName("SummarizationPipeline").getOrCreate()
# ---------------------------------------------------------------------------
# GLOBAL LLM INTERACTION FUNCTIONS (Reusable across all pipelines)
# ---------------------------------------------------------------------------
def load_comparative_metrics(base_path, page_name):
    """
    Loads the JSON file containing the regional comparative metrics.
    """
    print("Loading comparative data file...")
    comparative_path = os.path.join(base_path, f"{page_name}_comparative_metrics.json")

    try:
        with open(comparative_path, "r") as f:
            comparative_metrics = json.load(f)
            print(f"INFO: Successfully loaded comparative metrics from {comparative_path}")
            return comparative_metrics
    except FileNotFoundError:
        print(f"ERROR: Comparative metrics file not found at {comparative_path}. Aborting.")
        return None
    except json.JSONDecodeError as e:
        print(f"ERROR: Failed to parse comparative metrics JSON. Details: {e}")
        return None
def assemble_llm_payload(prompt_template, trial_data, comparative_data):
    """
    Assembles the final LLM prompt string by injecting trial and comparative data.
    """
    print("Assembling the final LLM payload...")
    trial_data_str = json.dumps(trial_data, indent=2)
    comparative_data_str = json.dumps(comparative_data, indent=2)
    final_prompt = prompt_template.format(
        trial_data=trial_data_str,
        comparative_data=comparative_data_str
    )
    llm_payload = {"messages": [{"role": "user", "content": final_prompt}]}
    print("Payload assembled successfully.")
    return llm_payload
def call_llm(payload_dict, endpoint_name):
    """
    Calls the Databricks Model Serving endpoint.
    """
    print("\n" + "=" * 70)
    print(f"Calling LLM Endpoint: {endpoint_name}...")
    try:
        client = mlflow.deployments.get_deploy_client("databricks")
        # Depending on the model type, you may need to use 'inputs' instead of 'messages'
        response = client.predict(endpoint=endpoint_name, inputs=payload_dict)
        summary = (
            response.get("choices", [{}])[0]
            .get("message", {})
            .get("content", "Error: Could not extract content from LLM response.")
            .strip()
        )
        print(":white_check_mark: LLM Summary Generated Successfully.")
        return summary
    except Exception as e:
        error_message = f"Error generating summary from LLM endpoint: {str(e)}"
        print(f":x: {error_message}")
        return error_message
# ---------------------------------------------------------------------------
# MAIN PIPELINE EXECUTION FUNCTION
# ---------------------------------------------------------------------------
def run_pipeline(page_name, base_data_path, llm_endpoint_name, specific_trial_data, llm_prompt_template):
    """
    Orchestrates the LLM summarization pipeline for a single trial.
    """
    print(f"--- Starting Summary Generation for Page '{page_name}' ---")
    comparative_data = load_comparative_metrics(base_data_path, page_name)
    if not comparative_data:
        print("ERROR: Failed to load comparative data. Aborting.")
        return
    llm_payload = assemble_llm_payload(llm_prompt_template, specific_trial_data, comparative_data)
    summary = call_llm(llm_payload, llm_endpoint_name)
    if summary and not summary.startswith("Error"):
        print("\n--- FINAL SUMMARY PREVIEW ---")
        print(summary)
        print("---------------------------\n")
    else:
        print("ERROR: Failed to generate summary. Aborting.")
    return summary
# ---------------------------------------------------------------------------
# CONFIGURATION SECTION
# ---------------------------------------------------------------------------
PAGE_TO_PROCESS = "page16"
BASE_DATA_PATH = "/Workspace/Users/ayush.dongardive@pivotbio.com/LLM Pipeline/Comparitive Metric"
LLM_ENDPOINT_NAME = "databricks-llama-4-maverick"
SPECIFIC_TRIAL_DATA = {
    
  "NTC: N 100%_yeild": 275.0,
  "NTC: N 100%_moisture": 27.1,
  "NTC: N Reduced_yeild": 275.0,
  "NTC: N Reduced_moisture": 26.3,
  "PVN G3: N Reduced_yeild": 274.0,
  "PVN G3: N Reduced_moisture": 26.6

}
# Use triple single quotes to avoid conflicts with double quotes in JSON-style text


## LLM Comparison Prompt for Yield Monitor Data (Text Format)

LLM_COMPARISON_PROMPT = '''
You are an Analytical AI Assistant for the Pivot Bio Sales and Agronomy Team.
You specialize in comparing trial setup, final yield performance, and moisture results against regional benchmarks (state-level averages of similar trials).
Company Context:
Pivot Bio is a U.S.-based agricultural biotechnology company that develops microbe-based nitrogen solutions to replace synthetic fertilizers.
Their flagship product, Pivot Bio PROVEN® 40, replaces synthetic nitrogen, improving yield consistency and environmental outcomes.
They run field-scale validation trials across the U.S. to demonstrate nitrogen-reduction performance.
Task:
Analyze the provided yield and moisture data for this specific trial, comparing **ALL** treatments (NTC: 100%, NTC: Reduced, PVN G3: Reduced) against the State Average metrics found in the 'comparative_data'. Generate a comprehensive comparative summary focusing on quantifiable yield and moisture gaps.
Context Summary:
"Final yield differences in this specific trial are minimal, spanning only 1 bu/ac (274.0 to 275.0 bu/ac). NTC: N Reduced yield is comparable to NTC: N 100%, indicating that the 40 lb/ac N reduction was safe in this low-stress environment."
Input Data:
- Specific Trial Data: {trial_data}
- Comparative Metrics: {comparative_data}
Output:
"title": "Yield Performance Against State Average",
"instruction": "Compare the trial's Full N Control yield (NTC: 100%) and the PVN G3: N Reduced yield against the regional average full N yield (from comparative_data). Quantify the yield difference in bu/ac (positive or negative) for both treatments relative to the state average. Also, compare the moisture content of the Full N Control against the state average moisture to set the context for trial maturity."
"title": "Efficacy of N-Reduction and PVN G3 Product",
"instruction": "Analyze the internal trial comparison: Did the PVN G3: N Reduced treatment achieve a lower moisture content than the NTC: N 100%? (The answer should be YES). Did the PVN G3 product maintain yield better or worse than the NTC: N Reduced control? (The answer is WORSE, -0.5 bu/ac gap). Use these findings to assess the product's performance in this specific environment."
"title": "Sales and Agronomic Takeaway",
"instruction": "Deliver a 2-3 sentence business interpretation. Frame the overall trial performance (NTC yields) relative to the higher/lower state average. Conclude with the main narrative: Focus the sales pitch not on a yield increase (which was negligible in this trial), but on the product's proven ability to maintain yield safety during an aggressive 40 lb/ac synthetic N reduction, differentiating it from the state's average practices."
"tone": "Analytical, comparative, and highly focused on business impact.",
"length": "250-300 words."
'''

# ---------------------------------------------------------------------------
# RUN PIPELINE
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    run_pipeline(
        page_name=PAGE_TO_PROCESS,
        base_data_path=BASE_DATA_PATH,
        llm_endpoint_name=LLM_ENDPOINT_NAME,
        specific_trial_data=SPECIFIC_TRIAL_DATA,
        llm_prompt_template=LLM_COMPARISON_PROMPT
    )


# COMMAND ----------

# MAGIC %md
# MAGIC ### Final Summarization of Page-Wise Summaries

# COMMAND ----------

# MAGIC %md
# MAGIC #### draft 1

# COMMAND ----------

# ===============================================
# :white_tick: Final Summary Generation (Python + Databricks LLM via mlflow.deployments)
# ===============================================
import os
import json
import re
import mlflow
# --- Configuration ---
LLM_ENDPOINT_NAME = "databricks-llama-4-maverick"
SUMMARIES_FOLDER_PATH = "/Workspace/Users/ayush.dongardive@pivotbio.com/LLM Pipeline/Pagewise_summary/"
# ===============================================
# :white_tick: STEP 1: Load All Page Summaries
# ===============================================
print(f":open_file_folder: Reading summary files from: {SUMMARIES_FOLDER_PATH}...")
def read_file(path, filename):
    """Tries to read the file either via open() or dbutils fallback."""
    try:
        with open(os.path.join(path, filename), "r", encoding="utf-8") as f:
            return f.read()
    except Exception:
        return dbutils.fs.head(path + filename, 1000000)  # Fallback for Databricks workspace
try:
    try:
        txt_files = [f for f in os.listdir(SUMMARIES_FOLDER_PATH) if f.endswith(".txt")]
    except Exception:
        txt_files = [f.name for f in dbutils.fs.ls(SUMMARIES_FOLDER_PATH) if f.name.endswith(".txt")]
    if not txt_files:
        raise ValueError("No .txt files found in the given path.")
    page_summaries = []
    for f in txt_files:
        match = re.search(r"page(\d+)_page_summary\.txt", f)
        page_num = int(match.group(1)) if match else 9999
        content = read_file(SUMMARIES_FOLDER_PATH, f)
        page_summaries.append((page_num, content.strip()))
    page_summaries.sort(key=lambda x: x[0])
    print(f":white_tick: Loaded {len(page_summaries)} summaries successfully.")
except Exception as e:
    print(f"""
    :x: Data loading failed.
    Error: {e}
    --- Common Fixes ---
    1. Verify path: '{SUMMARIES_FOLDER_PATH}'
    2. Ensure filenames follow 'page<number>_page_summary.txt' pattern.
    """)
    raise e
# ===============================================
# :white_tick: STEP 2: Aggregate Page Summaries
# ===============================================
print(":jigsaw: Aggregating page-wise summaries...")
aggregated_summaries = "\n\n=== END OF PAGE SUMMARY ===\n\n".join([s[1] for s in page_summaries])
print(":white_tick: Aggregation complete.")
# ===============================================
# :white_tick: STEP 3: Build Prompt for LLM
# ===============================================
FINAL_SUMMARY_PROMPT = """
**Persona:** You are an expert Marketing Content Strategist for a leading agricultural technology company, specializing in translating complex trial data into persuasive sales narratives.
**Company & Product Context:**
Our company is Pivot Bio, a U.S.-based agricultural biotechnology leader. Our flagship product, Pivot Bio PROVEN® 40, is a microbial solution designed to replace synthetic nitrogen fertilizer, offering farmers a more sustainable and potentially more profitable way to manage crop nutrition.
**Primary Goal:**
Your task is to synthesize the provided collection of page-wise summaries from a technical field report into a single, cohesive, and compelling final summary. This final document is for our **Marketing and Sales teams**, who are not scientists. It must be easy to understand, benefit-driven, and equip them with the key takeaways to drive sales conversations.
**Source Material Breakdown:**
The input you will receive contains multiple page summaries. Each summary has three parts:
1.  `ground truth`: The raw, objective data. **IGNORE THIS SECTION COMPLETELY.**
2.  `technical understanding`: Deeper analysis of the data.
3.  `sales takeaway`: High-level, benefit-oriented points.
**Your Core Task & Instructions:**
1.  **Synthesize, Don't Just Combine:** Read through all the provided page summaries. Your mission is to distill the most critical insights from the 'technical understanding' and 'sales takeaway' sections.
2.  **Focus on Sales-Critical Metrics:** Weave a narrative that highlights key performance indicators (KPIs) that matter to a farmer and our sales team. Prioritize metrics such as:
    - Yield improvements (bushels per acre)
    - Nitrogen efficiency and replacement value
    - Return on Investment (ROI) for the grower
    - Consistency of performance across conditions
    - Competitive advantages over synthetic fertilizers
3.  **Adopt the 'Sales Takeaway' Tone:** The final output's tone should be confident, clear, and persuasive, mirroring the 'sales takeaway' sections. Use strong, benefit-oriented language.
4.  **Create a Logical Flow:** Begin with an executive overview, highlight key findings and implications, and conclude with the value proposition for farmers.
5.  **Eliminate Redundancy:** Remove repeated or overlapping information for a concise document.
6.  **Length & Formatting:** 1–2 pages in length, written in paragraph form.
7.  **Strictly Adhere to Source:** Use only the provided summaries. No external info.
**Begin Synthesis.**
Here are the collected page-wise summaries:
"""
final_request = FINAL_SUMMARY_PROMPT + "\n\n" + aggregated_summaries
# ===============================================
# :white_tick: STEP 4: Define LLM Caller (Hybrid — mlflow if available, else REST)
# ===============================================
import importlib
import requests
def call_llm(payload_dict, endpoint_name):
    """
    Calls the Databricks LLM endpoint.
    Tries mlflow.deployments first; if not available, falls back to REST call.
    """
    print("\n" + "="*70)
    print(f"Calling LLM Endpoint: {endpoint_name}...")
    # Try mlflow.deployments path
    try:
        if importlib.util.find_spec("mlflow.deployments"):
            import mlflow.deployments
            print(":brain: Using mlflow.deployments client...")
            prompt = json.dumps(payload_dict, indent=2)
            client = mlflow.deployments.get_deploy_client("databricks")
            inputs_payload = {
                "messages": [
                    {"role": "system", "content": "You are a professional summarization assistant."},
                    {"role": "user", "content": prompt}
                ]
            }
            response = client.predict(endpoint=endpoint_name, inputs=inputs_payload)
            summary = (
                response.get('choices', [{}])[0]
                        .get('message', {})
                        .get('content', "")
            ).strip()
            print(":white_tick: LLM Summary Generated via mlflow.deployments.")
            return summary
        else:
            raise ImportError("mlflow.deployments not available")
    except Exception as e:
        print(f":warning: Falling back to REST API due to: {e}")
        # --- REST fallback (works universally in Databricks) ---
        try:
            workspace_url = dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiUrl().get()
            token = dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiToken().get()
            endpoint_url = f"{workspace_url}/serving-endpoints/{endpoint_name}/invocations"
            headers = {
                "Authorization": f"Bearer {token}",
                "Content-Type": "application/json"
            }
            # Convert to a simple chat-like payload
            prompt = json.dumps(payload_dict, indent=2)
            payload = {
                "messages": [
                    {"role": "system", "content": "You are a professional summarization assistant."},
                    {"role": "user", "content": prompt}
                ]
            }
            response = requests.post(endpoint_url, headers=headers, data=json.dumps(payload))
            if response.status_code != 200:
                return f":x: REST call failed: {response.status_code} {response.text}"
            response_json = response.json()
            summary = (
                response_json.get("predictions", [{}])[0].get("content", "")
                or response_json.get("output", "")
                or str(response_json)
            ).strip()
            print(":white_tick: LLM Summary Generated via REST API.")
            return summary
        except Exception as e2:
            return f":x: Both methods failed: {str(e2)}"
# ===============================================
# :white_tick: STEP 5: Generate Final Consolidated Summary
# ===============================================
print("\n:rocket: Generating the final consolidated summary using Databricks LLM...")
payload = {
    "task": "Final Summarization",
    "instruction": "Combine and synthesize all page summaries into a cohesive, marketing-oriented final summary as per the persona and guidelines.",
    "context": final_request
}
final_summary_text = call_llm(payload, LLM_ENDPOINT_NAME)
print("\n" + "="*70)
print(":receipt: FINAL CONSOLIDATED SUMMARY:\n")
print(final_summary_text[:5000])

# COMMAND ----------

# MAGIC %md
# MAGIC #### draft 2

# COMMAND ----------

# ===============================================
# Final Summary Generation (Python + Databricks LLM via mlflow.deployments)
# ===============================================
import os
import json
import re
import mlflow
# --- Configuration ---
LLM_ENDPOINT_NAME = "databricks-llama-4-maverick"
SUMMARIES_FOLDER_PATH = "/Workspace/Users/ayush.dongardive@pivotbio.com/LLM Pipeline/Pagewise_summary/"
# ===============================================
# STEP 1: Load All Page Summaries
# ===============================================
print(f":open_file_folder: Reading summary files from: {SUMMARIES_FOLDER_PATH}...")
def read_file(path, filename):
    """Tries to read the file either via open() or dbutils fallback."""
    try:
        with open(os.path.join(path, filename), "r", encoding="utf-8") as f:
            return f.read()
    except Exception:
        return dbutils.fs.head(path + filename, 1000000)  # Fallback for Databricks workspace
try:
    try:
        txt_files = [f for f in os.listdir(SUMMARIES_FOLDER_PATH) if f.endswith(".txt")]
    except Exception:
        txt_files = [f.name for f in dbutils.fs.ls(SUMMARIES_FOLDER_PATH) if f.name.endswith(".txt")]
    if not txt_files:
        raise ValueError("No .txt files found in the given path.")
    page_summaries = []
    for f in txt_files:
        match = re.search(r"page(\d+)_page_summary\.txt", f)
        page_num = int(match.group(1)) if match else 9999
        content = read_file(SUMMARIES_FOLDER_PATH, f)
        page_summaries.append((page_num, content.strip()))
    page_summaries.sort(key=lambda x: x[0])
    print(f"Loaded {len(page_summaries)} summaries successfully.")
except Exception as e:
    print(f"""
    :x: Data loading failed.
    Error: {e}
    --- Common Fixes ---
    1. Verify path: '{SUMMARIES_FOLDER_PATH}'
    2. Ensure filenames follow 'page<number>_page_summary.txt' pattern.
    """)
    raise e
# ===============================================
# STEP 2: Aggregate Page Summaries
# ===============================================
print("Aggregating page-wise summaries...")
aggregated_summaries = "\n\n=== END OF PAGE SUMMARY ===\n\n".join([s[1] for s in page_summaries])
print("Aggregation complete.")
# ===============================================
# STEP 3: Build Prompt for LLM
# ===============================================
FINAL_SUMMARY_PROMPT = """
**Persona**:
You are an expert Marketing Content Strategist for a leading agricultural technology company. You specialize in transforming complex field trial data into persuasive yet data-grounded marketing insights that help non-technical sales teams communicate value confidently.
Company & Product Context:
The company is Pivot Bio, a U.S.-based agricultural biotechnology leader. Its flagship product, Pivot Bio PROVEN® 40, is a microbial nitrogen replacement solution that delivers consistent, in-season nitrogen directly to the crop — reducing dependence on synthetic fertilizers while maintaining or improving yield and profitability.
Primary Goal:
Your task is to synthesize the provided page-wise summaries from a technical trial report into a cohesive, fact-driven, and persuasive final summary for internal marketing and sales use.
The goal is to equip the sales team with a clear, data-supported narrative showing how Pivot Bio PROVEN® 40 delivers measurable agronomic and economic value to farmers.
---
Detailed Instructions
1. Synthesize, Don’t Merge:
Distill insights from the technical understanding and sales takeaway sections. Do not simply restate content — interpret and connect the findings logically.
2. Include Quantitative Performance Metrics:
Wherever possible, weave specific metrics such as:
Yield outcomes (bushels/acre)
"%" reduction in synthetic nitrogen use (e.g., 23.5%)
Tissue nitrogen levels and biomass data
Stand count consistency (e.g., 30,451–31,264)
Soil nutrient levels (Nitrate, WEON, etc.)
Any model or environmental findings (e.g., “above-normal leaching potential”)
ROI or input savings implications for growers
3. Explain Why the Metrics Matter:
Translate technical metrics into farmer-relevant and business-relevant meaning — e.g., “maintaining 274 bu/acre with 40 lb less N translates to input cost savings without yield risk.”
4. Balance Persuasion with Credibility:
Use confident, benefit-oriented language like “demonstrates,” “validates,” and “proves,” but stay factual and avoid vague claims or excessive glorification. Let data speak.
5. Flow & Structure:
Executive Overview: Context + purpose of the trial
Key Agronomic and Technical Insights: Quantitative highlights (soil, tissue, biomass, yield, stand counts, environmental conditions)
Interpretation & Implications: What these results mean for nitrogen management, efficiency, and sustainability
Value Proposition for Farmers: Tie metrics to real-world impact — profitability, reliability, and reduced risk
6. Consistency & Clarity:
Write in full, cohesive paragraphs (no bullet lists).
Maintain professional tone with marketing polish.
Avoid redundancy or repeating metrics.
Keep length around 1.5–2 pages for a comprehensive yet digestible read.
7. Strict Data Adherence:
Use only details available in the provided summaries. Do not fabricate or speculate beyond what the data supports.
**Begin Synthesis.**
Here are the collected page-wise summaries:
"""
final_request = FINAL_SUMMARY_PROMPT + "\n\n" + aggregated_summaries
# ===============================================
# STEP 4: Define LLM Caller (Hybrid — mlflow if available, else REST)
# ===============================================
import importlib
import requests
def call_llm(payload_dict, endpoint_name):
    """
    Calls the Databricks LLM endpoint.
    Tries mlflow.deployments first; if not available, falls back to REST call.
    """
    print("\n" + "="*70)
    print(f"Calling LLM Endpoint: {endpoint_name}...")
    # Try mlflow.deployments path
    try:
        if importlib.util.find_spec("mlflow.deployments"):
            import mlflow.deployments
            print("Using mlflow.deployments client...")
            prompt = json.dumps(payload_dict, indent=2)
            client = mlflow.deployments.get_deploy_client("databricks")
            inputs_payload = {
                "messages": [
                    {"role": "system", "content": "You are a professional summarization assistant."},
                    {"role": "user", "content": prompt}
                ]
            }
            response = client.predict(endpoint=endpoint_name, inputs=inputs_payload)
            summary = (
                response.get('choices', [{}])[0]
                        .get('message', {})
                        .get('content', "")
            ).strip()
            print("LLM Summary Generated via mlflow.deployments.")
            return summary
        else:
            raise ImportError("mlflow.deployments not available")
    except Exception as e:
        print(f"Falling back to REST API due to: {e}")
        # --- REST fallback (works universally in Databricks) ---
        try:
            workspace_url = dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiUrl().get()
            token = dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiToken().get()
            endpoint_url = f"{workspace_url}/serving-endpoints/{endpoint_name}/invocations"
            headers = {
                "Authorization": f"Bearer {token}",
                "Content-Type": "application/json"
            }
            # Convert to a simple chat-like payload
            prompt = json.dumps(payload_dict, indent=2)
            payload = {
                "messages": [
                    {"role": "system", "content": "You are a professional summarization assistant."},
                    {"role": "user", "content": prompt}
                ]
            }
            response = requests.post(endpoint_url, headers=headers, data=json.dumps(payload))
            if response.status_code != 200:
                return f"REST call failed: {response.status_code} {response.text}"
            response_json = response.json()
            summary = (
                response_json.get("predictions", [{}])[0].get("content", "")
                or response_json.get("output", "")
                or str(response_json)
            ).strip()
            print("LLM Summary Generated via REST API.")
            return summary
        except Exception as e2:
            return f"Both methods failed: {str(e2)}"
# ===============================================
# STEP 5: Generate Final Consolidated Summary
# ===============================================
print("\nGenerating the final consolidated summary using Databricks LLM...")
payload = {
    "task": "Final Summarization",
    "instruction": "Combine and synthesize all page summaries into a cohesive, marketing-oriented final summary as per the persona and guidelines.",
    "context": final_request
}
final_summary_text = call_llm(payload, LLM_ENDPOINT_NAME)
print("\n" + "="*70)
print("FINAL CONSOLIDATED SUMMARY:\n")
print(final_summary_text[:5000])

# COMMAND ----------

# MAGIC %md
# MAGIC #### draft 3

# COMMAND ----------

# ===============================================
# :white_tick: Final Summary Generation (Python + Databricks LLM via mlflow.deployments)
# ===============================================
import os
import json
import re
import mlflow
# --- Configuration ---
LLM_ENDPOINT_NAME = "databricks-llama-4-maverick"
SUMMARIES_FOLDER_PATH = "/Workspace/Users/ayush.dongardive@pivotbio.com/LLM Pipeline/Pagewise_summary/"
# ===============================================
# :white_tick: STEP 1: Load All Page Summaries
# ===============================================
print(f":open_file_folder: Reading summary files from: {SUMMARIES_FOLDER_PATH}...")
def read_file(path, filename):
    """Tries to read the file either via open() or dbutils fallback."""
    try:
        with open(os.path.join(path, filename), "r", encoding="utf-8") as f:
            return f.read()
    except Exception:
        return dbutils.fs.head(path + filename, 1000000)  # Fallback for Databricks workspace
try:
    try:
        txt_files = [f for f in os.listdir(SUMMARIES_FOLDER_PATH) if f.endswith(".txt")]
    except Exception:
        txt_files = [f.name for f in dbutils.fs.ls(SUMMARIES_FOLDER_PATH) if f.name.endswith(".txt")]
    if not txt_files:
        raise ValueError("No .txt files found in the given path.")
    page_summaries = []
    for f in txt_files:
        match = re.search(r"page(\d+)_page_summary\.txt", f)
        page_num = int(match.group(1)) if match else 9999
        content = read_file(SUMMARIES_FOLDER_PATH, f)
        page_summaries.append((page_num, content.strip()))
    page_summaries.sort(key=lambda x: x[0])
    print(f":white_tick: Loaded {len(page_summaries)} summaries successfully.")
except Exception as e:
    print(f"""
    :x: Data loading failed.
    Error: {e}
    --- Common Fixes ---
    1. Verify path: '{SUMMARIES_FOLDER_PATH}'
    2. Ensure filenames follow 'page<number>_page_summary.txt' pattern.
    """)
    raise e
# ===============================================
# :white_tick: STEP 2: Aggregate Page Summaries
# ===============================================
print(":jigsaw: Aggregating page-wise summaries...")
aggregated_summaries = "\n\n=== END OF PAGE SUMMARY ===\n\n".join([s[1] for s in page_summaries])
print(":white_tick: Aggregation complete.")
# ===============================================
# :white_tick: STEP 3: Build Prompt for LLM
# ===============================================
FINAL_SUMMARY_PROMPT = """
Persona: You are an expert Agricultural Data Analyst and Technical Communicator. Your specialty is distilling complex field trial data into a clear, structured, and factual briefing document for internal teams.

Primary Goal: Your task is to synthesize the provided collection of page-wise summaries into a comprehensive, factual, and highly-detailed technical summary. This document is for our internal Sales Team. Its purpose is to equip them with all the necessary data points and context they need to understand the trial's results thoroughly before they engage with customers.

Crucial Constraint: This is NOT a sales pitch or a marketing piece. The tone must be objective, analytical, and completely free of persuasive language or marketing jargon.

Source Material Breakdown (Revised Interpretation): The input you will receive contains multiple page summaries. Each summary has three parts:

ground truth: This is your primary source for raw, objective data points (e.g., specific yield numbers, N rates, dates, soil metrics).

technical understanding: This provides context and interpretation. Use this to explain the significance of the raw numbers from an agronomic perspective.

sales takeaway: Use this ONLY as a guide to identify which data points are considered commercially important, but you MUST IGNORE its persuasive tone and language.

Your Core Task & Instructions: Your mission is to structure the data into a logical, easy-to-reference briefing document. Follow this format precisely:

1. Objective Executive Summary:

Start with a high-level, neutral summary. State the trial's core purpose, location (grower, county), crop, and the primary variable being tested (e.g., "The trial assessed the impact of replacing X lbs of synthetic N with PROVEN 40 on corn yield and plant health...").

2. Trial Parameters & Methodology:

Location & Grower: List the Grower Name, Field Name, County, and State.

Crop Information: Specify the Crop and Planting Date.

Treatment Design: Clearly list the different treatments, focusing on the Nitrogen rates for the "Full Rate" vs. the "Reduced Rate" (e.g., "Full N Control: 170 lb/ac," "PVN G3 N Reduced: 130 lb/ac").

Key Calculation: State the exact N Reduction in both lb/ac and %.

3. Key Performance Metrics & Results (Factual & Number-Driven):

This is the core of the document. Present the key findings with their specific values and units.

Final Yield: Report the bushels per acre for the key treatments (e.g., Full N Control vs. PVN G3 N Reduced). State the exact difference.

Plant Stand & Emergence: Report the stand count data to confirm seed safety and any effects on germination.

Early & Mid-Season Plant Health:

Report the fresh weight difference data (e.g., "a maximum of Xg or X% difference was observed...").

Report the key tissue analysis results (e.g., "At V16, nitrogen levels in the tissue were X%...") to show nutrient sufficiency.

Agronomic Context: Include any relevant soil analysis data from the ground truth or technical understanding sections (e.g., "Soil analysis at V3 showed a pH of 5.4 and sufficient nitrate levels...").

4. Key Analytical Takeaways:

Conclude with a bulleted list of the most significant findings from an objective, analytical perspective.

Example format:

"The 40 lb/acre reduction in synthetic nitrogen (a 23.5% decrease) in the PROVEN 40 treatment resulted in a statistically insignificant yield difference of 1.0 bu/ac compared to the full nitrogen control."

"Tissue and soil analysis confirmed that nitrogen was not a primary limiting factor for yield in this environment, even in the reduced-N treatment."

"No negative impact on plant stand or emergence was observed in any treatment."

Begin Synthesis.

Here are the collected page-wise summaries:
"""
final_request = FINAL_SUMMARY_PROMPT + "\n\n" + aggregated_summaries
# ===============================================
# :white_tick: STEP 4: Define LLM Caller (Hybrid — mlflow if available, else REST)
# ===============================================
import importlib
import requests
def call_llm(payload_dict, endpoint_name):
    """
    Calls the Databricks LLM endpoint.
    Tries mlflow.deployments first; if not available, falls back to REST call.
    """
    print("\n" + "="*70)
    print(f"Calling LLM Endpoint: {endpoint_name}...")
    # Try mlflow.deployments path
    try:
        if importlib.util.find_spec("mlflow.deployments"):
            import mlflow.deployments
            print(":brain: Using mlflow.deployments client...")
            prompt = json.dumps(payload_dict, indent=2)
            client = mlflow.deployments.get_deploy_client("databricks")
            inputs_payload = {
                "messages": [
                    {"role": "system", "content": "You are a professional summarization assistant."},
                    {"role": "user", "content": prompt}
                ]
            }
            response = client.predict(endpoint=endpoint_name, inputs=inputs_payload)
            summary = (
                response.get('choices', [{}])[0]
                        .get('message', {})
                        .get('content', "")
            ).strip()
            print(":white_tick: LLM Summary Generated via mlflow.deployments.")
            return summary
        else:
            raise ImportError("mlflow.deployments not available")
    except Exception as e:
        print(f":warning: Falling back to REST API due to: {e}")
        # --- REST fallback (works universally in Databricks) ---
        try:
            workspace_url = dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiUrl().get()
            token = dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiToken().get()
            endpoint_url = f"{workspace_url}/serving-endpoints/{endpoint_name}/invocations"
            headers = {
                "Authorization": f"Bearer {token}",
                "Content-Type": "application/json"
            }
            # Convert to a simple chat-like payload
            prompt = json.dumps(payload_dict, indent=2)
            payload = {
                "messages": [
                    {"role": "system", "content": "You are a professional summarization assistant."},
                    {"role": "user", "content": prompt}
                ]
            }
            response = requests.post(endpoint_url, headers=headers, data=json.dumps(payload))
            if response.status_code != 200:
                return f":x: REST call failed: {response.status_code} {response.text}"
            response_json = response.json()
            summary = (
                response_json.get("predictions", [{}])[0].get("content", "")
                or response_json.get("output", "")
                or str(response_json)
            ).strip()
            print(":white_tick: LLM Summary Generated via REST API.")
            return summary
        except Exception as e2:
            return f":x: Both methods failed: {str(e2)}"
# ===============================================
# :white_tick: STEP 5: Generate Final Consolidated Summary
# ===============================================
print("\n:rocket: Generating the final consolidated summary using Databricks LLM...")
payload = {
    "task": "Final Summarization",
    "instruction": "Combine and synthesize all page summaries into a cohesive, marketing-oriented final summary as per the persona and guidelines.",
    "context": final_request
}
final_summary_text = call_llm(payload, LLM_ENDPOINT_NAME)
print("\n" + "="*70)
print(":receipt: FINAL CONSOLIDATED SUMMARY:\n")
print(final_summary_text[:5000])

# COMMAND ----------

# MAGIC %md
# MAGIC # Comparison Summary 

# COMMAND ----------

# ===============================================
# :white_tick: Final Summary Generation (Python + Databricks LLM via mlflow.deployments)
# ===============================================
import os
import json
import re
import mlflow
# --- Configuration ---
LLM_ENDPOINT_NAME = "databricks-llama-4-maverick"
SUMMARIES_FOLDER_PATH = "/Workspace/Users/ayush.dongardive@pivotbio.com/LLM Pipeline/Comparison_summary"
# ===============================================
# :white_tick: STEP 1: Load All Page Summaries
# ===============================================
print(f":open_file_folder: Reading summary files from: {SUMMARIES_FOLDER_PATH}...")
def read_file(path, filename):
    """Tries to read the file either via open() or dbutils fallback."""
    try:
        with open(os.path.join(path, filename), "r", encoding="utf-8") as f:
            return f.read()
    except Exception:
        return dbutils.fs.head(path + filename, 1000000)  # Fallback for Databricks workspace
try:
    try:
        txt_files = [f for f in os.listdir(SUMMARIES_FOLDER_PATH) if f.endswith(".txt")]
    except Exception:
        txt_files = [f.name for f in dbutils.fs.ls(SUMMARIES_FOLDER_PATH) if f.name.endswith(".txt")]
    if not txt_files:
        raise ValueError("No .txt files found in the given path.")
    page_summaries = []
    for f in txt_files:
        match = re.search(r"page(\d+)_page_summary\.txt", f)
        page_num = int(match.group(1)) if match else 9999
        content = read_file(SUMMARIES_FOLDER_PATH, f)
        page_summaries.append((page_num, content.strip()))
    page_summaries.sort(key=lambda x: x[0])
    print(f":white_tick: Loaded {len(page_summaries)} summaries successfully.")
except Exception as e:
    print(f"""
    :x: Data loading failed.
    Error: {e}
    --- Common Fixes ---
    1. Verify path: '{SUMMARIES_FOLDER_PATH}'
    2. Ensure filenames follow 'page<number>_page_summary.txt' pattern.
    """)
    raise e
# ===============================================
# :white_tick: STEP 2: Aggregate Page Summaries
# ===============================================
print(":jigsaw: Aggregating page-wise summaries...")
aggregated_summaries = "\n\n=== END OF PAGE SUMMARY ===\n\n".join([s[1] for s in page_summaries])
print(":white_tick: Aggregation complete.")
# ===============================================
# :white_tick: STEP 3: Build Prompt for LLM
# ===============================================
FINAL_SUMMARY_PROMPT = """
**Persona:** You are an expert Marketing Content Strategist for a leading agricultural technology company, specializing in translating complex trial data into persuasive sales narratives.
**Company & Product Context:**
Our company is Pivot Bio, a U.S.-based agricultural biotechnology leader. Our flagship product, Pivot Bio PROVEN® 40, is a microbial solution designed to replace synthetic nitrogen fertilizer, offering farmers a more sustainable and potentially more profitable way to manage crop nutrition.
**Primary Goal:**
Your task is to synthesize the provided collection of page-wise summaries from a technical field report into a single, cohesive, and compelling final summary. This final document is for our **Marketing and Sales teams**, who are not scientists. It must be easy to understand, benefit-driven, and equip them with the key takeaways to drive sales conversations.
**Source Material Breakdown:**
The input you will receive contains multiple page summaries. Each summary has three parts:
1.  `ground truth`: The raw, objective data. **IGNORE THIS SECTION COMPLETELY.**
2.  `technical understanding`: Deeper analysis of the data.
3.  `sales takeaway`: High-level, benefit-oriented points.
**Your Core Task & Instructions:**
1.  **Synthesize, Don't Just Combine:** Read through all the provided page summaries. Your mission is to distill the most critical insights from the 'technical understanding' and 'sales takeaway' sections.
2.  **Focus on Sales-Critical Metrics:** Weave a narrative that highlights key performance indicators (KPIs) that matter to a farmer and our sales team. Prioritize metrics such as:
    - Yield improvements (bushels per acre)
    - Nitrogen efficiency and replacement value
    - Return on Investment (ROI) for the grower
    - Consistency of performance across conditions
    - Competitive advantages over synthetic fertilizers
3.  **Adopt the 'Sales Takeaway' Tone:** The final output's tone should be confident, clear, and persuasive, mirroring the 'sales takeaway' sections. Use strong, benefit-oriented language.
4.  **Create a Logical Flow:** Begin with an executive overview, highlight key findings and implications, and conclude with the value proposition for farmers.
5.  **Eliminate Redundancy:** Remove repeated or overlapping information for a concise document.
6.  **Length & Formatting:** 1–2 pages in length, written in paragraph form.
7.  **Strictly Adhere to Source:** Use only the provided summaries. No external info.
**Begin Synthesis.**
Here are the collected page-wise summaries:
"""
final_request = FINAL_SUMMARY_PROMPT + "\n\n" + aggregated_summaries
# ===============================================
# :white_tick: STEP 4: Define LLM Caller (Hybrid — mlflow if available, else REST)
# ===============================================
import importlib
import requests
def call_llm(payload_dict, endpoint_name):
    """
    Calls the Databricks LLM endpoint.
    Tries mlflow.deployments first; if not available, falls back to REST call.
    """
    print("\n" + "="*70)
    print(f"Calling LLM Endpoint: {endpoint_name}...")
    # Try mlflow.deployments path
    try:
        if importlib.util.find_spec("mlflow.deployments"):
            import mlflow.deployments
            print(":brain: Using mlflow.deployments client...")
            prompt = json.dumps(payload_dict, indent=2)
            client = mlflow.deployments.get_deploy_client("databricks")
            inputs_payload = {
                "messages": [
                    {"role": "system", "content": "You are a professional summarization assistant."},
                    {"role": "user", "content": prompt}
                ]
            }
            response = client.predict(endpoint=endpoint_name, inputs=inputs_payload)
            summary = (
                response.get('choices', [{}])[0]
                        .get('message', {})
                        .get('content', "")
            ).strip()
            print(":white_tick: LLM Summary Generated via mlflow.deployments.")
            return summary
        else:
            raise ImportError("mlflow.deployments not available")
    except Exception as e:
        print(f":warning: Falling back to REST API due to: {e}")
        # --- REST fallback (works universally in Databricks) ---
        try:
            workspace_url = dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiUrl().get()
            token = dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiToken().get()
            endpoint_url = f"{workspace_url}/serving-endpoints/{endpoint_name}/invocations"
            headers = {
                "Authorization": f"Bearer {token}",
                "Content-Type": "application/json"
            }
            # Convert to a simple chat-like payload
            prompt = json.dumps(payload_dict, indent=2)
            payload = {
                "messages": [
                    {"role": "system", "content": "You are a professional summarization assistant."},
                    {"role": "user", "content": prompt}
                ]
            }
            response = requests.post(endpoint_url, headers=headers, data=json.dumps(payload))
            if response.status_code != 200:
                return f":x: REST call failed: {response.status_code} {response.text}"
            response_json = response.json()
            summary = (
                response_json.get("predictions", [{}])[0].get("content", "")
                or response_json.get("output", "")
                or str(response_json)
            ).strip()
            print(":white_tick: LLM Summary Generated via REST API.")
            return summary
        except Exception as e2:
            return f":x: Both methods failed: {str(e2)}"
# ===============================================
# :white_tick: STEP 5: Generate Final Consolidated Summary
# ===============================================
print("\n:rocket: Generating the final consolidated summary using Databricks LLM...")
payload = {
    "task": "Final Summarization",
    "instruction": "Combine and synthesize all page summaries into a cohesive, marketing-oriented final summary as per the persona and guidelines.",
    "context": final_request
}
final_summary_text = call_llm(payload, LLM_ENDPOINT_NAME)
print("\n" + "="*70)
print(":receipt: FINAL CONSOLIDATED SUMMARY:\n")
print(final_summary_text[:5000])

# COMMAND ----------

# MAGIC %md
# MAGIC draft 2

# COMMAND ----------

# ===============================================
# :white_tick: Final Summary Generation (Python + Databricks LLM via mlflow.deployments)
# ===============================================
import os
import json
import re
import mlflow
# --- Configuration ---
LLM_ENDPOINT_NAME = "databricks-llama-4-maverick"
SUMMARIES_FOLDER_PATH = "/Workspace/Users/ayush.dongardive@pivotbio.com/LLM Pipeline/Comparison_summary"
# ===============================================
# :white_tick: STEP 1: Load All Page Summaries
# ===============================================
print(f":open_file_folder: Reading summary files from: {SUMMARIES_FOLDER_PATH}...")
def read_file(path, filename):
    """Tries to read the file either via open() or dbutils fallback."""
    try:
        with open(os.path.join(path, filename), "r", encoding="utf-8") as f:
            return f.read()
    except Exception:
        return dbutils.fs.head(path + filename, 1000000)  # Fallback for Databricks workspace
try:
    try:
        txt_files = [f for f in os.listdir(SUMMARIES_FOLDER_PATH) if f.endswith(".txt")]
    except Exception:
        txt_files = [f.name for f in dbutils.fs.ls(SUMMARIES_FOLDER_PATH) if f.name.endswith(".txt")]
    if not txt_files:
        raise ValueError("No .txt files found in the given path.")
    page_summaries = []
    for f in txt_files:
        match = re.search(r"page(\d+)_page_summary\.txt", f)
        page_num = int(match.group(1)) if match else 9999
        content = read_file(SUMMARIES_FOLDER_PATH, f)
        page_summaries.append((page_num, content.strip()))
    page_summaries.sort(key=lambda x: x[0])
    print(f":white_tick: Loaded {len(page_summaries)} summaries successfully.")
except Exception as e:
    print(f"""
    :x: Data loading failed.
    Error: {e}
    --- Common Fixes ---
    1. Verify path: '{SUMMARIES_FOLDER_PATH}'
    2. Ensure filenames follow 'page<number>_page_summary.txt' pattern.
    """)
    raise e
# ===============================================
# :white_tick: STEP 2: Aggregate Page Summaries
# ===============================================
print(":jigsaw: Aggregating page-wise summaries...")
aggregated_summaries = "\n\n=== END OF PAGE SUMMARY ===\n\n".join([s[1] for s in page_summaries])
print(":white_tick: Aggregation complete.")
# ===============================================
# :white_tick: STEP 3: Build Prompt for LLM
# ===============================================
FINAL_SUMMARY_PROMPT = """
**Persona:** You are an expert Agricultural Marketing Strategist for Pivot Bio — a leader in microbial nitrogen innovation. Your expertise lies in interpreting agronomic trial data and producing actionable, insight-driven summaries that guide sales conversations.
**Objective:**
Create a *Final Comparative Summary* for this specific trial site that shows how it performs relative to other regional sites. The goal is to equip the **Sales and Marketing team** with a clear understanding of performance, positioning, and value.
**Input Context:**
You will receive a collection of page-wise summaries from a field trial report. Each summary may include different kinds of information — some may emphasize sales takeaways, while others may contain analytical or technical details. Use **any available content** that provides meaningful performance insights. Do not assume a fixed structure.
**Your Core Task:**
1. **Comparative Evaluation:**
   - Assess how this trial site performs compared to other trial sites in the same region.
   - Clearly indicate whether the performance is **Above Average**, **In Line (Neutral)**, or **Below Average** versus the regional benchmark.
   - Support your conclusion with qualitative or quantitative cues found in the summaries.
2. **Key Focus Metrics:**
   - **Harvest Yield (bu/acre)** — comparative productivity.
   - **Nitrogen Use Efficiency / Replacement Value** — ability to replace or reduce synthetic fertilizer.
   - **Soil Nitrogen Retention or Health** — sustainability advantage.
3. **Highlight Key Insights:**
   - Identify patterns or drivers (e.g., weather, soil type, hybrid, or management practice) that influenced performance.
   - Point out any outliers, unexpected strengths, or improvement areas.
4. **Tone & Audience:**
   - Write for a **sales audience** — confident, clear, and insight-focused.
   - Use persuasive, benefit-oriented language.
   - Make sure the conclusion is actionable: what should the sales rep know or say about this trial’s performance?
5. **Structure:**
   - **Executive Overview:** 3–5 lines summarizing overall comparative performance.
   - **Detailed Insights:** Paragraphs on yield, nitrogen, ROI, consistency, and notable factors.
   - **Final Takeaway:** 1 paragraph summarizing the product’s perceived value and implications for the region.
6. **Style & Length:**
   - 1–2 pages max.
   - Use plain, professional language.
   - No external data — rely only on provided summaries.
**Now, based on the provided inputs, synthesize the final comparative summary:**
"""
final_request = FINAL_SUMMARY_PROMPT + "\n\n" + aggregated_summaries
# ===============================================
# :white_tick: STEP 4: Define LLM Caller (Hybrid — mlflow if available, else REST)
# ===============================================
import importlib
import requests
def call_llm(payload_dict, endpoint_name):
    """
    Calls the Databricks LLM endpoint.
    Tries mlflow.deployments first; if not available, falls back to REST call.
    """
    print("\n" + "="*70)
    print(f"Calling LLM Endpoint: {endpoint_name}...")
    # Try mlflow.deployments path
    try:
        if importlib.util.find_spec("mlflow.deployments"):
            import mlflow.deployments
            print(":brain: Using mlflow.deployments client...")
            prompt = json.dumps(payload_dict, indent=2)
            client = mlflow.deployments.get_deploy_client("databricks")
            inputs_payload = {
                "messages": [
                    {"role": "system", "content": "You are a professional summarization assistant."},
                    {"role": "user", "content": prompt}
                ]
            }
            response = client.predict(endpoint=endpoint_name, inputs=inputs_payload)
            summary = (
                response.get('choices', [{}])[0]
                        .get('message', {})
                        .get('content', "")
            ).strip()
            print(":white_tick: LLM Summary Generated via mlflow.deployments.")
            return summary
        else:
            raise ImportError("mlflow.deployments not available")
    except Exception as e:
        print(f":warning: Falling back to REST API due to: {e}")
        # --- REST fallback (works universally in Databricks) ---
        try:
            workspace_url = dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiUrl().get()
            token = dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiToken().get()
            endpoint_url = f"{workspace_url}/serving-endpoints/{endpoint_name}/invocations"
            headers = {
                "Authorization": f"Bearer {token}",
                "Content-Type": "application/json"
            }
            # Convert to a simple chat-like payload
            prompt = json.dumps(payload_dict, indent=2)
            payload = {
                "messages": [
                    {"role": "system", "content": "You are a professional summarization assistant."},
                    {"role": "user", "content": prompt}
                ]
            }
            response = requests.post(endpoint_url, headers=headers, data=json.dumps(payload))
            if response.status_code != 200:
                return f":x: REST call failed: {response.status_code} {response.text}"
            response_json = response.json()
            summary = (
                response_json.get("predictions", [{}])[0].get("content", "")
                or response_json.get("output", "")
                or str(response_json)
            ).strip()
            print(":white_tick: LLM Summary Generated via REST API.")
            return summary
        except Exception as e2:
            return f":x: Both methods failed: {str(e2)}"
# ===============================================
# :white_tick: STEP 5: Generate Final Consolidated Summary
# ===============================================
print("\n:rocket: Generating the final consolidated summary using Databricks LLM...")
payload = {
    "task": "Final Summarization",
    "instruction": "Combine and synthesize all page summaries into a cohesive, marketing-oriented final summary as per the persona and guidelines.",
    "context": final_request
}
final_summary_text = call_llm(payload, LLM_ENDPOINT_NAME)
print("\n" + "="*70)
print(":receipt: FINAL CONSOLIDATED SUMMARY:\n")
print(final_summary_text[:5000])