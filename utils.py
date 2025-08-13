import openai
import os
import json
import pandas as pd
import matplotlib.pyplot as plt
import uuid
import io
import traceback

openai.api_key = os.getenv("OPENAI_API_KEY")

# Load the system prompt (optional)
def load_system_prompt():
    try:
        with open("prompts/system_prompt.txt", "r") as f:
            return f.read()
    except:
        return "You are a helpful data science assistant."

async def run_analysis_with_llm(question: str, data_path: str = None):
    try:
        # Prepare context
        system_prompt = load_system_prompt()
        user_prompt = f"Question:\n{question.strip()}"
        if data_path:
            user_prompt += f"\n\nThe user has uploaded a data file: {data_path}"

        # Call GPT-4 to generate analysis code
        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.3
        )

        code = response['choices'][0]['message']['content']

        # Securely execute the generated code
        exec_env = {
            "pd": pd,
            "plt": plt,
            "__builtins__": {
                "print": print,
                "len": len,
                "str": str,
                "int": int,
                "float": float,
                "list": list,
                "dict": dict,
                "range": range,
                "sum": sum,
                "zip": zip,
                "enumerate": enumerate,
            }
        }

        # Load data if needed
        if data_path:
            if data_path.endswith(".csv"):
                exec_env["df"] = pd.read_csv(data_path)
            elif data_path.endswith(".json"):
                exec_env["df"] = pd.read_json(data_path)

        # Execute the code
        output_path = f"uploads/output_{uuid.uuid4().hex}.png"
        exec_env["output_path"] = output_path

        exec(code, exec_env)

        # Return output file or printed result
        if os.path.exists(output_path):
            return output_path

        return "Task completed successfully."

    except Exception as e:
        return traceback.format_exc()
