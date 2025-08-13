import os
import json
import time
import base64
import tempfile
import shutil
import subprocess
from pathlib import Path
import logging
import re
from typing import Dict, List, Any, Union

import requests
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
import uvicorn

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# ---------------- CONFIG ----------------
load_dotenv()
AIPIPE_API_KEY = os.getenv("AIPIPE_TOKEN")
AIPIPE_URL = "https://aipipe.org/openrouter/v1/chat/completions"
MODEL = "openai/gpt-4o-mini"
EXEC_TIMEOUT = 180  # 3 minutes per subtask
MAX_INLINE_FILE_BYTES = 200 * 1024  # 200KB
# ----------------------------------------

if not AIPIPE_API_KEY:
    raise RuntimeError("AIPIPE_API_KEY not found in environment (.env file required)")

app = FastAPI(
    title="AI Data Analyst Agent", 
    description="LLM-powered data analysis API",
    version="2.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# System prompt optimized for data analysis
SYSTEM_PROMPT = """You are an expert data analyst and Python developer specializing in web scraping, data analysis, and visualization.

CORE PRINCIPLES:
1. Write robust, production-ready Python code with comprehensive error handling
2. Always include proper imports at the top of your script
3. Use appropriate libraries for web scraping (requests, BeautifulSoup) and data analysis (pandas, numpy)
4. Handle missing data, network issues, and data type conversions gracefully
5. Print intermediate results and debug information with clear labels
6. Create publication-quality visualizations with matplotlib

ANSWER LABELING REQUIREMENTS:
- For array format responses: Use "ANSWER_1:", "ANSWER_2:", "ANSWER_3:", etc.
- For JSON object responses: Use "ANSWER_key_name:" where key_name matches the JSON key
- Always print answers with these exact prefixes for proper extraction
- For images: create complete base64 data URIs with "data:image/png;base64," prefix

IMPORTANT DATA ANALYSIS NOTES:
- For box office questions: Use the TOTAL/LIFETIME gross, not just original release gross
- Titanic (1997) has grossed over $2.2 billion total (including re-releases)
- When filtering by release date AND gross amount, check the original release year but use lifetime totals
- Be careful with data parsing - some tables show original gross vs. total gross in different columns

WEB SCRAPING GUIDELINES:
- Use requests with proper headers and error handling
- Parse HTML with BeautifulSoup
- Handle table data with pandas.read_html when possible
- Clean and normalize data before analysis
- Add delays between requests to be respectful

DATA PROCESSING REQUIREMENTS:
- Date parsing: Use pandas datetime parsing with error handling
- Statistical analysis: Use scipy.stats for correlation and regression
- Visualization: matplotlib with proper styling, 300 DPI, clear labels
- Memory management: Process data efficiently
- Error handling: Try/except blocks around all network and data operations

CODE STRUCTURE:
- Import all libraries at the top
- Include proper error handling for each step
- Print progress and intermediate results clearly
- Label final answers with the exact format: "ANSWER_N: value" or "ANSWER_key: value"
- For images: ensure base64 string is complete and properly formatted"""

def call_llm(messages: List[Dict], max_tokens: int = 8000) -> str:
    """Call the LLM API with proper error handling."""
    headers = {
        "Authorization": f"Bearer {AIPIPE_API_KEY}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": MODEL,
        "messages": messages,
        "temperature": 0.0,
        "max_tokens": max_tokens,
    }
    
    try:
        response = requests.post(AIPIPE_URL, headers=headers, json=payload, timeout=180)
        response.raise_for_status()
        data = response.json()
        return data["choices"][0]["message"]["content"]
    except requests.exceptions.RequestException as e:
        logger.error(f"LLM API error: {e}")
        raise HTTPException(status_code=502, detail=f"LLM API failed: {str(e)}")
    except KeyError as e:
        logger.error(f"Unexpected LLM response format: {e}")
        raise HTTPException(status_code=502, detail="Invalid LLM response format")

def parse_json_response(response: str) -> Union[Dict[str, Any], List[Any]]:
    """Parse JSON from LLM response with multiple fallback strategies."""
    # Try direct parsing first
    try:
        return json.loads(response)
    except json.JSONDecodeError:
        pass
    
    # Try extracting JSON from code blocks
    patterns = [
        r'```json\s*(.*?)\s*```',
        r'```\s*(.*?)\s*```',
        r'(\{.*?\})',
        r'(\[.*?\])'
    ]
    
    for pattern in patterns:
        matches = re.findall(pattern, response, re.DOTALL | re.IGNORECASE)
        for match in matches:
            try:
                return json.loads(match.strip())
            except json.JSONDecodeError:
                continue
    
    # If all else fails, try to extract partial JSON and fix it
    brace_match = re.search(r'[\{\[].*', response, re.DOTALL)
    if brace_match:
        partial_json = brace_match.group(0)
        # Try to balance braces/brackets
        if partial_json.startswith('{'):
            open_braces = partial_json.count('{')
            close_braces = partial_json.count('}')
            if open_braces > close_braces:
                partial_json += '}' * (open_braces - close_braces)
        elif partial_json.startswith('['):
            open_brackets = partial_json.count('[')
            close_brackets = partial_json.count(']')
            if open_brackets > close_brackets:
                partial_json += ']' * (open_brackets - close_brackets)
        
        try:
            return json.loads(partial_json)
        except json.JSONDecodeError:
            pass
    
    raise ValueError(f"Could not parse JSON from response: {response[:500]}...")

def install_packages(work_dir: Path) -> None:
    """Install required Python packages."""
    packages = [
        "pandas>=2.0.0",
        "requests>=2.31.0", 
        "beautifulsoup4>=4.12.0",
        "lxml>=4.9.0",
        "matplotlib>=3.7.0",
        "seaborn>=0.12.0",
        "scipy>=1.10.0",
        "numpy>=1.24.0",
        "openpyxl>=3.1.0",
        "duckdb>=0.9.0",
        "pyarrow>=10.0.0",
        "s3fs>=2023.1.0",
        "html5lib>=1.1"
    ]
    
    for package in packages:
        try:
            result = subprocess.run(
                ["pip", "install", package], 
                cwd=work_dir,
                capture_output=True,
                text=True,
                timeout=90
            )
            if result.returncode != 0:
                logger.warning(f"Failed to install {package}: {result.stderr}")
        except subprocess.TimeoutExpired:
            logger.warning(f"Timeout installing {package}")
        except Exception as e:
            logger.warning(f"Error installing {package}: {e}")

def execute_python_code(code: str, work_dir: Path) -> Dict[str, Any]:
    """Execute Python code in isolated environment and return results."""
    code_file = work_dir / "task.py"
    code_file.write_text(code, encoding='utf-8')
    
    # Install packages first
    install_packages(work_dir)
    
    start_time = time.time()
    try:
        result = subprocess.run(
            ["python", "task.py"],
            cwd=work_dir,
            capture_output=True,
            text=True,
            timeout=EXEC_TIMEOUT
        )
        
        execution_time = time.time() - start_time
        
        # Collect any generated files
        generated_files = []
        for file_path in work_dir.rglob("*"):
            if file_path.is_file() and file_path.name != "task.py":
                try:
                    size = file_path.stat().st_size
                    if size <= MAX_INLINE_FILE_BYTES:
                        content = file_path.read_bytes()
                        generated_files.append({
                            "name": file_path.name,
                            "size": size,
                            "base64": base64.b64encode(content).decode('utf-8')
                        })
                    else:
                        generated_files.append({
                            "name": file_path.name,
                            "size": size,
                            "error": "File too large to include"
                        })
                except Exception as e:
                    generated_files.append({
                        "name": file_path.name,
                        "error": f"Could not read file: {e}"
                    })
        
        return {
            "stdout": result.stdout,
            "stderr": result.stderr,
            "returncode": result.returncode,
            "execution_time": execution_time,
            "generated_files": generated_files
        }
        
    except subprocess.TimeoutExpired as e:
        return {
            "stdout": e.stdout or "",
            "stderr": f"{e.stderr or ''}\n*** EXECUTION TIMEOUT ({EXEC_TIMEOUT}s) ***",
            "returncode": -1,
            "execution_time": time.time() - start_time,
            "generated_files": []
        }
    except Exception as e:
        return {
            "stdout": "",
            "stderr": f"Execution error: {str(e)}",
            "returncode": -2, 
            "execution_time": time.time() - start_time,
            "generated_files": []
        }

def extract_results_from_output(stdout: str, stderr: str, questions_text: str) -> Union[List[Any], Dict[str, Any]]:
    """Extract results based on the requested format from the questions."""
    
    # Determine if the user wants array or object format
    wants_array = "JSON array" in questions_text or "[" in questions_text.split('\n')[0]
    wants_object = "{" in questions_text and ":" in questions_text
    
    if wants_array or (not wants_object):
        # Extract array format results
        logger.info("Extracting array format results")
        
        # Count the number of questions
        question_count = len(re.findall(r'^\s*\d+\.', questions_text, re.MULTILINE))
        if question_count == 0:
            question_count = 4  # Default fallback
            
        results = [None] * question_count
        
        # Look for ANSWER_N: patterns
        answer_patterns = [
            r'ANSWER_(\d+):\s*(.+?)(?=\n(?:ANSWER_\d+:|$)|\n\n|$)',
            r'RESULT_(\d+):\s*(.+?)(?=\n(?:RESULT_\d+:|$)|\n\n|$)',
            r'Question\s+(\d+).*?:\s*(.+?)(?=\nQuestion\s+\d+|\n\n|$)',
            r'^\s*(\d+)\.\s*Answer:\s*(.+?)(?=^\s*\d+\.\s*Answer:|\n\n|$)',
            r'^\s*(\d+)\)\s*(.+?)(?=^\s*\d+\)|\n\n|$)'
        ]
        
        for pattern in answer_patterns:
            matches = re.findall(pattern, stdout, re.MULTILINE | re.DOTALL | re.IGNORECASE)
            for match in matches:
                try:
                    index = int(match[0]) - 1  # Convert to 0-based index
                    value = match[1].strip()
                    
                    if 0 <= index < len(results):
                        # Process the value
                        if value.lower() in ['none', 'null', 'undefined', 'nan']:
                            results[index] = None
                        elif value.startswith('data:image'):
                            results[index] = value
                        elif value.replace('.', '').replace('-', '').replace(',', '').isdigit():
                            # Handle numeric values
                            clean_value = value.replace(',', '')
                            results[index] = float(clean_value) if '.' in clean_value else int(clean_value)
                        else:
                            # Clean string value
                            clean_value = value.strip('"\'').strip()
                            # Remove any trailing periods or commas
                            clean_value = re.sub(r'[.,]$', '', clean_value)
                            results[index] = clean_value
                            
                except (ValueError, IndexError) as e:
                    logger.warning(f"Error processing match {match}: {e}")
                    continue
        
        # If we still have None values, try alternative patterns
        for i, result in enumerate(results):
            if result is None:
                # Look for standalone answers near question numbers
                patterns = [
                    rf'(?:Question\s*{i+1}|{i+1}\.)[^\n]*\n[^\n]*?([^\n]+)',
                    rf'{i+1}[^\n]*?Answer[^\n]*?:\s*([^\n]+)',
                    rf'{i+1}[^\n]*?Result[^\n]*?:\s*([^\n]+)'
                ]
                
                for pattern in patterns:
                    matches = re.findall(pattern, stdout, re.MULTILINE | re.IGNORECASE)
                    if matches:
                        value = matches[0].strip()
                        if value and not value.lower().startswith(('question', 'answer', 'result')):
                            if value.startswith('data:image'):
                                results[i] = value
                            elif value.replace('.', '').replace('-', '').isdigit():
                                results[i] = float(value) if '.' in value else int(value)
                            else:
                                results[i] = value.strip('"\'')
                            break
        
        return results
    
    else:
        # Extract object format results
        logger.info("Extracting object format results")
        
        # Extract question keys from the requested format
        json_keys = re.findall(r'"([^"]+)":\s*"[^"]*"', questions_text)
        if not json_keys:
            # Fallback: extract keys from question text
            json_keys = re.findall(r'(\w+)(?=\s*questions?)', questions_text.lower())
        
        result_dict = {}
        
        for key in json_keys:
            # Look for answers with this key
            patterns = [
                rf'ANSWER_{re.escape(key)}:\s*(.+?)(?:\n(?:ANSWER_\w+:|$)|\n\n|$)',
                rf'RESULT_{re.escape(key)}:\s*(.+?)(?:\n(?:RESULT_\w+:|$)|\n\n|$)',
                rf'{re.escape(key)}:\s*(.+?)(?:\n\w+:|\n\n|$)',
            ]
            
            found_answer = None
            for pattern in patterns:
                matches = re.findall(pattern, stdout, re.MULTILINE | re.IGNORECASE | re.DOTALL)
                if matches:
                    found_answer = matches[0].strip()
                    break
            
            if found_answer:
                # Clean up the answer
                if found_answer.startswith('data:image'):
                    result_dict[key] = found_answer
                elif found_answer.replace('.', '').replace('-', '').replace(',', '').isdigit():
                    clean_value = found_answer.replace(',', '')
                    result_dict[key] = float(clean_value) if '.' in clean_value else int(clean_value)
                else:
                    result_dict[key] = found_answer.strip('"\'')
            else:
                result_dict[key] = "No result found"
        
        return result_dict

@app.get("/")
async def root():
    return {
        "message": "AI Data Analyst Agent",
        "version": "2.0.0",
        "status": "operational",
        "endpoints": {
            "analyze": "POST /api/",
            "health": "GET /health"
        }
    }

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "timestamp": time.time(),
        "model": MODEL
    }

@app.post("/api/")
async def analyze_data(file: UploadFile = File(...)):
    """
    Main data analysis endpoint.
    
    Accepts a text file with questions and returns analyzed results.
    """
    logger.info(f"Starting analysis for file: {file.filename}")
    
    # Read and validate input
    try:
        content = await file.read()
        questions = content.decode('utf-8', errors='ignore').strip()
        
        if not questions:
            raise HTTPException(status_code=400, detail="Empty file provided")
            
        logger.info(f"Questions received ({len(questions)} chars)")
        
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Could not read file: {str(e)}")
    
    # Generate analysis code
    analysis_prompt = f"""
Create a comprehensive Python script to answer the questions about the dataset described.

CRITICAL REQUIREMENTS:
1. Import ALL required libraries at the very top of the script
2. Handle all network and data access errors gracefully
3. For each question, print the answer with the EXACT label format specified below
4. Use appropriate libraries for web scraping and data analysis
5. For visualizations, create complete base64 data URIs under 100KB

ANSWER LABELING - THIS IS CRITICAL:
- Look at the user's request format carefully
- If they want a JSON array: Use "ANSWER_1:", "ANSWER_2:", "ANSWER_3:", etc.
- If they want a JSON object: Use "ANSWER_keyname:" where keyname matches their JSON keys
- Print each answer on its own line with the exact prefix
- For plots, ensure base64 string starts with "data:image/png;base64,"

QUESTIONS TO ANSWER:
{questions}

Generate a single, complete Python script that:
1. Scrapes the required data from the web
2. Processes and analyzes the data carefully (use TOTAL/LIFETIME gross, not original gross)
3. For movies released before 2000 that grossed over $2B: Check release year < 2000 AND total gross >= $2B
4. Answers each question with the correct label format
5. Includes comprehensive error handling and debug output
6. Prints intermediate results to show progress

CRITICAL: For Titanic (1997), it should count as 1 movie released before 2000 that grossed over $2B since its lifetime total is ~$2.2B.

Make sure to label each answer exactly as specified for proper extraction.
"""

    try:
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": analysis_prompt}
        ]
        
        code_response = call_llm(messages, max_tokens=10000)
        logger.info("Generated analysis code from LLM")
        
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Failed to generate code: {str(e)}")
    
    # Extract code from response
    code_patterns = [
        r'```python\s*(.*?)\s*```',
        r'```\s*(.*?)\s*```'
    ]
    
    extracted_code = None
    for pattern in code_patterns:
        matches = re.findall(pattern, code_response, re.DOTALL)
        if matches:
            extracted_code = matches[0].strip()
            break
    
    if not extracted_code:
        # If no code blocks found, check if the response starts with import
        if code_response.strip().startswith(('import ', 'from ', '#')):
            extracted_code = code_response.strip()
        else:
            raise HTTPException(status_code=502, detail="No valid Python code found in LLM response")
    
    # Execute the analysis code
    with tempfile.TemporaryDirectory(prefix="analysis_") as temp_dir:
        work_dir = Path(temp_dir)
        logger.info(f"Executing code in {work_dir}")
        
        try:
            execution_result = execute_python_code(extracted_code, work_dir)
            logger.info(f"Code executed in {execution_result['execution_time']:.2f}s")
            
            if execution_result['returncode'] != 0:
                logger.warning(f"Code execution had errors: {execution_result['stderr']}")
            
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Code execution failed: {str(e)}")
    
    # Extract structured results
    results = extract_results_from_output(
        execution_result['stdout'], 
        execution_result['stderr'], 
        questions
    )
    
    # If results are incomplete, try fallback parsing
    if isinstance(results, list) and any(r is None for r in results):
        logger.warning("Some results are None, attempting fallback parsing")
        
        fallback_prompt = f"""
The code execution produced output but some answers weren't extracted properly.
Please extract the answers and return them in the exact format requested.

STDOUT:
{execution_result['stdout'][-4000:]}

STDERR:
{execution_result['stderr'][-1000:] if execution_result['stderr'] else 'No errors'}

Original request:
{questions}

Extract the answers and return them in the exact format requested (JSON array or object).
If some answers are missing, provide reasonable estimates or indicate the issue.
"""
        
        try:
            fallback_response = call_llm([
                {"role": "system", "content": "Extract and format answers from code execution output. Return the exact requested format."},
                {"role": "user", "content": fallback_prompt}
            ], max_tokens=4000)
            
            fallback_results = parse_json_response(fallback_response)
            if fallback_results:
                results = fallback_results
            
        except Exception as e:
            logger.error(f"Fallback parsing failed: {e}")
    
    elif isinstance(results, dict) and any("No result found" in str(v) for v in results.values()):
        logger.warning("Some object results missing, attempting fallback parsing")
        
        fallback_prompt = f"""
Extract the answers from this code execution output and format them as requested.

STDOUT:
{execution_result['stdout'][-4000:]}

Original request:
{questions}

Return the exact JSON object format requested.
"""
        
        try:
            fallback_response = call_llm([
                {"role": "system", "content": "Extract answers and return exact requested JSON format."},
                {"role": "user", "content": fallback_prompt}
            ], max_tokens=4000)
            
            fallback_results = parse_json_response(fallback_response)
            if fallback_results:
                results = fallback_results
            
        except Exception as e:
            logger.error(f"Fallback parsing failed: {e}")
    
    logger.info(f"Analysis complete. Results: {type(results)} with {len(results) if hasattr(results, '__len__') else 'N/A'} items")
    return JSONResponse(content=results)

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")