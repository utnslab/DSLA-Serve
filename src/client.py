MODEL_NAME = "meta-llama/Llama-3.1-8B-Instruct"

import csv
import requests
import time
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
from transformers import AutoTokenizer
from datasets import load_dataset
import collections
import statistics 
import numpy as np
import random


CSV_FILE = "AzureLLMInferenceTrace_conv_short.csv"
VLLM_API_URL = "http://localhost:8000/v1/completions"
MAX_WORKERS = 16  # Number of concurrent threads for client to send requests
HEADERS = {"Content-Type": "application/json"}
LOG_FILE = "log.csv"
DATASET_NAME = "lmsys/lmsys-chat-1m"

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

all_latencies = []
all_generated_tokens = []
all_successful_requests_info = []


def make_prompt_from_length(num_tokens: int) -> str:
    base = "Hello"
    while len(tokenizer.encode(base)) < num_tokens:
        base += " Hello"
    return base

def make_prompt_from_dataset(dataset_iterator):
    """
    Reads from the lmsys/lmsys-chat-1m dataset and returns the first English
    "content" from the "conversation" field.
    """
    for item in dataset_iterator:
        conversation = item.get("conversation")
        language = item.get("language")

        if conversation and language and language == "English":
            return_message = ""
            for message in conversation:
                if message.get("content"):
                    return_message += f"{message['content']}\n"
            if return_message: 
                return return_message
    return None

def parse_timestamp(ts: str) -> datetime:
    if '.' in ts:
        pre, frac = ts.split('.')
        ts = f"{pre}.{frac[:6]}"
    return datetime.strptime(ts, "%Y-%m-%d %H:%M:%S.%f")

def send_request_and_log(idx, prompt_content, gen_tokens, log_list):
    # These globals are *not* thread-safe for direct modification
    # They will be updated in the main thread after all futures complete.
    # So we only use log_list (thread-safe append) here.

    prompt = prompt_content
    prompt_tokens = len(tokenizer.encode(prompt))

    payload = {
        "model": MODEL_NAME,
        "prompt": prompt,
        "prompt_tokens": prompt_tokens,
        "max_tokens": gen_tokens,
        "temperature": 0.0,
    }

    start_time = time.time()
    start_iso = datetime.now().isoformat()
    
    status = "ERROR_UNKNOWN" # Default status

    try:
        response = requests.post(VLLM_API_URL, headers=HEADERS, json=payload, timeout=120) # Increased timeout
        latency = time.time() - start_time
        end_iso = datetime.now().isoformat()

        if response.status_code == 200:
            result = response.json()
            text = result["choices"][0]["text"]
            actual_generated_tokens = len(tokenizer.encode(text)) 
            prompt_tokens = len(tokenizer.encode(payload['prompt']))

            print(f"[{idx}]  {latency:.3f}s (Prompt: {prompt_tokens} tokens): {' '.join(payload['prompt'][:30].splitlines())}... (Response: {actual_generated_tokens} tokens): {' '.join(text[:30].splitlines())}...")
            status = "OK"
            
            all_latencies.append(latency)
            all_generated_tokens.append(actual_generated_tokens)
            all_successful_requests_info.append({
                "start": start_time,
                "end": time.time(),
                "prompt_tokens": prompt_tokens,
                "generated_tokens": actual_generated_tokens
            })

        else:
            print(f"[{idx}]  HTTP {response.status_code} - {response.text}")
            status = f"HTTP_{response.status_code}"

    except requests.exceptions.Timeout:
        latency = time.time() - start_time
        end_iso = datetime.now().isoformat()
        print(f"[{idx}]  Timeout after {latency:.3f}s")
        status = "TIMEOUT"
    except requests.exceptions.RequestException as e:
        latency = time.time() - start_time
        end_iso = datetime.now().isoformat()
        print(f"[{idx}]  Request Error: {e}")
        status = "REQUEST_ERROR"
    except Exception as e:
        latency = time.time() - start_time
        end_iso = datetime.now().isoformat()
        print(f"[{idx}]  Unexpected Exception: {e}")
        status = "EXCEPTION"

    log_list.append({
        "RequestID": idx,
        "StartTime": start_iso,
        "EndTime": end_iso,
        "Latency": f"{latency:.6f}",
        "PromptTokens": prompt_tokens,
        "RequestedGeneratedTokens": gen_tokens, # Original requested amount
        "ActualGeneratedTokens": actual_generated_tokens if status == "OK" else 0, # Actual generated
        "Status": status,
    })

def main():
    print(f"Starting trace with {MAX_WORKERS} concurrent workers...")

    trace_data = []
    with open(CSV_FILE, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            ts = parse_timestamp(row["TIMESTAMP"])
            gen = int(row["GeneratedTokens"])
            trace_data.append((ts, gen))

    print(f"Loading dataset: {DATASET_NAME}...")
    try:
        dataset = load_dataset(DATASET_NAME, split='train', streaming=True) # Use streaming for large datasets
        dataset_iterator = iter(dataset)
        print("Dataset loaded successfully.")
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return

    base_time = trace_data[0][0]
    wall_start = time.time()
    log_list = collections.deque() # Use deque for efficient appends

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = []

        for idx, (trace_ts, gen_tokens) in enumerate(trace_data):
            prompt_content = make_prompt_from_dataset(dataset_iterator)
            if prompt_content is None:
                print(f"[{idx}] Warning: No more English prompts found in the dataset. Stopping.")
                break

            trace_offset = (trace_ts - base_time).total_seconds()
            real_elapsed = time.time() - wall_start
            sleep_time = trace_offset - real_elapsed

            if sleep_time > 0:
                time.sleep(sleep_time)

            future = executor.submit(send_request_and_log, idx, prompt_content, gen_tokens, log_list)
            futures.append(future)

            if idx == 15:
                break

        for future in as_completed(futures):
            _ = future.result()

    print(f"\n--- Performance Statistics ---")

    if not all_latencies:
        print("No successful requests to calculate statistics.")
        return

    # Latency Statistics (ms)
    latencies_ms = [l * 1000 for l in all_latencies]
    
    avg_latency = np.mean(latencies_ms)
    median_latency = np.median(latencies_ms)
    p99_latency = np.percentile(latencies_ms, 99)

    print(f"Average Latency: {avg_latency:.2f} ms")
    print(f"Median Latency (p50): {median_latency:.2f} ms")
    print(f"P99 Latency: {p99_latency:.2f} ms")

    total_successful_requests = len(all_latencies)
    total_generated_tokens_all = sum(all_generated_tokens)

    if all_successful_requests_info:
        overall_start_time = min(r["start"] for r in all_successful_requests_info)
        overall_end_time = max(r["end"] for r in all_successful_requests_info)
        total_duration = overall_end_time - overall_start_time
    else:
        total_duration = 0

    if total_duration > 0:
        requests_per_sec = total_successful_requests / total_duration
        tokens_per_sec = total_generated_tokens_all / total_duration
        print(f"Requests/sec: {requests_per_sec:.2f}")
        print(f"Tokens/sec (Generated): {tokens_per_sec:.2f}")
    else:
        print("Not enough successful requests or duration for throughput calculation.")

    with open(LOG_FILE, "w", newline="") as csvfile:
        fieldnames = ["RequestID", "StartTime", "EndTime", "Latency",
                      "PromptTokens", "RequestedGeneratedTokens", "ActualGeneratedTokens", "Status"]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for row in sorted(log_list, key=lambda r: r["RequestID"]):
            writer.writerow(row)

    print(f"\nDetailed logs written to {LOG_FILE}")


if __name__ == "__main__":
    main()