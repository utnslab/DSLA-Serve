from fastapi import FastAPI, Request, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles # Add this import
import uvicorn
import torch
import asyncio
import time
from collections import deque
from transformers import AutoTokenizer, LlamaConfig, AutoModelForCausalLM
import transformers
from typing import Optional, Tuple, List, Dict, Any, Deque

from layer import DualStateLinearAttention, convert_transformer_to_dsla, convert_dsla_to_transformer
from request_response import *

from safetensors import safe_open
import torch.nn.functional as F
import math

app = FastAPI()

BASE_MODEL_NAME = "meta-llama/Llama-3.1-8B-Instruct"

# Global variables for model and tokenizer
MODEL: torch.nn.Module = None
TOKENIZER: AutoTokenizer = None
MODEL_CONFIG: LlamaConfig = None
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
active_requests_count: int = 0
BASE_MODEL: AutoModelForCausalLM = None

ORIGINAL = {}
DSLA_LAYERS = {}
layers_in_transformer = []
layers_in_dsla = []
CHECKPOINT_DIR = '' # SET DIRECTORY TO THE CHECKPOINTS

request_queue: Deque[Tuple[CompletionRequest, asyncio.Future]] = deque()

# Batching parameters (kept as before)
BATCH_MAX_SIZE = 1
BATCH_TIMEOUT_S = 0.05
BATCH_MIN_SIZE = 1

# Load monitoring and swap parameters (kept as before)
MAX_QUEUE_SIZE_FOR_SWAP = 50
MAX_PROMPT_TOKENS_IN_QUEUE_FOR_SWAP = 1024
LAST_SWAP_TIME = time.time()
SWAP_COOLDOWN = 60
MAX_CONV_RATE = 0.75

# Basic metric
total_generated_tokens_since_last_update = 0
total_requests_completed_since_last_update = 0
last_metrics_update_time = time.time()


def calculate_attention_entropy_order(model: torch.nn.Module, tokenizer: AutoTokenizer) -> List[int]:
    dataset = load_dataset("lmsys/lmsys-chat-1m", split='train')  # Use whatever sample text
    dataset_iterator = iter(dataset)

    sample_prompt = next(dataset_iterator)["conversation"][0]['content']
    inputs = tokenizer(sample_prompt, return_tensors="pt", padding=True, truncation=True).to(model.device)

    input_ids = inputs["input_ids"]
    attention_mask = inputs["attention_mask"] # (batch_size, seq_len)

    if input_ids.shape[1] < 2:
        raise ValueError("Sample prompt is too short for attention entropy calculation. Please use a longer prompt.")

    last_token_idx = input_ids.shape[1] - 1

    model.eval()
    with torch.no_grad():
        output = model(input_ids, attention_mask=attention_mask, output_attentions=True)

    if not hasattr(output, 'attentions') or output.attentions is None:
        raise RuntimeError("Model did not return attention weights. Ensure 'output_attentions=True' is supported and correctly handled by the model's forward method.")

    if not isinstance(output.attentions, tuple):
        raise TypeError(f"Expected output.attentions to be a tuple, but got {type(output.attentions)}")

    layer_entropies = {}
    num_layers = model.config.num_hidden_layers

    for i in range(num_layers):
        attn_weights_tensor = output.attentions[i] # Shape: (batch_size, num_heads, seq_len, seq_len)

        if not isinstance(attn_weights_tensor, torch.Tensor):
            raise TypeError(f"Expected output.attentions[{i}] to be a torch.Tensor, but got {type(attn_weights_tensor)}")

        if last_token_idx >= attn_weights_tensor.shape[2]:
            raise IndexError(f"last_token_idx ({last_token_idx}) is out of bounds for query_length ({attn_weights_tensor.shape[2]}) in layer {i} attention weights.")
        if last_token_idx >= attn_weights_tensor.shape[3]:
             raise IndexError(f"last_token_idx ({last_token_idx}) is out of bounds for key_length ({attn_weights_tensor.shape[3]}) in layer {i} attention weights.")

        final_query_attn_scores = attn_weights_tensor[:, :, last_token_idx, :] # Shape: (batch_size, num_heads, key_length)

        batch_size, num_heads, key_length = final_query_attn_scores.shape
        expanded_mask_for_indexing = attention_mask.unsqueeze(1).expand(-1, num_heads, -1) # Shape (B, H, S)
        masked_scores = final_query_attn_scores.clone() 
        masked_scores[expanded_mask_for_indexing == 0] = 0.0

        sum_scores = masked_scores.sum(dim=-1, keepdim=True)
        sum_scores_clamped = sum_scores.clamp(min=1e-9)
        normalized_scores = masked_scores / sum_scores_clamped

        entropy_per_head_seq = -torch.where(
            normalized_scores > 1e-9,
            normalized_scores * torch.log(normalized_scores),
            torch.tensor(0.0, device=model.device, dtype=model.config.torch_dtype)
        ).sum(dim=-1)

        avg_entropy = entropy_per_head_seq.mean().item()

        layer_entropies[i] = avg_entropy

    sorted_layers = sorted(layer_entropies.items(), key=lambda item: item[1])
    layers_in_transformer = [layer_idx for layer_idx, _ in sorted_layers]

    return layers_in_transformer


async def load_models_at_startup():
    global BASE_MODEL, TOKENIZER, MODEL_CONFIG, MODEL
    global ORIGINAL, DSLA_LAYERS, CHECKPOINT_DIR
    global layers_in_transformer

    TOKENIZER = AutoTokenizer.from_pretrained(BASE_MODEL_NAME)
    if TOKENIZER.pad_token is None:
        TOKENIZER.pad_token = TOKENIZER.eos_token
    TOKENIZER.padding_side = "left"

    BASE_MODEL = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_NAME,
        torch_dtype=torch.float16,
        device_map="auto"
    )
    BASE_MODEL.eval()
    print(f"Base model loaded in {BASE_MODEL.device}")

    MODEL_CONFIG = BASE_MODEL.config
    MODEL_CONFIG.kernel_hidden_size = 128
    MODEL_CONFIG.ker_hid = 128
    MODEL_CONFIG.gate_logit_normalizer = 16.0
    MODEL_CONFIG.clamp_min = -50.0

    for name, param in BASE_MODEL.named_parameters():
        if 'self_attn' in name:
            ORIGINAL[name] = param.detach().clone()

    for i in range(1,5):
        with safe_open(f"{CHECKPOINT_DIR}/model-0000{i}-of-00004.safetensors", framework="pt", device='cpu') as f:
            for k in f.keys():
                if 'self_attn' in k:
                    tensor = f.get_tensor(k).to('cuda', non_blocking=True)
                    DSLA_LAYERS[k] = tensor 

    MODEL = BASE_MODEL
    print("Initial active model: Base Model.")

    try:
        layers_in_transformer = calculate_attention_entropy_order(BASE_MODEL, TOKENIZER)
    except Exception as e:
        layers_in_transformer = list(range(31,-1,-1))


async def model_swap_monitor():
    global MODEL, LAST_SWAP_TIME
    global MAX_QUEUE_SIZE_FOR_SWAP, MAX_PROMPT_TOKENS_IN_QUEUE_FOR_SWAP
    global BASE_MODEL
    global MODEL_CONFIG, MAX_CONV_RATE

    while True:
        max_prompt_len_in_queue = 0
        if request_queue:
            max_prompt_len_in_queue = max(item[0].prompt_tokens for item in request_queue) 

        current_pending_requests = len(request_queue)

        if time.time() - LAST_SWAP_TIME < SWAP_COOLDOWN:
            await asyncio.sleep(5)
            continue

        if max_prompt_len_in_queue >= MAX_PROMPT_TOKENS_IN_QUEUE_FOR_SWAP or \
               current_pending_requests >= MAX_QUEUE_SIZE_FOR_SWAP:
            # Convert to DSLA
            print(f"[{time.time():.2f}] High load detected (Max prompt in queue: {max_prompt_len_in_queue}, Pending reqs: {current_pending_requests}). Converting Layer to DSLA...")
            if len(layers_in_transformer) > 0 and len(layers_in_dsla)/32 < MAX_CONV_RATE:
                layer_to_convert = layers_in_transformer.pop(0)
                layers_in_dsla.insert(0, layer_to_convert)
                MODEL_CONFIG.dsla_list = layers_in_dsla
                convert_transformer_to_dsla(MODEL, MODEL_CONFIG, DSLA_LAYERS, layer_to_convert)
                LAST_SWAP_TIME = time.time()
                print(f"[{time.time():.2f}] Swapped layer {layer_to_convert} to DSLA. Current conversion rate: {len(layers_in_dsla)/32*100}%")
            else:
                print(f"[{time.time():.2f}] No layers to convert. Current conversion rate: {len(layers_in_dsla)/32*100}")

        elif max_prompt_len_in_queue < MAX_PROMPT_TOKENS_IN_QUEUE_FOR_SWAP * 0.8 and \
               current_pending_requests < MAX_QUEUE_SIZE_FOR_SWAP * 0.5 and len(layers_in_dsla) > 0:
            # Convert back to Transformer
            print(f"[{time.time():.2f}] Load reduced (Max prompt in queue: {max_prompt_len_in_queue}, Pending reqs: {current_pending_requests}). Converting Layer to Transformer...")
            if len(layers_in_dsla) > 0:
                layer_to_convert = layers_in_dsla.pop(0)
                layers_in_transformer.insert(0, layer_to_convert)
                MODEL_CONFIG.dsla_list = layers_in_dsla
                convert_dsla_to_transformer(MODEL, MODEL_CONFIG, ORIGINAL, layer_to_convert)
                LAST_SWAP_TIME = time.time()
                print(f"[{time.time():.2f}] Swapped layer {layer_to_convert} back to transformer model. Current conversion rate: {len(layers_in_dsla)/32*100}")
            else:
                print(f"[{time.time():.2f}] No layers to convert. Current conversion rate: {len(layers_in_dsla)/32*100}%")
        await asyncio.sleep(5)


async def batch_scheduler():
    global active_requests_count, MODEL, TOKENIZER
    global total_generated_tokens_since_last_update, total_requests_completed_since_last_update

    while True:
        if not request_queue:
            await asyncio.sleep(BATCH_TIMEOUT_S)
            continue

        batch_start_time = time.time()
        while len(request_queue) < BATCH_MIN_SIZE and (time.time() - batch_start_time) < BATCH_TIMEOUT_S:
            await asyncio.sleep(0.005)

        current_batch: List[Tuple[CompletionRequest, asyncio.Future]] = []
        num_requests_to_process = min(len(request_queue), BATCH_MAX_SIZE)

        if num_requests_to_process == 0:
            continue

        for _ in range(num_requests_to_process):
            current_batch.append(request_queue.popleft())
        
        active_requests_count += len(current_batch)
        print(f"[{time.time():.2f}] Batch of {len(current_batch)} requests pulled. Active requests: {active_requests_count}")

        prompts = [req.prompt for req, _ in current_batch]
        inputs = TOKENIZER(prompts, return_tensors="pt", padding=True, truncation=True).to(MODEL.device)
        max_new_tokens_batch = max(req.max_tokens for req, _ in current_batch)

        batch_outputs = None
        try:
            with torch.no_grad():
                outputs = await asyncio.to_thread(
                    MODEL.generate,
                    **inputs,
                    max_new_tokens=max_new_tokens_batch,
                    do_sample=True,
                    temperature=0.7,
                    pad_token_id=TOKENIZER.eos_token_id,
                )
            
            current_batch_idx = 0
            for req, future in current_batch:
                prompt_tokens = inputs.input_ids[current_batch_idx].shape[0]
                output_tokens = outputs[current_batch_idx].shape[0]
                generated_tokens = output_tokens - prompt_tokens

                decoded_completion = TOKENIZER.decode(outputs[current_batch_idx][prompt_tokens:], skip_special_tokens=True)

                response_choice = CompletionChoice(
                    text=decoded_completion,
                    index=0,
                    logprobs=None,
                    finish_reason="length" if generated_tokens >= req.max_tokens else "eos_token"
                )
                
                response_usage = CompletionUsage(
                    prompt_tokens=prompt_tokens,
                    completion_tokens=generated_tokens,
                    total_tokens=prompt_tokens + generated_tokens
                )

                response_data = CompletionResponse(
                    model=req.model,
                    choices=[response_choice],
                    usage=response_usage
                )
                
                future.set_result(response_data)
                total_generated_tokens_since_last_update += generated_tokens
                total_requests_completed_since_last_update += 1
                current_batch_idx += 1

        except Exception as e:
            print(f"[{time.time():.2f}] Error during batch generation: {e}")
            for _, future in current_batch:
                if not future.done():
                    future.set_exception(HTTPException(status_code=500, detail=f"Batch generation failed: {e}"))
        finally:
            active_requests_count -= len(current_batch)
            print(f"[{time.time():.2f}] Batch processing finished. Active requests: {active_requests_count}")


async def send_live_metrics():
    global last_metrics_update_time, total_generated_tokens_since_last_update, total_requests_completed_since_last_update
    global active_requests_count, BASE_MODEL

    while True:
        # Calculate throughput over the last interval
        current_time = time.time()
        interval = current_time - last_metrics_update_time

        req_per_sec = total_requests_completed_since_last_update / interval if interval > 0 else 0
        tokens_per_sec = total_generated_tokens_since_last_update / interval if interval > 0 else 0

        # Reset counters for the next interval
        total_requests_completed_since_last_update = 0
        total_generated_tokens_since_last_update = 0
        last_metrics_update_time = current_time

        metrics_data = {
            "timestamp": current_time,
            "queue_size": len(request_queue),
            "active_requests": active_requests_count,
            "requests_per_sec": req_per_sec,
            "tokens_per_sec": tokens_per_sec,
            "current_model": BASE_MODEL
        }

        await asyncio.sleep(1)


@app.post("/v1/completions", response_model=CompletionResponse)
async def create_completion(request: CompletionRequest):
    future_result = asyncio.Future()
    
    request_queue.append((request, future_result))
    print(f"[{time.time():.2f}] Request added to queue. Queue size: {len(request_queue)}. Active requests (model processing): {active_requests_count}")

    if len(request_queue) > MAX_QUEUE_SIZE_FOR_SWAP * 2:
        request_queue.pop()
        raise HTTPException(status_code=503, detail="Server is heavily overloaded, please try again later.")

    try:
        response_data = await future_result
        return response_data
    except Exception as e:
        raise e 


@app.on_event("startup")
async def startup_event():
    await load_models_at_startup()
    asyncio.create_task(model_swap_monitor())
    asyncio.create_task(batch_scheduler())
    asyncio.create_task(send_live_metrics()) 


@app.on_event("shutdown")
async def shutdown_event():
    if GPU_AVAILABLE:
        nvmlShutdown()
        print("NVML shut down.")


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)