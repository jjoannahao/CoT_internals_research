import json
import random
import torch
import gc
from transformer_lens import HookedTransformer
from tqdm import tqdm

from task_generation import MechanisticTaskGenerator
from cot_baseline import CoTBaselineRunner



if __name__ == "__main__":
    # --- make data generator 
    gen = MechanisticTaskGenerator(seed=42)
    linear_dataset = [gen.generate_linear_pair() for _ in range(500)]
    cblg_dataset = [gen.generate_cblg_pair() for _ in range(500)]
    multiway_dataset = [gen.generate_multiway_pair() for _ in range(500)]
    pat_dataset = [gen.generate_parity_pat_pair() for _ in range(500)]
    
    # --- initialize CoT runners 
    phi_runner = CoTBaselineRunner(model_name="microsoft/phi-1_5")
    # llama_runner = CoTBaselineRunner(model_name="TinyLlama/TinyLlama-1.1B-Chat-v1.0")
    # gemma_runner = CoTBaselineRunner(model_name="gemma-2-2b")
    # qwen_runner = CoTBaselineRunner(model_name="Qwen/Qwen1.5-1.8B")
    
    # --- mini experiment 
    
    
    
    
