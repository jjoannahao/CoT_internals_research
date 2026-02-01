import json
import random
import torch
import gc
from transformer_lens import HookedTransformer
from tqdm import tqdm

from setup import loadModel
from task_generation import MechanisticTaskGenerator, generateExemplars
from cot_baseline import CoTBaselineRunner


# TODO: fill in docstrings for all classes, functions
# TODO: lightweight error checking; update test scripts?
# TODO: setup logs for experiment runs (initialization of models, generator, exemplars, datasets)

if __name__ == "__main__":
    # --- load models
    phi = loadModel("microsoft/phi-1_5")
    llama = loadModel("TinyLlama/TinyLlama-1.1B-Chat-v1.0")
    gemma = loadModel("gemma-2-2b")
    qwen = loadModel("Qwen/Qwen1.5-1.8B")
    
    # --- initialize data generator 
    gen = MechanisticTaskGenerator(seed=42)
    
    # --- create exemplars & dataset of synthetic examples
    for model in [phi, llama, gemma, qwen]:
        linear_exemplars = generateExemplars()
        cblg_exemplars = ""
        multiway_exemplars = ""
        pat_exemplars = ""
    
    linear_dataset = [gen.generate_linear_pair() for _ in range(500)]
    cblg_dataset = [gen.generate_cblg_pair() for _ in range(500)]
    multiway_dataset = [gen.generate_multiway_pair() for _ in range(500)]
    pat_dataset = [gen.generate_parity_pat_pair() for _ in range(500)]
    
    # --- initialize CoT runners 
    # TODO: refactor CoTBaselineRunner with loadModel()
    phi_runner = CoTBaselineRunner(model_name="microsoft/phi-1_5")
    # llama_runner = CoTBaselineRunner(model_name="TinyLlama/TinyLlama-1.1B-Chat-v1.0")
    # gemma_runner = CoTBaselineRunner(model_name="gemma-2-2b")
    # qwen_runner = CoTBaselineRunner(model_name="Qwen/Qwen1.5-1.8B")
    
    # --- mini experiment 
    
    
    
    
