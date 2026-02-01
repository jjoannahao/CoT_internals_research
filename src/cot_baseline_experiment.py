import json
import random
import torch
import gc
from transformer_lens import HookedTransformer
from tqdm import tqdm

from setup import loadModel
from task_generation import *
from cot_baseline import CoTBaselineRunner
from setup import *


# TODO: fill in docstrings for all classes, functions
# TODO: lightweight error checking; update test scripts?
# TODO: setup logs for experiment runs (initialization of models, generator, exemplars, datasets)?

def clearMemory():
    """Helper to aggressively free GPU memory between model loads."""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
    gc.collect()


if __name__ == "__main__":
  clearMemory()
  phi_name = "microsoft/phi-1_5"
  llama_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
  gemma_name = "gemma-2-2b"
  qwen_name = "Qwen/Qwen1.5-1.8B"
  
  # ===== SEQUENTIAL TESTING LOOP to save on memory ===== #
  # --- initialization 
  gen = MechanisticTaskGenerator(seed=42)
  dataset = generateDataset(gen, 50)
  exemplars = generateExemplars(generator=gen, num_exemplars=8)
  
  print(f"--- FROM EXEMPLARS:")
  if len(exemplars) > 0:
    print(json.dumps(exemplars, indent=4))
  else:
    print("nOTHING IN EXEMPLARS ????????????")
  print("\n")

  # [phi_name, llama_name, gemma_name, qwen_name]
  for model_name in [phi_name]:
    print(f"\n{'='*20}\nSTARTING MODEL: {model_name}\n{'='*20}\n")
    try:
      # --- load models
      model = HookedTransformer.from_pretrained(
          model_name, 
          device="cuda",  # loads to GPU if available, otherwise CPU
          dtype=torch.float16,  # save on memory
          fold_ln=False
      ) 
      print(f"{'-'*10} Successfully loaded {model_name}\n")

      # --- initialize CoT runner
      runner = CoTBaselineRunner(model=model, model_name=model_name, device="cuda")

      # --- run dataset
      formatted_dataset = []

      print(f"\n{'-'*20}\nsample from global dataset:\n{'-'*20}")
      print(json.dumps(dataset[0], indent=4))

      # print(json.dumps(dataset[1], indent=4))
      for item in dataset[:3]:


        print(f"\n---- CURRENT ITEM:\n{json.dumps(item, indent=4)}\n")
        full_prompt = buildPrompt(task_item=item, exemplars=exemplars)
        print(f"\n---- FULL PROMPT:\n{full_prompt}\n")


        formatted_item = {
            "task_class": item["task_class"],
            "clean": {
              "prompt": full_prompt, # REPLACES raw prompt with Few-Shot Prompt
              "answer": item['clean']['answer']
          }
        }


        print(f"\n{'-'*20}\nFORMATTED ITEM:")
        print(json.dumps(formatted_item, indent=4))
        formatted_dataset.append(formatted_item)

      # # --- outputs
      output_filename = f"baseline_results_{model_name.split("/")[1]}.jsonl"
      runner.run_baseline(formatted_dataset, output_file=output_filename)
      print(f">>> Finished {model_name}. Results in {output_filename}")

      # --- cleanup
      del model
      del runner
      clearMemory()
      print(">>> Memory cleared")

    except Exception as e:
        print(f"! ----- Failed: {e}\n")
        clearMemory()




# if __name__ == "__main__":
#     phi_name = "microsoft/phi-1_5"
#     llama_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
#     gemma_name = "gemma-2-2b"
#     qwen_name = "Qwen/Qwen1.5-1.8B"
    
#     # ===== SEQUENTIAL TESTING LOOP to save on memory ===== #
#     # --- initialize data generator 
#     gen = MechanisticTaskGenerator(seed=42)
    
#     for model_name in [phi_name, llama_name, gemma_name, qwen_name]:
#         # --- load models
#         try:
#             model = HookedTransformer.from_pretrained(
#                 model_name, 
#                 device="cuda",
#                 dtype=torch.float16,
#                 fold_ln=False
#             ) # loads to GPU if available, otherwise CPU
#         except Exception as e:
#             print(f"! ----- Failed to load {model_name}: {e}")
#             clear_memory()
            
#         # --- create exemplars & dataset of synthetic examples
#         linear_exemplars = generateExemplars(model, gen, "linear_symbolic", 8)
#         cblg_exemplars = generateExemplars(model, gen, "cblg", 8)
#         multiway_exemplars = generateExemplars(model, gen, "multiway", 8)
#         pat_exemplars = generateExemplars(model, gen, "pat", 8)
    
#         linear_dataset = [gen.generate_linear_pair() for _ in range(500)]
#         cblg_dataset = [gen.generate_cblg_pair() for _ in range(500)]
#         multiway_dataset = [gen.generate_multiway_pair() for _ in range(500)]
#         pat_dataset = [gen.generate_parity_pat_pair() for _ in range(500)]
        
#         # randomly shuffle indices w/in task types then create overall dataset
#         task_dataset = [linear_dataset, cblg_dataset, multiway_dataset, pat_dataset]
        
#         # create few-shot prompts w/ exemplars:
        

#         # --- initialize runner for CoT baseline 
#         runner = CoTBaselineRunner(model, model_name)
#         runner.run_baseline()
#         cot_baseline_accuracy = runner.check_compliance_and_accuracy()
    
    
    
    
    
    # phi = loadModel(phi_name)
    # llama = loadModel(llama_name)
    # gemma = loadModel(gemma_name)
    # qwen = loadModel(qwen_name)
    
    # --- create exemplars & dataset of synthetic examples
    # for model in [phi, llama, gemma, qwen]:
    #     linear_exemplars = generateExemplars()
    #     cblg_exemplars = ""
    #     multiway_exemplars = ""
    #     pat_exemplars = ""
    
    # linear_dataset = [gen.generate_linear_pair() for _ in range(500)]
    # cblg_dataset = [gen.generate_cblg_pair() for _ in range(500)]
    # multiway_dataset = [gen.generate_multiway_pair() for _ in range(500)]
    # pat_dataset = [gen.generate_parity_pat_pair() for _ in range(500)]
    
    # --- initialize CoT runners 
    # TODO: refactor CoTBaselineRunner with loadModel()?
    # phi_runner = CoTBaselineRunner(model_name=phi_name)
    # llama_runner = CoTBaselineRunner(model_name=llama_name)
    # gemma_runner = CoTBaselineRunner(model_name=gemma_name)
    # qwen_runner = CoTBaselineRunner(model_name=qwen_name)
    
    # --- mini experiment 
    
    
    
    
