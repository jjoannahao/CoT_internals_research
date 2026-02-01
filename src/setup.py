from transformer_lens import HookedTransformer
import torch
import gc
import random

def clear_memory():
    """Helper to aggressively free GPU memory between model loads."""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
    gc.collect()

def generateDataset(generator, examples_per_task):
  linear_dataset = [generator.generate_linear_pair() for _ in range(examples_per_task)]
  cblg_dataset = [generator.generate_cblg_pair() for _ in range(examples_per_task)]
  multiway_dataset = [generator.generate_multiway_pair() for _ in range(examples_per_task)]
  pat_dataset = [generator.generate_parity_pat_pair() for _ in range(examples_per_task)]
  
  full_dataset = linear_dataset + cblg_dataset + multiway_dataset + pat_dataset
  
  random.shuffle(full_dataset)
  
  print(f"# --- Generated {examples_per_task} per task, for total dataset of length {len(full_dataset)}")
  
  return full_dataset
    
def generateExemplars(generator, num_exemplars):
    """
    generate examples to form few-shot prompt for experiments 
    """
    exemplars = dict()
    
    for task_class in ["linear_symbolic", "CBLG", "MultiWay", "Parity_PAT"]:
      if task_class == "linear_symbolic":
          tasks = [generator.generate_linear_pair() for _ in range(num_exemplars)]
      elif task_class == "CBLG":
          tasks = [generator.generate_cblg_pair() for _ in range(num_exemplars)]
      elif task_class == "MultiWay":
          tasks = [generator.generate_multiway_pair() for _ in range(num_exemplars)]
      elif task_class == "Parity_PAT":
          tasks = [generator.generate_parity_pat_pair() for _ in range(num_exemplars)]

      prompt_block = ""
      for item in tasks:
        prompt_block += f"Q: {item['clean']['prompt']}\nA: {item['clean']['answer']}\n\n"
      
      exemplars[task_class] = prompt_block

    return exemplars


def buildPrompt(task_item, exemplars):
  """
  form few-shot prompt with exemplars
  """
  task_type = task_item["task_class"]
  curr_exemplars = exemplars[task_type]

  current_q = task_item['clean']['prompt']
        
  # Standard CoT Format
  full_prompt = (
      f"Solve the following problems step-by-step.\n\n"
      f"{exemplars}"
      f"Q: {current_q}\nA: Let's think step by step." # Trigger phrase
  )

  return full_prompt
