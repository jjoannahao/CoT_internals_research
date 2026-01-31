from transformer_lens import HookedTransformer
import torch
import json
from tqdm import tqdm

# currently only using phi-1.5 for sake of memory
class CoTBaselineRunner:
    def __init__(self, model_name="microsoft/phi-1_5", device="cuda"):
        print(f">> Loading {model_name}...")
        # Loading in fp16 to save memory as requested
        self.model = HookedTransformer.from_pretrained(
            model_name, 
            device=device, 
            dtype=torch.float16,
            fold_ln=False
        )
        self.model_name = model_name
        self.tokenizer = self.model.tokenizer
        self.stop_tokens = ["\n\n", "Q:", "Question:", "###"]

    def _extract_answer(self, full_text):
        """
        Attempts to isolate the final numeric/boolean answer from the text.
        Splits by 'Answer:' or looks for the last number.
        """
        # 1. Try splitting by explicit marker
        if "A:" in full_text:
            after_marker = full_text.split("A:")[-1].strip()
            # Take the first token/word after the marker
            return after_marker.split()[0].replace(".", "")
        
        # 2. Fallback: Heuristic cleanup
        # This is messy for base models; we'll refine based on your results
        return "PARSE_ERROR"

    def run_baseline(self, dataset, output_file="baseline_results.jsonl", debug_limit=5):
        # 1. REMOVED: results = [] (This was the memory leak)
        
        print(f">> Starting Baseline Run on {len(dataset)} tasks...")
        
        # Track errors just for the print limit
        error_count = 0
        
        for task in tqdm(dataset):
            prompt = task['clean']['prompt']
            ground_truth = task['clean']['answer']
            
            # --- GENERATION ---
            output = self.model.generate(
                prompt, 
                max_new_tokens=100, 
                temperature=0,
                top_k=1,
                prepend_bos=True,
                verbose=False
            )
            
            # --- CLEANING ---
            generated_only = output[len(prompt):]
            
            # Stop token logic
            for stop_tok in self.stop_tokens:
                if stop_tok in generated_only:
                    generated_only = generated_only.split(stop_tok)[0]
            
            # --- EXTRACTION ---
            predicted_ans = self._extract_answer(generated_only)
            
            # --- DEBUGGING BLOCK (The Solution) ---
            if predicted_ans == "PARSE_ERROR":
                error_count += 1
                if error_count <= debug_limit:
                    print(f"\n[DEBUG FAILURE #{error_count}]")
                    print(f"EXPECTED: {ground_truth}")
                    print(f"MODEL OUTPUT (First 200 chars): {generated_only[:200]!r}...") 
                    print("-" * 30)

            # --- SCORING ---
            is_correct = (ground_truth.strip().lower() in predicted_ans.lower())
            
            result_entry = {
                "id": task.get("id", "unknown"),
                "prompt": prompt, # Warning: Prompts are large. If low disk space, remove this.
                "generated_cot": generated_only,
                "predicted_answer": predicted_ans,
                "ground_truth": ground_truth,
                "is_correct": is_correct
            }
            
            # --- STREAM TO DISK
            # We append immediately and don't keep result_entry in RAM
            with open(output_file, "a") as f:
                f.write(json.dumps(result_entry) + "\n")
            
            # Force Python to clear the large string variables immediately
            del output, generated_only, result_entry

        return None # Don't return the huge list
    