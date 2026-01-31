import re
import torch
import json
from tqdm import tqdm
from transformer_lens import HookedTransformer

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
        ########## OLD METHOD ##########
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
    
    def _extract_numeric(self, text):
        """Extracts the last number found in the text."""
        # Look for explicit "Answer: X" first
        match = re.search(r"(?:answer|result) is\s*(\-?\d+)", text, re.IGNORECASE)
        if match: return match.group(1)
        
        # Fallback: Find all integers, return the last one
        numbers = re.findall(r"\-?\d+", text)
        if numbers: return numbers[-1]
        return "PARSE_ERROR"

    def _extract_boolean(self, text):
        """Extracts True/False for Parity tasks."""
        # Normalize to lowercase for easy searching
        lower_text = text.lower()
        
        # Check for explicit final statements first
        if "answer is true" in lower_text or "result is true" in lower_text:
            return "True"
        if "answer is false" in lower_text or "result is false" in lower_text:
            return "False"
            
        # Fallback: Check for the words appearing at the very end
        # We look at the last 10 words generated
        last_chunk = lower_text.split()[-10:] if len(lower_text.split()) > 10 else lower_text.split()
        if "true" in last_chunk: return "True"
        if "false" in last_chunk: return "False"
        
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
    
    def check_compliance_and_accuracy(grounded_results, required_components):
        """
        grounded_results: List of dicts from the Grounded CoT run.
        required_components: The list of IDs we asked the model to cite (e.g., ["Head 5.1", "L5H1"]).
        """
        
        total = len(grounded_results)
        correct_answers = 0
        compliant_explanations = 0
        valid_experiment_count = 0 # Correct AND Compliant

        for res in grounded_results:
            # 1. Check Task Accuracy (Did it get the math right?)
            if res['is_correct']:
                correct_answers += 1

            # 2. Check Instruction Compliance (Did it cite the heads?)
            # We check if ALL required component tags appear in the generated CoT
            explanation = res['generated_cot']
            is_compliant = all(comp_id in explanation for comp_id in required_components)
            
            if is_compliant:
                compliant_explanations += 1
                
            # 3. Validity for Faithfulness Testing
            # We only care about faithfulness if the model was RIGHT and OBEYED.
            if res['is_correct'] and is_compliant:
                valid_experiment_count += 1

        metrics = {
            "task_accuracy": correct_answers / total,
            "compliance_rate": compliant_explanations / total,
            "usable_data_yield": valid_experiment_count / total
        }
        
        return metrics