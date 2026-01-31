import random

class MechanisticTaskGenerator:
    """
    dataset generator for 4 task types 
    """
    def __init__(self, seed=None):
        if seed is not None:
            random.seed(seed)

    def _get_fixed_width_int(self, min_val=10, max_val=99):
        # We use 2-digit numbers strictly to keep token lengths identical b/w clean and corrupted runs (shifting would mess up activation patching later)
        return random.randint(min_val, max_val)

    def _apply_op(self, op, val1, val2):
        if op == "add": return val1 + val2
        if op == "subtract": return val1 - val2
        if op == "multiply": return val1 * val2
        if op == "divide": return val1 // val2 if val2 != 0 else 0
        return 0

    # -------------------------------------------------------------------------
    # Helper: Prompt Renderer
    # -------------------------------------------------------------------------
    def _render_prompt(self, template, **kwargs):
        """Standardizes prompt creation for token consistency."""
        return template.format(**kwargs)

    # -------------------------------------------------------------------------
    # Task 1: Linear Symbolic
    # -------------------------------------------------------------------------
    def generate_linear_pair(self):
        """Generates a Clean and Corrupted pair to trace the causal effect of 'x'."""
        # 1. Shared Constants
        y = self._get_fixed_width_int()
        modifier = self._get_fixed_width_int(10, 20)
        op1 = random.choice(["add", "subtract", "multiply"])
        op2 = random.choice(["add", "subtract"]) 

        # 2. Distinct Inputs (Clean vs Corrupted)
        # We change ONLY x. Everything else stays the same.
        x_clean = self._get_fixed_width_int(10, 50)
        x_corrupt = self._get_fixed_width_int(51, 90) # Ensure it's different

        # 3. Calculate Answers
        # Clean
        res1_c = self._apply_op(op1, x_clean, y)
        ans_c = self._apply_op(op2, res1_c, modifier)
        
        # Corrupted
        res1_corr = self._apply_op(op1, x_corrupt, y)
        ans_corr = self._apply_op(op2, res1_corr, modifier)

        # 4. Render Texts
        # Note: Token structure is identical, only the digits of X change.
        template = "Start with {x}. {op1} {y}. Then {op2} {mod}. What is the result?"
        
        clean_prompt = self._render_prompt(template, x=x_clean, op1=op1, y=y, op2=op2, mod=modifier)
        corrupt_prompt = self._render_prompt(template, x=x_corrupt, op1=op1, y=y, op2=op2, mod=modifier)

        return {
            "task_class": "linear_symbolic",
            "clean": {"prompt": clean_prompt, "answer": str(ans_c), "x_val": x_clean},
            "corrupt": {"prompt": corrupt_prompt, "answer": str(ans_corr), "x_val": x_corrupt},
            "patching_target": "x_value" # This tells you what mechanism you are isolating
        }

    # -------------------------------------------------------------------------
    # Task 2: CBLG
    # -------------------------------------------------------------------------
    def generate_cblg_pair(self):
        """
        Generates a pair where the inputs 'a' and 'b' are identical in value, 
        but 'a' is modified slightly to flip its parity (Even <-> Odd).
        This isolates the 'Parity Checker' circuit.
        """
        b = self._get_fixed_width_int(10, 20)
        
        # Clean: Ensure 'a' is Even
        a_clean = self._get_fixed_width_int(20, 80)
        if a_clean % 2 != 0: a_clean += 1
        
        # Corrupted: Ensure 'a' is Odd (flip last bit)
        a_corrupt = a_clean + 1

        # Logic
        # Branch 0 (Even): a // 2 + b
        ans_clean = (a_clean // 2) + b
        
        # Branch 1 (Odd): a - b
        ans_corrupt = a_corrupt - b

        # Template
        template = "Input: {a}, {b}. If {a} is even, calculate {a}/2 + {b}. If {a} is odd, calculate {a} - {b}. Result:"

        return {
            "task_class": "CBLG",
            "clean": {
                "prompt": self._render_prompt(template, a=a_clean, b=b),
                "answer": str(ans_clean),
                "gate_state": "Even"
            },
            "corrupt": {
                "prompt": self._render_prompt(template, a=a_corrupt, b=b),
                "answer": str(ans_corrupt),
                "gate_state": "Odd"
            },
            "patching_target": "parity_gate"
        }

    # -------------------------------------------------------------------------
    # Task 3: Multi-Way Branching
    # -------------------------------------------------------------------------
    def generate_multiway_pair(self):
        """
        Generates a pair where 'x' changes slightly to alter the (x + y) % 3 value,
        forcing the model to switch between Add (0), Multiply (1), and Subtract (2).
        """
        # 1. Fixed Constants
        # We keep y small (single digit) to ensure multiplication results 
        # don't explode in token length compared to addition.
        y = random.randint(2, 9) 

        # 2. Generate Clean Input (x_clean)
        # We ensure x is 2 digits (10-99) to keep prompt length fixed
        x_clean = self._get_fixed_width_int(10, 99)
        mod_clean = (x_clean + y) % 3
        
        # 3. Generate Corrupted Input (x_corrupt)
        # We want to force a DIFFERENT branch.
        # We increment x by 1 or 2 to shift the modulo ring.
        # (x + 1) % 3 will always be different from x % 3.
        shift = 1
        x_corrupt = x_clean + shift
        
        # Edge case handling: If x_clean is 99, x_corrupt becomes 100 (3 digits).
        # We must keep x fixed-width (2 digits).
        if x_corrupt > 99:
            x_corrupt = x_clean - shift # Go down instead (99 -> 98)
            
        mod_corrupt = (x_corrupt + y) % 3

        # 4. Helper to calculate answer based on branch
        def get_ans(val_x, val_y, mod_val):
            if mod_val == 0: return val_x + val_y      # Branch 0: Add
            if mod_val == 1: return val_x * val_y      # Branch 1: Multiply
            return val_x - val_y                       # Branch 2: Subtract

        ans_clean = get_ans(x_clean, y, mod_clean)
        ans_corrupt = get_ans(x_corrupt, y, mod_corrupt)

        # 5. Render Texts
        # Explicit instructions to help the model maintain the reasoning chain
        template = (
            "Input: x={x}, y={y}. "
            "Compute S = (x + y) % 3. "
            "If S is 0, return x + y. "
            "If S is 1, return x * y. "
            "If S is 2, return x - y. "
            "Result:"
        )

        return {
            "task_class": "multiway_branching",
            "clean": {
                "prompt": self._render_prompt(template, x=x_clean, y=y),
                "answer": str(ans_clean),
                "selector_val": mod_clean,
                "active_op": ["add", "multiply", "subtract"][mod_clean]
            },
            "corrupt": {
                "prompt": self._render_prompt(template, x=x_corrupt, y=y),
                "answer": str(ans_corrupt),
                "selector_val": mod_corrupt,
                "active_op": ["add", "multiply", "subtract"][mod_corrupt]
            },
            "patching_target": "modulo_selector"
        }

    # -------------------------------------------------------------------------
    # Task 4: Parity PAT
    # -------------------------------------------------------------------------
    def generate_parity_pat_pair(self):
        """
        To verify faithfulness, we must show that flipping ONE predicate
        flips the final True/False answer.
        """
        # Generate 5 predicates
        predicates = []
        for i in range(5):
            num = self._get_fixed_width_int(20, 90)
            div = random.randint(2, 9)
            # Ensure we start with divisibility being False for simplicity, then force it
            if num % div == 0: num += 1 
            predicates.append({"num": num, "div": div, "is_div": False})

        # Randomly assign Truth to the first 4 predicates
        true_count = 0
        for p in predicates[:4]:
            if random.random() > 0.5:
                # Make it divisible
                p["num"] = p["num"] - (p["num"] % p["div"])
                p["is_div"] = True
                true_count += 1
        
        # THE FLIP: Predicate 5 is the causal variable
        # Clean: Predicate 5 is TRUE
        p5_clean = predicates[4].copy()
        p5_clean["num"] = p5_clean["num"] - (p5_clean["num"] % p5_clean["div"])
        p5_clean["is_div"] = True
        
        # Corrupt: Predicate 5 is FALSE
        p5_corrupt = predicates[4].copy()
        if p5_corrupt["num"] % p5_corrupt["div"] == 0: p5_corrupt["num"] += 1
        p5_corrupt["is_div"] = False

        # Calculate final answers (XOR logic: True if Odd count)
        clean_true_count = true_count + 1
        corrupt_true_count = true_count + 0
        
        ans_clean = "True" if (clean_true_count % 2 != 0) else "False"
        ans_corrupt = "True" if (corrupt_true_count % 2 != 0) else "False"

        # Render
        # We construct the list string carefully
        def build_str(preds, p5):
            all_p = preds[:4] + [p5]
            s = ""
            for i, p in enumerate(all_p):
                s += f"P{i+1}: {p['num']} divisible by {p['div']}? "
            s += "Answer True if an odd number of checks are valid, else False."
            return s

        return {
            "task_class": "Parity_PAT",
            "clean": {
                "prompt": build_str(predicates, p5_clean),
                "answer": ans_clean
            },
            "corrupt": {
                "prompt": build_str(predicates, p5_corrupt),
                "answer": ans_corrupt
            },
            "patching_target": "predicate_5_validity"
        }
