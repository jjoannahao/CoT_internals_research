import json

from cot_baseline import CoTBaselineRunner
from task_generation import MechanisticTaskGenerator

# ------- TEST LINEAR SYMBOLIC TASK -------- #
cot_prompt = """
Q: Start with 5. Subtract 2. Multiply by 3.
Reasoning: Start with 5. 5 − 2 = 3. 3 * 3 = 9. 
A: The answer is 9.

Q: Start with 7. Multiply by 2. Add 10.
Reasoning: Start with 7. 7 * 2 = 14. 14 + 10 = 24.
A: The answer is 24.

Q: Start with 15. Subtract 4. Divide by 11.
Reasoning: Start with 15. 15 − 4 = 11. 11 ÷ 11 = 1.
A: The answer is 1.

Q: Start with 4. Multiply by 3. Add 10. Divide by 2. Subtract 5.
Reasoning: Start with 4. 4 * 3 = 12. 12 + 10 = 22. 22 ÷ 2 = 11. 11 − 5 = 6.
A: The answer is 6.

Q: Mia has 12 apples. She gives 5 to her friend. Then she buys 8 more. How many apples does she have now?
Reasoning: Start with 12 apples. 12 − 5 = 7. 7 + 8 = 15.
A: Mia has 15 apples now.

Q: A number is doubled, then 7 is added. If the result is 21, what was the original number?
Reasoning: Let the number be x. 2 * x + 7 = 21. 2 * x = 14. x = 7.
A: The original number is 7.

Q: Sarah has 3 boxes with 6 candies each. She eats 5 candies. How many candies are left?
Reasoning: Start with 3 * 6 = 18 candies. 18 − 5 = 13.
A: Sarah has 13 candies left.

Q: Tom has twice as many pencils as Lisa. Together they have 18 pencils. How many pencils does Tom have?
Reasoning: Let Lisa have x pencils. Then Tom has 2 * x pencils. x + 2 * x = 18. 3 * x = 18. x = 6. Tom has 2 * 6 = 12 pencils. The answer is 12.
A: Tom has 12 pencils.

Q: Start with 10. Add 5. Multiply by 2.
Reasoning:
A:
"""

if __name__ == "__main__":
    gen = MechanisticTaskGenerator(seed=42)
    # set up test dataset
    sample_dataset = []
    for i in range(4):
        sample_dataset.append(gen.generate_linear_pair())
        sample_dataset.append(gen.generate_cblg_pair())
        sample_dataset.append(gen.generate_multiway_pair())
        sample_dataset.append(gen.generate_parity_pat_pair())
    
    print()
    
    # check CoT
    phi_runner = CoTBaselineRunner()
    output = phi_runner.model.generate(cot_prompt, max_new_tokens=50, temperature=0)
    print(output)

# --- results in output file:
# phi_runner.run_baseline(sample_dataset, output_file="gpt2_xl_baseline.jsonl")
# with open("phi-1_5_baseline.jsonl", "r") as f:
#     formatted = [json.loads(line) for line in f]
#     print(json.dumps(formatted, indent=2))

# In real usage: 
# import json
# with open("my_dataset.jsonl", "r") as f:
#    dataset = [json.loads(line) for line in f]