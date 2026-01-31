from task_generation import MechanisticTaskGenerator

gen = MechanisticTaskGenerator(seed=42)
cblg_task = gen.generate_cblg_pair()
print(cblg_task)