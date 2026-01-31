# install relevant libraries
```bash
pip install --upgrade huggingface-hub
pip install transformer_lens
```
```datasets``` and ```transformers``` not needed since we're using ```transformer_lens``` and generating a synthetic dataset of tasks

# set up & generate dataset
models to be tested:
- phi-1.5
- TinyLlama-1.1B (or Llama-3-8B if I can get a hold of compute resources)
- Qwen-2.5-1.5B (or Qwen-2.5-3B)
- Gemma-2-2B

4 task types
- linear symbolic reasoning (sanity check)
- conditional branching with latent gating (CBLG) (binary gating to stress test core faithfulness)
- multi-way branching (scalability of grounded explanations)
- parity PAT (parallel aggregation task) (testing limits of faithfulness via component removal)

other potential tasks:
- CLUTRR
- SCAN / PCFG / COGS
- soft gating for partial causality?

**make generator class** 

**create dataset of 500 examples per task type for total of 2000 examples**


# get standard CoT baseline
1. make model runners
2. test each task type with 1 CoT prompt
3. 


# activation patching with causal tracing

# internals-ground CoT generation

# faithfulness with component ablation

# metrics & comparisons

# extensions: partial grounding? cross-task generalization?