from transformer_lens import HookedTransformer

def loadModel(model_name):
    """
    load pre-trained model from hugging face
    :param model: str (1 of 4 allowed models)
    :returns: 
    """
    allowed_models = {
        "microsoft/phi-1_5",
        "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        "gemma-2-2b",
        "Qwen/Qwen1.5-1.8B"
    }
    if model_name not in allowed_models:
        print(f"Error: {model_name} isn't a specified model for this experiment")
        return 0
    model = HookedTransformer.from_pretrained(model_name, device="cuda") # loads to GPU if available, otherwise CPU
    return model
    
    