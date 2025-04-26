import torch

def gen_with_adapter_steering(
    model,
    tokenizer,
    prompt,
    steering_layer: int = None,
    steering_direction: torch.Tensor = None,
    steering_strength: float = 0.1,
    max_new_tokens: int = 100,
    num_return_sequences: int = 1,
    temperature: float = 1
    ):
    
    if steering_layer is None:
        raise ValueError("steering_layer must be provided")
    if steering_direction is None:
        raise ValueError("steering_direction must be provided")
    
    # Tokenize the prompt
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    # divide steering strength by norm of steering direction
    steering_strength = steering_strength / torch.norm(steering_direction)
    steering_vector = steering_direction * steering_strength
    steering_vector = steering_vector.reshape(1, 1, -1).to(model.device)
    # def collect_hook(self, input, output):
    #     return output

    def add_hook(self, input, output):
        output[:,:,:] += steering_vector


    # Use context manager to ensure hooks are properly removed after generation
    hooks = []
    try:
        # Register the hook and store the handle
        hook_handle = model.model.model.layers[steering_layer].mlp.down_proj.register_forward_hook(add_hook)
        hooks.append(hook_handle)
        
        # Generate text with the hook applied
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens, 
            num_return_sequences=num_return_sequences, 
            temperature=temperature,
            do_sample=True,
            top_p=0.95,
            use_cache=True
        )  
    finally:
        # Clean up by removing all hooks
        for hook in hooks:
            hook.remove()
    #batch decode
    return [tokenizer.decode(output, skip_special_tokens=True) for output in outputs]