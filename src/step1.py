from llama_cpp import Llama

llm = Llama.from_pretrained(
    repo_id="MaziyarPanahi/Mistral-7B-Instruct-v0.3-GGUF",
    filename="*Q4_K_M.gguf",
    verbose=True,
    n_gpu_layers=-1,
)

print ("Model loaded")

output = llm(
    "Q: Name the planets in the solar system? A: ",  # Prompt
    max_tokens=64,
    stop=["Q:", "\n", ],  # Stop generating just before the model would generate a new question
    echo=True,
)
print(output)
print()
print(output['choices'][0]['text'])
