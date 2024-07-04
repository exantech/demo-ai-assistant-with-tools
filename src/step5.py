from langchain_community.chat_models import ChatLlamaCpp
from langchain_core.prompts import PromptTemplate

from .utils.hf_model import download_hf_model

model_path = download_hf_model(
    repo_id="MaziyarPanahi/Mistral-7B-Instruct-v0.3-GGUF",
    filename="*Q4_K_M.gguf"
)

llm = ChatLlamaCpp(
    temperature=0.5,
    model_path=model_path,
    n_ctx=8192,
    n_gpu_layers=-1,
    n_batch=1024,  # Should be between 1 and n_ctx, consider the amount of VRAM in your GPU.
    max_tokens=512,
    n_threads=4,
    repeat_penalty=1.5,
    top_p=0.95,
    top_k=40,
    verbose=False,
)

PROMPT = PromptTemplate.from_template(
    "<s> [INST] {system} {text} [/INST] "
)

prompt = PROMPT.format(
    system = "You are a helpful assistant that translates English to French. Translate the user sentence.\n",
    text = "I haven't eat for six days."
)

ai_msg = llm.invoke(prompt)

print (ai_msg.content.strip())
