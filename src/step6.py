from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_community.llms.llamacpp import LlamaCpp
from langchain_core.callbacks import StdOutCallbackHandler
from langchain_core.messages import SystemMessage
from langchain_core.prompts.chat import (ChatPromptTemplate,
                                         HumanMessagePromptTemplate,
                                         MessagesPlaceholder)
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_experimental.chat_models.llm_wrapper import ChatWrapper

from .utils.hf_model import download_hf_model

template_messages = [
    SystemMessage(content="You are a helpful assistant speaking in Spanish."),
    MessagesPlaceholder(variable_name="chat_history"),
    HumanMessagePromptTemplate.from_template("{text}"),
]
prompt_template = ChatPromptTemplate.from_messages(template_messages)


class Mistral(ChatWrapper):
    @property
    def _llm_type(self) -> str:
        return "mistral"

    sys_beg: str = "[INST] "
    sys_end: str = "\n"
    ai_n_beg: str = " "
    ai_n_end: str = " </s>"
    usr_n_beg: str = " [INST] "
    usr_n_end: str = " [/INST]"
    usr_0_beg: str = ""
    usr_0_end: str = " [/INST]"


store = {}
def get_session_history(session_id: str) -> ChatMessageHistory:
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]


model_path = download_hf_model(
    repo_id="MaziyarPanahi/Mistral-7B-Instruct-v0.3-GGUF",
    filename="*Q4_K_M.gguf"
)

llm = LlamaCpp(
    temperature=0.1,
    model_path=model_path,
    n_ctx=8192,
    n_gpu_layers=-1,
    n_batch=1024,
    max_tokens=512,
    n_threads=8,
    repeat_penalty=1.1,
    top_p=0.95,
    top_k=40,
    verbose=False
)

model = Mistral(llm=llm)
runnable = prompt_template | model
runnable_with_history = RunnableWithMessageHistory(
    runnable,
    get_session_history,
    input_messages_key="text",
    history_messages_key="chat_history",
)

while True:
    q = input("> ")
    ai_msg = runnable_with_history.invoke({"text": q}, config={"configurable": {"session_id": "1"}})
    print (f"< {ai_msg.content.strip()}")
