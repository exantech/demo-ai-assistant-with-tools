from decouple import config
from griptape.config import StructureConfig
from griptape.drivers import OllamaPromptDriver
from griptape.memory.structure import ConversationMemory
from griptape.rules import Rule, Ruleset
from griptape.structures import Agent
from griptape.tools import (Calculator, FileManager, TaskMemoryClient,
                            WebScraper)

from .utils.prices.tool import Price
from .utils.search.tool import WebSearch

SERPER_API_KEY = config("SERPER_API_KEY", cast=str)
CMC_API_KEY = config("CMC_KEY", cast=str)

memory = ConversationMemory()
tools = [
    WebScraper(off_prompt=False),
    WebSearch(serper_api_key=SERPER_API_KEY, off_prompt=False),
    Price(api_key=CMC_API_KEY, off_prompt=False)
]

prompt_agent = Agent(
    config=StructureConfig(
        prompt_driver=OllamaPromptDriver(
            model="taozhiyuai/hermes-2-pro-llama-3:8b-q5_k_m", temperature=0.4
        ),
    ),
    conversation_memory=memory,
)

tools_agent = Agent(
    config=StructureConfig(
        prompt_driver=OllamaPromptDriver(
            model="taozhiyuai/hermes-2-pro-llama-3:8b-q5_k_m", temperature=0.01
        ),
    ),
    rules=[Rule("Never use task memory.")],
    tools=tools,
    conversation_memory=memory,
)

tools_list = "\n".join([f"{x.activity_name(y)}: {x.activity_description(y)}"  for x in tools for y in x.activities()])

router = Agent(
    config=StructureConfig(
        prompt_driver=OllamaPromptDriver(
            model="phi3:3.8b-mini-128k-instruct-q4_K_M",
            temperature=0.5
        ),
    ),
)

while True:
    q = input("> ")

#     resp = router.run(f"""Your main task is to route user queries to LLM agents with support for function calls and without it.
# Determine if you need to call ANY of the functions listed to answer this user query.
# You don't have any access to real world without these functions. Output only 'YES' or 'NO'.

# Functions:
# {tools_list}

# User query: {q}""")

#     use_tools = resp.output.value.startswith('YES')

#     response = tools_agent.run(q) if use_tools else prompt_agent.run(q)
    response = tools_agent.run(q)
    print (f"< {response.output.value}")
