from decouple import config
from griptape.config import StructureConfig
from griptape.drivers import OllamaPromptDriver, OpenAiChatPromptDriver
from griptape.memory.structure import ConversationMemory
from griptape.rules import Rule, Ruleset
from griptape.structures import Agent
from griptape.tools import (Calculator, FileManager, TaskMemoryClient,
                            WebScraper)

from .utils.prices.tool import Price
from .utils.search.tool import WebSearch

SERPER_API_KEY = config("SERPER_API_KEY", cast=str)
CMC_API_KEY = config("CMC_KEY", cast=str)
OPENAI_KEY = config("OPENAI_KEY", cast=str)

memory = ConversationMemory()
tools = [
    WebScraper(off_prompt=False),
    WebSearch(serper_api_key=SERPER_API_KEY, off_prompt=False),
    Price(api_key=CMC_API_KEY, off_prompt=False)
]

tools_agent = Agent(
    config=StructureConfig(
        prompt_driver=OpenAiChatPromptDriver(model='gpt-4o', api_key=OPENAI_KEY),
    ),
    rules=[Rule("Never use task memory.")],
    tools=tools,
    conversation_memory=memory,
)

while True:
    q = input("> ")
    response = tools_agent.run(q)
    print (f"< {response.output.value}")
