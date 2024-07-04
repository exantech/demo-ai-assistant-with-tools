from griptape.config import StructureConfig
from griptape.drivers import OllamaPromptDriver
from griptape.memory.structure import ConversationMemory
from griptape.structures import Agent

agent = Agent(
    config=StructureConfig(
        prompt_driver=OllamaPromptDriver(
            model="mistral:7b-instruct-v0.3-q4_K_M",
        ),
    ),
    conversation_memory=ConversationMemory()
)

while True:
    q = input("> ")
    response = agent.run(q)

    print (f"< {response.output.value}")