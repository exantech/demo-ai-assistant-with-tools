
# LLM Function Calling Demo (AI Assistant) #
This demo project illustrates the concept of function calling in Large Language Models (LLMs). We will explore various steps to implement function calling manually and utilize popular LLM frameworks to build an AI assistant with function calling capabilities, primarily focusing on smaller local open-weight LLMs.

## Structure ##
### Step 1 ###
We start by loading the quantized model of Mistral-7b using llama-cpp-python on top of llama.cpp. This step demonstrates how to load a model from HuggingFace and provide it with a basic prompt.

### Step 2 ###
Next, we define a custom system prompt and include function descriptions in JSON Schema format. With this setup, the model can identify and define a function call from a list in its responses, thanks to few-shot prompting with examples.

### Step 3 ###
We then add methods to extract function calls from the model's response and execute them. This allows us to display the actual data received from an external function called by the model.

### Step 4 ###
A loop is introduced to feed the model with the results of the external function call. This enables us to obtain a final response from the model using the provided data. Additionally, we demonstrate that the model can determine whether it needs to call a function or respond based on its general knowledge.

### Step 5 ###
We integrate the LangChain framework to show how to load and interact with the model using LangChain primitives. A custom prompt template is employed to optimize the performance of Mistral.

### Step 6 ###
We expand the functionality by adding more LangChain abstractions to build a simple LLM chat bot with message history support.

### Step 7 ###
Although further development with LangChain is possible, we switch to the Griptape framework for this step. Griptape is simpler for pipelines and agents that require function calling ("tools"). Instead of manually loading the model, we use the local Ollama with the necessary models loaded. Using Griptape and Ollama, we build another simple chat bot.

### Step 8 ###
We introduce several useful tools (both Griptape-provided and custom), such as a web search tool or a currency price getter. These tools are integrated into our agent flow to observe how the model utilizes them. Note that smaller quantized models may not always perform optimally, as they might make unnecessary function calls or mishandle parameters in complex JSON Schemas.

### Step 9 ###
Finally, we demonstrate how changing only the model provider from Ollama to OpenAI can significantly improve performance. This highlights that while the rest of the code remains unchanged, using a more advanced model like GPT-4 yields better reasoning and function calling capabilities compared to the basic Mistral.
