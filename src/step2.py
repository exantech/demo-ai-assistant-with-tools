from llama_cpp import Llama

llm = Llama.from_pretrained(
    repo_id="MaziyarPanahi/Mistral-7B-Instruct-v0.3-GGUF",
    filename="*Q4_K_M.gguf",
    verbose=False,
    n_gpu_layers=-1,
)

print("Model loaded")

def fn_get_current_weather():
    return {
            "name": "get_current_weather",
            "description": "Get the current weather",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {
                        "type": "string",
                        "description": "The city and state, e.g. San Francisco, CA",
                    },
                    "format": {
                        "type": "string",
                        "enum": ["celsius", "fahrenheit"],
                        "description": "The temperature unit to use. Infer this from the users location.",
                    },
                },
                "required": ["location", "format"],
            },
    }


functions = [fn_get_current_weather()]

fn = """{"name": "function_name", "arguments": {"arg_1": "value_1", "arg_2": value_2, ...}}"""
fn_str = "\n".join([str(x) for x in functions])

system_prompt = f"""
You are a helpful assistant with access to the following functions:

{fn_str}

To use these functions respond with:

<multiplefunctions>
    <functioncall> {fn} </functioncall>
    <functioncall> {fn} </functioncall>
    ...
</multiplefunctions>

Edge cases you must handle:
- If there are no functions that match the user request, you will respond politely that you cannot help.
- If the user has not provided all information to execute the function call, ask for more details. Only, respond with the information requested and nothing else.
- If asked something that cannot be determined with the user's request details, respond that it is not possible to fullfill the request and explain why.
"""

user_prompt = "What's the current weather in New York?"



output = llm(
   f"""[INST] {system_prompt} {user_prompt} [/INST] """,
   stop=["</s>", "[INST]"],
   echo=False,
   max_tokens=512,
   temperature=0.1
)

print(output['choices'][0]['text'])
