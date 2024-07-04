import json
import re
from typing import List

import requests
from decouple import config
from llama_cpp import Llama

llm = Llama.from_pretrained(
    repo_id="MaziyarPanahi/Mistral-7B-Instruct-v0.3-GGUF",
    filename="*Q4_K_M.gguf",
    verbose=False,
    n_gpu_layers=-1,
)

print("Model loaded")


def extract_function_calls(completion: str):
    if isinstance(completion, str):
        content = completion
    else:
        content = completion.content

    # Multiple functions lookup
    mfn_pattern = r"<multiplefunctions>(.*?)</multiplefunctions>"
    mfn_match = re.search(mfn_pattern, content, re.DOTALL)

    # Single function lookup
    single_pattern = r"<functioncall>(.*?)</functioncall>"
    single_match = re.search(single_pattern, content, re.DOTALL)

    functions = []

    if not mfn_match and not single_match:
        # No function calls found
        return None
    elif mfn_match:
        # Multiple function calls found
        multiplefn = mfn_match.group(1)
        for fn_match in re.finditer(
            r"<functioncall>(.*?)</functioncall>", multiplefn, re.DOTALL
        ):
            fn_text = fn_match.group(1)
            try:
                functions.append(json.loads(fn_text.replace("\\", "")))
            except json.JSONDecodeError:
                pass  # Ignore invalid JSON
    else:
        # Single function call found
        fn_text = single_match.group(1)
        try:
            functions.append(json.loads(fn_text.replace("\\", "")))
        except json.JSONDecodeError:
            pass  # Ignore invalid JSON
    return functions


def execute_function(function_list: list):
    for function_dict in function_list:
        function_name = function_dict["name"]
        arguments = function_dict["arguments"]

        # Check if the function exists in the current scope
        if function_name in globals():
            func = globals()[function_name]

            # Call the function with the provided arguments
            result = func(arguments)
            return result
        else:
            return {"error": f"Function '{function_name}' not found."}


def fn_get_currency_price():
    return {
            "name": "get_currency_price",
            "description": "Get the current price of the requested cryptocurrency",
            "parameters": {
                "type": "object",
                "properties": {
                    "ticker": {
                        "type": "string",
                        "description": "The cryptocurrency ticker",
                    },
                },
                "required": ["ticker"],
            },
    }


def get_currency_price(params: dict) -> str:
    print (">>> calling get_currency_price", params)

    key = config("CMC_KEY", cast=str)
    resp = requests.get(
        "https://pro-api.coinmarketcap.com/v2/cryptocurrency/quotes/latest",
        params={"symbol": params["ticker"]},
        headers={"X-CMC_PRO_API_KEY": key},
        timeout=30
    )
    resp.raise_for_status()
    data = resp.json()
    quote = str(data['data'][params['ticker']][0]['quote']['USD']['price'])

    print (f"<<< result of get_currency_price = {quote}")
    return quote


def generate_output(messages: List[str]):
    functions = [fn_get_currency_price()]

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

    If you can answer without calling a function, just do it.
    """

    prompt = f"""[INST] {system_prompt} {messages[0]} [/INST] """
    for idx, msg in enumerate(messages[1:]):
        if idx % 2 == 0:
            prompt += (msg + "</s>")
        else:
            prompt += f" [INST] {msg} [/INST]"

    output = llm(
        prompt,
        stop=["</s>", "[INST]"],
        echo=False,
        max_tokens=512,
        temperature=0.0
    )

    text = output["choices"][0]["text"].strip()
    # print(text)

    calls = extract_function_calls(text)
    if calls:
        # If function calls are found, execute them and return the response
        fn_response = execute_function(calls)
        return fn_response, text
    else:
        # If no function calls are found, return the completion content
        return None, text


if __name__ == "__main__":
    messages = ["What's the current price of BTC?"]
    # messages = ["How are you today?"]

    print (f"> {messages[0]}")

    fn_call, response = generate_output(messages)
    messages.append(response)

    if fn_call:
        messages.append(f"Function call result: {fn_call}")
        fn_call, response = generate_output(messages)

    print (f"< {response}")