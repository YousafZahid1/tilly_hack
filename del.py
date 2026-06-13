KEY = 

LLM|1668678254439551|PEVFjzLAU_oK2Aje4hmlV0qrB9Q






from llama_api_client import LlamaAPIClient

client = LlamaAPIClient(
    api_key="LLM|1924423828467790|sdaQbCuuHgeCkyWzvWy91Kocy-c",
    base_url="https://api.llama.com/v1/",
)


response = client.chat.completions.create(
    model="Llama-4-Maverick-17B-128E-Instruct-FP8",
    messages=[
        {"role": "user", "content": "Hello Llama! Can you give me a quick intro?"},
    ],
)


# print(response) for the whole thing


text = response.completion_message.content.text
print(text)

