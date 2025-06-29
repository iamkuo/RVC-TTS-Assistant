from ollama import chat

for chunk in chat('mistral', messages=[{"role": "user", "content": "how to walk"}], stream=True):
  print(chunk['message']['content'], end='', flush=True)