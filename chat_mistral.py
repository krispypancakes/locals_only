from llama_cpp import Llama
import os


def init_cpp():
  model_path = ".models/mistral-7b-instruct-v0.1.Q6_K.gguf"
  llm = Llama(model_path=model_path, n_ctx=8192, n_batch=512,  n_threads=10, n_gpu_layers=4, verbose=False, seed=42, stream=True, chat_format="llama-2")

  human_input = input("type something to talk to Mistral\n")
  memory = []
  system_msg = os.getenv("SYSTEM_MSG")
  if system_msg:
    memory.append({"role":"system", "content":system_msg})
  memory.append({"role":"user", "content":human_input})
  
  return memory, llm

def main():
  memory, llm = init_cpp()
  memory, llm = init_plain()
  
  while True:
    out = llm.create_chat_completion(messages=memory)["choices"][0]["message"]["content"]
    memory.append({"role":"assistant", "content":out})
    out = llm()
    print(out + "\n")
    voice = "say " + out.replace("'", "").replace("\n", " ")
    
    os.system(voice)
    memory.append({"role":"user", "content":input("")})
    

if __name__ == "__main__":
  main()
  