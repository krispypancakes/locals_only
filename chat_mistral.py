from llama_cpp import Llama
import os
import argparse


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

def get_args():
  parser = argparse.ArgumentParser()
  parser.add_argument("--temp", type=float, default=0.0, required=False)
  parser.add_argument("--personality", type=str, required=False)

  args = parser.parse_args()
  return args

def main():
  memory, llm = init_cpp()
  args = get_args()
   
  while True:
    out = llm.create_chat_completion(messages=memory, temperature=args.temp)["choices"][0]["message"]["content"]
    # add output to memory
    memory.append({"role":"assistant", "content":out})
    print(" [AI]: " + out + "\n\n")
    voice = "say " + out.replace("'", "").replace("\n", " ")
    # speak
    os.system(voice)
    # ask for input and add to memory
    prompt = input("\n [USER]: ")
    memory.append({"role":"user", "content":prompt})
    

if __name__ == "__main__":
  main()
  