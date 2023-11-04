import warnings
warnings.filterwarnings("ignore")
import os,json 

script_dir = os.path.dirname(os.path.abspath(__file__))
relative_path = os.path.join(script_dir, ".cache/huggingface")
os.environ["HF_HOME"] = relative_path

## ---

# https://www.kreactiva.com/ia/auto-gpt-que-es-auto-gpt-como-funciona-y-como-instalarlo/
# https://cryptocity.press/noticias/como-instalar-y-usar-auto-gpt-una-guia-paso-a-paso
from transformers import GPT2LMHeadModel, GPT2Tokenizer

tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
model = GPT2LMHeadModel.from_pretrained('gpt2')

import torch

def generate_text(prompt, max_length=200):
    input_ids = tokenizer.encode(prompt, return_tensors='pt')
    output = model.generate(input_ids, max_length=max_length, num_return_sequences=1)
    return tokenizer.decode(output[0], skip_special_tokens=True)

# Ejemplo de uso
generated_text = generate_text('El cambio climático es un problema', max_length=50)
print(generated_text)