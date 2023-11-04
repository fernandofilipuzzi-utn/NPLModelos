import warnings
warnings.filterwarnings("ignore")
import os,json 

script_dir = os.path.dirname(os.path.abspath(__file__))
relative_path = os.path.join(script_dir, ".cache/huggingface")
os.environ["HF_HOME"] = relative_path

import os
import sys

import transformers


class AutoGPTBot:

    def __init__(self, model_name):
        self.model_name = model_name
        self.model = transformers.AutoModelForSeq2SeqLM.from_pretrained(self.model_name)

    def generate_code(self, prompt):
        """
        Genera código a partir de una indicación.

        Args:
            prompt: La indicación para el código.

        Returns:
            El código generado.
        """

        tokenizer = transformers.AutoTokenizer.from_pretrained(self.model_name)
        input_ids = tokenizer(prompt, return_tensors="pt").input_ids
        outputs = self.model.generate(input_ids, max_length=1000, do_sample=True)
        return tokenizer.decode(outputs[0], skip_special_tokens=True)


def main():
    """
    Ejemplo de uso del bot de Auto-GPT.
    """

    bot = AutoGPTBot("gpt2")

    print(bot.generate_code("Escribe un programa que imprima 'Hola, mundo!'"))


if __name__ == "__main__":
    main()