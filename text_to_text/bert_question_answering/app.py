import warnings
warnings.filterwarnings("ignore")
import os,json 

script_dir = os.path.dirname(os.path.abspath(__file__))
relative_path = os.path.join(script_dir, ".cache/huggingface")
os.environ["HF_HOME"] = relative_path

## ---

#https://github.com/alexaapo/BERT-based-pretrained-model-using-SQuAD-2.0-dataset/blob/main/Evaluate_Existed_Fine_Tuned_Bert.ipynb

import torch
from transformers import AutoTokenizer,BertTokenizerFast, BertForQuestionAnswering

tokenizer = AutoTokenizer.from_pretrained('bert-large-uncased-whole-word-masking-finetuned-squad')
model = BertForQuestionAnswering.from_pretrained('bert-large-uncased-whole-word-masking-finetuned-squad')

context = "Hi! My name is Alexa and I am 21 years old. I used to live in Peristeri of Athens, but now I moved on in Kaisariani of Athens."
query = "Where does Alexa live now?"
query = "How old is Alexa?"

inputs = tokenizer.encode_plus(query, context, return_tensors='pt')
outputs = model(**inputs)
answer_start = torch.argmax(outputs[0])  # get the most likely beginning of answer with the argmax of the score
answer_end = torch.argmax(outputs[1]) + 1 

answer = tokenizer.convert_tokens_to_string(tokenizer.convert_ids_to_tokens(inputs['input_ids'][0][answer_start:answer_end]))

#----

print("Respuesta:")
print(answer)