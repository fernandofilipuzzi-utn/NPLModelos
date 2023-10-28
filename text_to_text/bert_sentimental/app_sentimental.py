#app_sentimental.py

import warnings
warnings.filterwarnings("ignore")
import os

script_dir = os.path.dirname(os.path.abspath(__file__))
relative_path = os.path.join(script_dir, ".cache/huggingface")
os.environ["HF_HOME"] = relative_path

##----------

from transformers import AutoTokenizer, BertForSequenceClassification, AdamW
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import torch

model_base = "bert-base-uncased"
model_base = "mrm8488/distill-bert-base-spanish-wwm-cased-finetuned-spa-squad2-es"
#model_base = "Toshifumi/bert-base-multilingual-cased-finetuned-emotion"

#modelo="edumunozsala/beto_sentiment_analysis_es"
#modelo="nlptown/bert-base-multilingual-uncased-sentiment"
# Preparación de datos y tokenización

tokenizer = AutoTokenizer.from_pretrained(model_base)
model = BertForSequenceClassification.from_pretrained(model_base, num_labels=3,ignore_mismatched_sizes=True)  # 3 etiquetas: positivo, negativo, neutro

# Datos de entrenamiento (reemplaza esto con tus propios datos)
#train_texts = ["Esto es un gran producto!", "No me siento feliz con este producto", "No encontré ventajas comparativas"]
#train_labels = [0, 1, 2]  # 0: positivo, 1: negativo, 2: neutro

# Datos de entrenamiento (reemplaza esto con tus propios datos)
# Datos de entrenamiento
train_texts = [
    "esto es muy bueno!",
    "está bien",
    "no me siento feliz",
    "no encontré ventajas comparativas",
    "amo este producto",
    "me resultó horrible.",
    "está bien por lo que vale",
    "muy bueno",
    "demasiados problemas",
    "regular",
    "excelente calidad",
    "calidad deficiente",
    "calidad regular",
    "es increíble",
    "no lo recomendaría",
    "no está mal",
    "no es lo que esperaba",
    "buen precio",
    "no vale la pena ningún comentario",
    "mejor de lo que pensaba",
    "pobre calidad",
    "odio a este producto",
    "odio",
    "amor",
    "excelente",
    "si hubiera sabido que era tan bueno!, me dejó sin palabras!",
    "hace un rato encargué otro",
    "voy a volver a comprarlo nuevamente",
    "los voy a demandar",
    "me resultó tremendamente útil",
    "no ha destacado demasiado",
    "odio",
    "amor",
    "desastre",
    "espanto"
]

train_labels = [0, 2, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 1, 0, 1, 0, 1, 1,1,0,0,0,0,0,1,0,2,1,0,1,1]


# División de datos en conjunto de entrenamiento y prueba
train_texts, val_texts, train_labels, val_labels = train_test_split(train_texts, train_labels, test_size=0.2, random_state=42)

train_inputs = tokenizer(train_texts, return_tensors="pt", padding=True, truncation=True)
val_inputs = tokenizer(val_texts, return_tensors="pt", padding=True, truncation=True)

train_labels = torch.tensor(train_labels)
val_labels = torch.tensor(val_labels)

print(val_labels)

# Entrenamiento
optimizer = AdamW(model.parameters(), lr=1e-5)

for epoch in range(20):  # Número de épocas
    outputs = model(**train_inputs, labels=train_labels)
    loss = outputs.loss
    loss.backward()
    optimizer.step()
    print(f"Epoch {epoch+1} - Loss: {loss.item()}")

# Evaluación en el conjunto de prueba
with torch.no_grad():
    val_outputs = model(**val_inputs)
    predicted_labels = torch.argmax(val_outputs.logits, dim=1)
    accuracy = accuracy_score(val_labels, predicted_labels)
    print(f"Accuracy on validation set: {accuracy:.2f}")

# Predicciones en nuevos textos
new_texts = ["Es muy bueno", "Es malo", "Me parece bien.","No lo sé"]
new_inputs = tokenizer(new_texts, return_tensors="pt", padding=True, truncation=True)
new_outputs = model(**new_inputs)
predicted_labels = torch.argmax(new_outputs.logits, dim=1)

for text, label in zip(new_texts, predicted_labels):
    sentiment = "Positivo" if label == 0 else "Negativo" if label == 1 else "Neutro"
    print(f"Texto: {text} - Sentimiento: {sentiment}")
