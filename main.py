from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import torch
from transformers import AutoTokenizer
from transformers import BertModel
from torch import nn

#inicializacion
N_CLASSES = 2
MAX_LEN = 200
PRE_TRAINED_MODEL = 'bert-base-cased'
MODEL_PATH= 'bert_sentiment_model.pth'
TOKENIZER_PATH = 'bert_tokenizer'
#device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Permitir peticiones desde cualquier origen
    allow_credentials=True,
    allow_methods=["*"],  # Permitir todos los métodos (GET, POST, etc.)
    allow_headers=["*"],  # Permitir todos los headers
)

#clase del modelo
class BERTSentimentClassifier(nn.Module):
    def __init__(self, n_classes):
        super(BERTSentimentClassifier, self).__init__()
        self.bert = BertModel.from_pretrained(PRE_TRAINED_MODEL)
        self.drop = nn.Dropout(p=0.3)
        self.linear = nn.Linear(self.bert.config.hidden_size, n_classes)

    def forward(self, input_ids, attention_mask):
        '''_, cls_output = self.bert(
                input_ids = input_ids,
                attention_mask = attention_mask
        )'''
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        # Intentamos usar pooler_output, pero si es None, tomamos el CLS token de last_hidden_state
        cls_output = outputs.pooler_output if outputs.pooler_output is not None else outputs.last_hidden_state[:, 0, :]

        drop_output = self.drop(cls_output)
        output = self.linear(drop_output)
        return output
# Cargar modelo preentrenado
model = BERTSentimentClassifier(N_CLASSES)
model.load_state_dict(torch.load(MODEL_PATH, map_location='cpu')) #cambia cpu por device que esta comentado arriba
model.to('cpu')
print("Modelo cargado exitosamente.")

# Cargar tokenizer
tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_PATH)
print("Tokenizer cargado exitosamente.")

def predict_sentiment(text, model, tokenizer, device='cpu'):
    encoding = tokenizer.encode_plus(
        text,
        max_length=MAX_LEN,
        truncation=True,
        add_special_tokens=True,
        return_token_type_ids=False,
        padding='max_length',
        return_attention_mask=True,
        return_tensors='pt'
    )
    
    input_ids = encoding["input_ids"].to(device)
    attention_mask = encoding["attention_mask"].to(device)

    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        prediction = torch.argmax(outputs, dim=1).item()
    
    return "Positive" if prediction == 1 else "Negative"

@app.get('/predecir/')
async def predecir(text: str):
    # Ejemplo
    test_text = text
    prediccion = f"{predict_sentiment(test_text, model, tokenizer, 'cpu')}" #cambia cpu por device
    return {'Sentimiento': prediccion}

#estructura del json
from pydantic import BaseModel
class SentimentRequest(BaseModel):
    text: str
@app.post('/predict/')
async def predecir(request: SentimentRequest):
    test_text = request.text  # Obtener el texto del cuerpo
    prediccion = predict_sentiment(test_text, model, tokenizer, 'cpu') #cambia cpu por device
    return {'Sentimiento': prediccion}

import uvicorn
if __name__ == "__main__":
    port = int(os.getenv("PORT", 10000))  # Usa el puerto asignado por Render
    uvicorn.run(app, host="0.0.0.0", port=port)