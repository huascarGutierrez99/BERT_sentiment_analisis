from transformers import AutoTokenizer,  AutoModelForQuestionAnswering, pipeline
from transformers import BertModel, BertTokenizer, AdamW, get_linear_schedule_with_warmup
import torch
import numpy as np
from sklearn.model_selection import train_test_split
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from textwrap import wrap

#inicializacion
RANDOM_SEED = 42
N_CLASSES = 2
MAX_LEN = 200
BATCH_SIZE = 16 
DATASET_PATH = 'BERT_sentiment_IMDB_Dataset.csv'

np.random.seed(RANDOM_SEED)
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(device)

Dataframe = pd.read_csv(DATASET_PATH)
Dataframe = Dataframe[0:10000]

print(Dataframe.head())
print(Dataframe.shape)
"""print('\n'.join(wrap(Dataframe['review'][200])))"""

Dataframe['label'] = (Dataframe['sentiment'] == 'positive').astype(int)
Dataframe.drop(columns=['sentiment'], inplace=True)
print(Dataframe.head())

PRE_TRAINED_MODEL = 'bert-base-cased'
tokenizer = AutoTokenizer.from_pretrained(PRE_TRAINED_MODEL)

sample_text = 'I really loved this movie!'

tokens = tokenizer.tokenize(sample_text)
token_ids = tokenizer.convert_tokens_to_ids(tokens=tokens)
print('Frase: ',sample_text)
print('Tokens: ', tokens)
print('Tokens numericos: ',token_ids)

#codificacion para introducir a BERT
encoding = tokenizer.encode_plus(
    sample_text,
    max_length=10, #limite maximo de palabras y simbolos
    truncation=True, #en caso de superar el limite, esta funcion permite truncarlo
    add_special_tokens=True, #los tokens CLS y ESP 
    return_token_type_ids=False,
    padding='max_length', #en caso de un maximo de tokens, que este lo llene con PADs vacios
    return_attention_mask=True, #que preste atencion unicamente al texto
    return_tensors='pt'  #tensor final
)

print(tokenizer.convert_ids_to_tokens(encoding['input_ids'][0]))
print(encoding['input_ids'][0])
print(encoding['attention_mask'][0])

#CREACION DEL DATASET

class IMDBDataset(Dataset):
    def __init__ (self, reviews, labels, tokenizer, max_length):
        self.reviews = reviews
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.reviews)

    def __getitem__(self, item): #con esto devolvemos paquetes de 16 en 16
        review = str(self.reviews[item])
        label = self.labels[item]
        encoding = tokenizer.encode_plus(
                    review,
                    max_length=self.max_length, #limite maximo de palabras y simbolos
                    truncation=True, #en caso de superar el limite, esta funcion permite truncarlo
                    add_special_tokens=True, #los tokens CLS y ESP 
                    return_token_type_ids=False,
                    padding='max_length', #en caso de un maximo de tokens, que este lo llene con PADs vacios
                    return_attention_mask=True, #que preste atencion unicamente al texto
                    return_tensors='pt'  #tensor final
                )
        return {
            'review': review,
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'label': torch.tensor(label, dtype=torch.long)
        }
    
#data loader
def data_loader(df, tokenizer, max_len, batch_size):
    dataset = IMDBDataset(
        reviews=df.review.to_numpy(),
        labels=df.label.to_numpy(),
        tokenizer=tokenizer,
        max_length=MAX_LEN
    )
    return DataLoader(dataset, batch_size= BATCH_SIZE, num_workers= 0)

df_train, df_test = train_test_split(Dataframe, test_size=0.2, random_state= RANDOM_SEED)

train_data_loader = data_loader(df_train, tokenizer, MAX_LEN, BATCH_SIZE)

test_data_loader = data_loader(df_test, tokenizer, MAX_LEN, BATCH_SIZE)

#   MODELO
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
    
model = BERTSentimentClassifier(N_CLASSES)
model = model.to(device)

#print(model)

# RE-ENTRENAMIENTO
EPOCHS = 5
optimizer = AdamW(model.parameters(), lr=2e-5, correct_bias=False) #lr es el learning rate
total_steps = len(train_data_loader) * EPOCHS
scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps = 0,
        num_training_steps = total_steps 
)
loss_fn = nn.CrossEntropyLoss().to(device)

def train_model(model, data_loader, loss_fn, optimizer, device, scheduler, n_examples):
    model = model.train()
    losses = []
    correct_predictions = 0
    for batch in data_loader:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['label'].to(device)
        outputs = model(input_ids = input_ids, attention_mask = attention_mask)
        _, preds = torch.max(outputs, dim=1)
        loss = loss_fn(outputs, labels)
        correct_predictions += torch.sum(preds == labels)
        losses.append(loss.item())
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()

    return correct_predictions.double()/n_examples, np.mean(losses)

def eval_model(model, data_loader, loss_fn, device, n_examples):
    model = model.eval()
    losses = []
    correct_predictions = 0
    with torch.no_grad():
        for batch in data_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)
            outputs = model(input_ids = input_ids, attention_mask = attention_mask)
            _, preds = torch.max(outputs, dim=1)
            loss = loss_fn(outputs, labels)
            correct_predictions += torch.sum(preds == labels)
            losses.append(loss.item())

        return correct_predictions.double()/n_examples, np.mean(losses)

# Entrenamiento 
for epoch in range(EPOCHS):
    print('Epoch {} de {}'.format(epoch+1, EPOCHS))
    print('_____________________')
    train_acc, train_loss = train_model(
            model, train_data_loader, loss_fn, optimizer, device, scheduler, len(df_train)
    )
    test_acc, test_loss = eval_model(
            model, test_data_loader, loss_fn, device, len(df_test)
    )
    print('Entrenamiento: Loss: {}, accuracy: {}'.format(train_loss, train_acc))
    print('Validacion: Loss: {}, accuracy: {}'.format(test_loss, test_acc))
    print('')

MODEL_PATH = "bert_sentiment_model.pth"
TOKENIZER_PATH = "bert_tokenizer"

torch.save(model.state_dict(),MODEL_PATH)
tokenizer.save_pretrained(TOKENIZER_PATH)
print("Modelo y tokenizer guardados exitosamente.")