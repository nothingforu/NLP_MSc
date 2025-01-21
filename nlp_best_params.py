import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from transformers import DistilBertForSequenceClassification, Trainer, TrainingArguments, DistilBertTokenizer
from transformers import BertForSequenceClassification, BertTokenizer
from datasets import load_dataset
from sklearn.metrics import f1_score
from sklearn.metrics import f1_score, confusion_matrix, roc_curve, auc
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import re
from sklearn.model_selection import ParameterGrid

# Ładowanie danych
dataset = load_dataset('laugustyniak/abusive-clauses-pl')

# Model i tokenizer
model_name = 'huawei-noah/TinyBERT_General_4L_312D'
model = BertForSequenceClassification.from_pretrained(model_name, num_labels=2)
tokenizer = BertTokenizer.from_pretrained(model_name)

# Funkcja do czyszczenia tekstu
"""def clean_text(text):
    text = text.lower()
    text = re.sub(r"[^\w\s]", "", text)  # Usunięcie interpunkcji
    text = re.sub(r"\d+", "", text)      # Usunięcie cyfr
    text = re.sub(r"\s+", " ", text)     # Usunięcie wielokrotnych spacji
    return text.strip()"""

# Funkcja do tokenizacji z czyszczeniem
def tokenize_function(examples):
   # examples['text'] = [clean_text(text) for text in examples['text']]
    return tokenizer(examples['text'], padding="max_length", truncation=True, max_length=128)


# Tokenizacja danych
tokenized_datasets = dataset.map(tokenize_function, batched=True)

# Ustawienie urządzenia
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)
print(f"Using device: {device}")

# Funkcja do obliczania metryk
def compute_metrics(p):
    preds, labels = p
    preds = preds.argmax(axis=1)
    return {'f1': f1_score(labels, preds, average='weighted')}

# Hiperparametry do przeszukania
param_grid = {
    'learning_rate': [1e-4],
    'per_device_train_batch_size': [32],
    'gradient_accumulation_steps': [4]
}

# Generowanie kombinacji hiperparametrów
grid = list(ParameterGrid(param_grid))

# Funkcja do treningu z danymi hiperparametrami
def train_with_params(params):
    print(f"\nTrening z parametrami: {params}")

    training_args = TrainingArguments(
        output_dir=f"./results_lr{params['learning_rate']}_bs{params['per_device_train_batch_size']}_ga{params['gradient_accumulation_steps']}",
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        per_device_train_batch_size=params['per_device_train_batch_size'],
        per_device_eval_batch_size=24,
        num_train_epochs=1,
        learning_rate=params['learning_rate'],
        weight_decay=0.01,
        logging_dir="./logs",
        logging_steps=50,
        metric_for_best_model="f1",
        greater_is_better=True,
        fp16=True,
        gradient_accumulation_steps=params['gradient_accumulation_steps'],
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets['train'],
        eval_dataset=tokenized_datasets['validation'],
        compute_metrics=compute_metrics
    )

    trainer.train()
    eval_results = trainer.evaluate(tokenized_datasets['validation'])
    f1_score = eval_results['eval_f1']
    print(f"F1 score dla parametrów {params}: {f1_score:.4f}")
    return f1_score, params

# Wyszukiwanie najlepszych hiperparametrów
best_f1 = 0
best_params = None
results = []

for params in grid:
    f1, tested_params = train_with_params(params)
    results.append((f1, tested_params))
    if f1 > best_f1:
        best_f1 = f1
        best_params = tested_params

print(f"\nNajlepszy F1: {best_f1:.4f} uzyskany dla parametrów: {best_params}")

# Zapis wyników do pliku CSV
results_df = pd.DataFrame(results, columns=['F1', 'Params'])
results_df.to_csv("grid_search_results.csv", index=False)

# Testowanie na najlepszych hiperparametrach
print(f"\nTestowanie na najlepszych hiperparametrach: {best_params}")
training_args = TrainingArguments(
    output_dir="./final_results",
    eval_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    per_device_train_batch_size=best_params['per_device_train_batch_size'],
    per_device_eval_batch_size=24,
    num_train_epochs=1,
    learning_rate=best_params['learning_rate'],
    weight_decay=0.01,
    logging_dir="./logs",
    logging_steps=50,
    metric_for_best_model="f1",
    greater_is_better=True,
    fp16=True,
    gradient_accumulation_steps=best_params['gradient_accumulation_steps'],
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets['train'],
    eval_dataset=tokenized_datasets['validation'],
    compute_metrics=compute_metrics
)

trainer.train()

# Ewaluacja na zbiorze testowym
test_results = trainer.evaluate(tokenized_datasets['test'])
print("Test results with best params:", test_results)

# Wyciągnięcie F1 score
test_f1_score = test_results['eval_f1']
# Zapisanie F1 score do pliku CSV
test_f1_df = pd.DataFrame([{'F1_Score': test_f1_score}])
test_f1_df.to_csv("test_f1_score.csv", index=False)
print(f"Test F1 Score zapisany do test_f1_score.csv: {test_f1_score:.4f}")

# Przewidywanie etykiet na zbiorze testowym
predictions = trainer.predict(tokenized_datasets['test'])
predicted_labels = predictions.predictions.argmax(axis=1)
true_labels = predictions.label_ids

# Macierz konfuzji
cm = confusion_matrix(true_labels, predicted_labels)

sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Illegal', 'legal'], yticklabels=['Illegal', 'legal'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()
cm_df = pd.DataFrame(cm, index=['Legal', 'Illegal'], columns=['Legal', 'Illegal'])
# Zapis do pliku CSV
cm_df.to_csv("confusion_matrix_1epoch_modif.csv")
# Podpisanie wartości macierzy konfuzji:
TP = cm[1, 1]  # True Positives: Legal (model predicted 'Legal' and it was indeed 'Legal')
FP = cm[0, 1]  # False Positives: Illegal (model predicted 'Legal' but it was 'Illegal')
TN = cm[0, 0]  # True Negatives: Illegal (model predicted 'Illegal' and it was indeed 'Illegal')
FN = cm[1, 0]  # False Negatives: Legal (model predicted 'Illegal' but it was 'Legal')

# Wydrukowanie wyników:
print(f"True Positives (TP): {TP}")
print(f"False Positives (FP): {FP}")
print(f"True Negatives (TN): {TN}")
print(f"False Negatives (FN): {FN}")
# Wykres ROC i AUC
fpr, tpr, _ = roc_curve(true_labels, predictions.predictions[:, 1])
roc_auc = auc(fpr, tpr)
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic')
plt.legend(loc="lower right")
plt.show()
