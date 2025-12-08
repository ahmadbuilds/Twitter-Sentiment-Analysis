import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_recall_curve
from transformers import TFBertForSequenceClassification,TFDistilBertForSequenceClassification
from transformers import AutoTokenizer
import pandas as pd

#function to plot the accuracy
def plot_accuracy(train_acc,val_acc):
    plt.plot(train_acc, label='Training Accuracy')
    plt.plot(val_acc, label='Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.show()

#function to plot the loss
def plot_loss(train_loss,val_loss):
    plt.plot(train_loss, label='Training Loss')
    plt.plot(val_loss, label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()


#function to plot the confusion matrix
def plot_confusion_matrix(y_true, y_pred,classes=[0,1]):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.show()


#function to plot the Precision-Recall curve
def plot_precision_recall(y_true, y_scores):
    precision, recall, _ = precision_recall_curve(y_true, y_scores)
    plt.plot(recall, precision, marker='.')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.show()

#function to plot the F1-score curve
def plot_f1_score(y_true, y_scores):
    precision, recall, _ = precision_recall_curve(y_true, y_scores)
    f1_scores = 2 * (precision * recall) / (precision + recall)
    plt.plot(recall, f1_scores, marker='.')
    plt.xlabel('Recall')
    plt.ylabel('F1 Score')
    plt.title('F1 Score Curve')
    plt.show()

#function to plot the dataset
def plot_dataset_distribution(texts, targets):
    plt.figure(figsize=(8, 6))
    sns.countplot(x=targets)
    plt.xlabel('Classes')
    plt.ylabel('Number of Samples')
    plt.title('Dataset Distribution')
    plt.show()
#function to load the BERT model
def load_bert_model(num_labels):
    model = TFBertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=num_labels,from_pt=True)
    return model

#function to load the DistilBERT model
def load_distilbert_model(num_labels):
    model = TFDistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased', num_labels=num_labels,from_pt=True)
    return model

#function to load the BERT Checkpoint
def load_bert_checkpoint(checkpoint_path, num_labels):
    model = TFBertForSequenceClassification.from_pretrained(checkpoint_path, num_labels=num_labels)
    return model

#function to load the DistilBERT Checkpoint
def load_distilbert_checkpoint(checkpoint_path, num_labels):
    model = TFDistilBertForSequenceClassification.from_pretrained(
        "distilbert-base-uncased",
        num_labels=num_labels,
        from_pt=True
    )
    model.load_weights(checkpoint_path)
    return model

#function to load the tokenizer
def load_tokenizer(model_name):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    return tokenizer

#function to return the dataframe
def load_dataframe(file_index):
    with open(f'/content/drive/MyDrive/Colab Notebooks/backend/Data/processed/Chunks/chunk_{file_index}.csv', 'r', encoding='utf-8') as f:
        df = pd.read_csv(f)
        df['target'] = df['target'].map({0: 0, 4: 1})
        text = df['text'].tolist()
        target = df['target'].tolist()
    return text, target