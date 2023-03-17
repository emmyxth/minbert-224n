import csv
import numpy as np

import matplotlib.pyplot as plt
import numpy as np
from collections import Counter
import sys

def preprocess_string(s):
    return ' '.join(s.lower()
                    .replace('.', ' .')
                    .replace('?', ' ?')
                    .replace(',', ' ,')
                    .replace('\'', ' \'')
                    .split())

def load_sentiment_data(predicted, gt):
    predictions = []
    labels = []
    with open(predicted, 'r') as fp:
        for record in csv.DictReader(fp,delimiter = ','):
            label = int(record['Predicted_Sentiment'].strip())
            predictions.append((label))
    with open(gt, 'r') as fp:
        for record in csv.DictReader(fp,delimiter = '\t'):
            label = int(record['sentiment'].strip())
            labels.append((label))
    
    labels = np.array(labels)
    predictions = np.array(predictions)
    arr = zip(labels,predictions)
    counts = Counter(arr)
    arr = np.zeros((5,5))
    for i in range(5):
        for j in range(5):
            arr[i][j] = counts[(i,j)]
    fig, axis = plt.subplots()
    heatmap = axis.imshow(arr, cmap='hot', interpolation='nearest')
    axis.set_title("Heatmap of sentiment predictions")
    axis.set_xlabel('Predictions')
    axis.set_ylabel('True Labels')
    plt.gca().invert_yaxis()
    plt.colorbar(heatmap)
    plt.show()

def load_paraphrase_data(predicted, gt):
    predictions = []
    labels = []
    with open(predicted, 'r') as fp:
        for record in csv.DictReader(fp,delimiter = ','):
            predictions.append(int(float(record['Predicted_Is_Paraphrase'])))
    with open(gt, 'r') as fp:
        for record in csv.DictReader(fp,delimiter = '\t'):
            try:
                labels.append(int(float(record['is_duplicate'])))
            except:
                pass
    labels = np.array(labels)
    predictions = np.array(predictions)
    arr = zip(labels,predictions)
    counts = Counter(arr)
    arr = np.zeros((2,2))
    for i in range(2):
        for j in range(2):
            arr[i][j] = counts[(i,j)]
    fig, axis = plt.subplots()
    heatmap = axis.imshow(arr, cmap='hot', interpolation='nearest')
    axis.set_title("Heatmap of paraphrase predictions")
    axis.set_xlabel('Predictions')
    axis.set_ylabel('True Labels')
    plt.locator_params(axis="both", integer=True, tight=True)
    plt.gca().invert_yaxis()
    plt.colorbar(heatmap)
    plt.show()

if __name__ == "__main__":
    load_sentiment_data("predictions/sst-dev-output.csv", "data/ids-sst-dev.csv") 
    load_paraphrase_data("predictions/para-dev-output.csv", "data/quora-dev.csv")

# def load_similarity_data(similarity_filename):
#     similarity_data = []
#     with open(similarity_filename, 'r') as fp:
#         for record in csv.DictReader(fp,delimiter = '\t'):
#             similarity_data.append((float(record['similarity'])))
#     arr = np.array(similarity_data)
#     plt.imshow(arr, cmap='hot', interpolation='nearest')
#     plt.show()