import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import torch
from torch import nn
from skorch import NeuralNetClassifier
from modAL.models import ActiveLearner
import os
from modAL.uncertainty import margin_sampling, uncertainty_sampling, entropy_sampling
import matplotlib as mpl
import matplotlib.pyplot as plt
import statistics 
import sklearn
from sklearn.preprocessing import LabelEncoder
from tabulate import tabulate 
import math 
from PIL import Image
import csv


def load_images(folder, meta_data):
    images = []  
    labels = []
    for filename in os.listdir(folder):
        if filename.endswith('.png'):
            img_path = os.path.join(folder, filename)
            img = Image.open(img_path) 
            
            row = meta_data.loc[meta_data['fname'] == filename]
            if row.empty:
                print(f"Label not found for image: {filename}")
                continue
            
            h_min, w_min, h_max, w_max = row['h_min'].values[0], row['w_min'].values[0], row['h_max'].values[0], row['w_max'].values[0]
            
    
            if h_min < 0 or w_min < 0 or h_max > img.height or w_max > img.width:
                print(f"Invalid crop coordinates for {img_path}: {(h_min, w_min, h_max, w_max)}")
                continue

    
            img = img.crop((w_min, h_min, w_max, h_max))

            # Resize the image
            img = img.resize((224, 224)) 
            
            # Convert to float32 and normalize
            img = np.array(img).astype(np.float32) / 255.0
            
            # Reorganize dimensions to [channels, height, width]
            img = np.transpose(img, (2, 0, 1))
            
            # Get the corresponding label
            label = row['structure'].values[0]
            images.append(img)
            labels.append(label)

    assert len(images) == len(labels), "Number of images does not match number of labels"

    return np.array(images), np.array(labels)

def initial_data_assemble(X_train, y_train, n_initial):
    initial_idx = np.random.choice(range(len(X_train)), size=n_initial, replace=False)
    X_initial = X_train[initial_idx]
    y_initial = y_train[initial_idx]
    return initial_idx, X_initial, y_initial

def generate_pool(X_train, y_train, initial_idx):
    X_pool = np.delete(X_train, initial_idx, axis=0)
    y_pool = np.delete(y_train, initial_idx, axis=0)
    return X_pool, y_pool

def load_new_images(folder):
    images = []
    for filename in os.listdir(folder):
        if filename.endswith('.png'):
            img_path = os.path.join(folder, filename)
            img = Image.open(img_path) 
            
            img = img.resize((224, 224)) 
            
            img = np.array(img).astype(np.float32) / 255.0
            
            img = np.transpose(img, (2, 0, 1))
            
            images.append(img)
    
    return np.array(images)

""" Import and pre-process of data """

df = pd.read_excel('ObjectDetection.xlsx')
images, labels = load_images("Dataset\\Set1-Training&Validation Sets CNN\\Standard", df)


images2, labels2 = load_images("Dataset\\Set2-Training&Validation Sets ANN Scoring system\\Standard", df)

images_combined = np.concatenate((images, images2), axis=0)
labels_combined = np.concatenate((labels, labels2), axis=0)

# Encode the labels
label_encoder = LabelEncoder()
labels_encoded = label_encoder.fit_transform(labels_combined)

assert len(images_combined) == len(labels_encoded), "Number of images and labels do not match after merging."

X_train, X_test, y_train, y_test = train_test_split(images_combined,labels_encoded, test_size=0.2, random_state=42)

initial_idx, X_initial, y_initial = initial_data_assemble(X_train, y_train, 10)
X_pool, y_pool = generate_pool(X_train, y_train, initial_idx)

""" Classifier """

class CNN_Model(nn.Module):
    def __init__(self, num_layers, dropout_rate):
        super().__init__()
        layers = []
        in_channels = 3  # RGB channels

        # Dynamically create convolutional layers
        for _ in range(num_layers):
            layers.append(nn.Conv2d(in_channels, 32, kernel_size=3, stride=1, padding=1))
            layers.append(nn.ReLU())
            layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
            layers.append(nn.Dropout(0.25)) 
            in_channels = 32  

        self.conv_layers = nn.Sequential(*layers)

        # Fully Connected Layers
        self.fc_layers = nn.Sequential(
            nn.Linear(32 * (224 // (2 ** num_layers))**2, 512), 
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(256, 9)  
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1)
        x = self.fc_layers(x)
        return x


def create_classifier(num_layers, dropout_rate, lr, weight_decay):
    return NeuralNetClassifier(
        CNN_Model(num_layers, dropout_rate),
        criterion=nn.CrossEntropyLoss,
        optimizer=torch.optim.Adam,
        lr=lr, 
        max_epochs=50,  
        batch_size=32,  
        train_split=None,  
        verbose=1, 
        device='cuda' if torch.cuda.is_available() else 'cpu', 
        optimizer__weight_decay=weight_decay # L2 regularization
    )

def save_results_to_csv(filename, results):
    with open(filename, mode='a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(results)

# Cabeçalhos para os arquivos CSV
header1 = ['id', 'Layers', 'Epochs(Learner)', 'Epochs(ALloop)', 'Samplingstrategy', 'Instancestotrain', 'Queries', 'Trainaccuracy', 'Testaccuracy']
"""
with open('results1.csv', mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(header1)
"""

""" Active Learn model """

experiments = [
    {'id': 26, 'layers': 1, 'dropout_rate': 0.6, 'epochs_learner': 40, 'epochs_al': 20, 'sampling_strategy': uncertainty_sampling, 'instances_to_train': 20, 'queries': 10, 'lr': 0.0001, 'weight_decay': 1e-4}
]

for exp in experiments:
    exp_id = exp['id']
    print(f"\nStarting experiment {exp_id} with configuration: {exp}")

    # Create the CNN model with the specified number of layers and dropout
    classifier = create_classifier(exp['layers'], exp['dropout_rate'], exp['lr'], exp['weight_decay'])

    learner = ActiveLearner(
        estimator=classifier,
        X_training=X_initial,
        y_training=y_initial,
        query_strategy=exp['sampling_strategy'],
        epochs=exp['epochs_learner']
    )

    # Train and evaluate the initial model
    learner.fit(X_initial, y_initial)
    accuracy = learner.score(X_train, y_train)
    t_accuracy = learner.score(X_test, y_test)

    accuracy_historial = [accuracy]
    test_accuracy = [t_accuracy]

    for idx in range(exp['queries']):
        print(f'Query iteration no. {idx + 1} for experiment {exp_id}')
        query_idx, query_instance = learner.query(X_pool, n_instances=exp['instances_to_train'])
        learner.teach(
            X=X_pool[query_idx], y=y_pool[query_idx].astype(int), epochs=exp['epochs_al']
        )
        
        #Save accuracy history
        accuracy_historial.append(learner.score(X_train, y_train))
        test_accuracy.append(learner.score(X_test, y_test))

        #Update the data pool
        X_pool = np.delete(X_pool, query_idx, axis=0)
        y_pool = np.delete(y_pool, query_idx, axis=0)

 
    save_results_to_csv('results1.csv', [
        exp_id,
        exp['layers'],
        exp['epochs_learner'],
        exp['epochs_al'],
        exp['sampling_strategy'].__name__,
        exp['instances_to_train'],
        exp['queries'],
        f"{accuracy_historial[-1] * 100:.2f}%",
        f"{test_accuracy[-1] * 100:.2f}%"
    ])

    """ Model evaluation - Metrics """
    prediction = learner.predict_proba(X_test)

    num_classes_y_test = len(np.unique(y_test))
    num_classes_prediction = prediction.shape[1]
    assert num_classes_y_test == num_classes_prediction, "The number of classes in y_test does not match the number of columns in prediction"

    auc = sklearn.metrics.roc_auc_score(y_test, prediction, multi_class='ovr')

    fig, ax = plt.subplots(figsize=(8.5, 6), dpi=130)

    ax.plot(accuracy_historial, label='Train Accuracy')
    ax.plot(test_accuracy, label='Test Accuracy')
    ax.legend()

    ax.xaxis.set_major_locator(mpl.ticker.MaxNLocator(nbins=5, integer=True))
    ax.yaxis.set_major_locator(mpl.ticker.MaxNLocator(nbins=10))
    ax.yaxis.set_major_formatter(mpl.ticker.PercentFormatter(xmax=1))

    ax.set_ylim(bottom=0, top=1)
    ax.grid(True)

    ax.set_title('Incremental classification accuracy')
    ax.set_xlabel('Query iteration')
    ax.set_ylabel('Classification Accuracy')

    plt.show()

    train_acc = statistics.mean(accuracy_historial)
    test_acc = statistics.mean(test_accuracy)

    print("Train accuracy mean: %4.2f " % (train_acc * 100))
    print("Test accuracy mean:  %4.2f" % (test_acc * 100))
    print("Area under the ROC curve: %4.2f" % auc)

    classes = []

    for i in range(len(prediction)):
        classes.append(np.argmax(prediction[i]))

    confusion_matrix = sklearn.metrics.confusion_matrix(y_test, classes)

    FP = confusion_matrix.sum(axis=0) - np.diagonal(confusion_matrix)
    FN = confusion_matrix.sum(axis=1) - np.diagonal(confusion_matrix)
    TP = np.diagonal(confusion_matrix)
    TN = confusion_matrix.sum() - (FP + FN + TP)

    sensitivity = TP / (TP + FN)
    specificity = TN / (TN + FP)
    ppv = TP / (TP + FP)
    npv = TN / (TN + FN)
    acc = (TP + TN) / (TP + FP + FN + TN)

    metrics = []
    header = ['Structure', 'Sensitivity', 'Specificity', 'Positive predictive value', 'Negative predictive value', 'Accuracy']
    diseases = ['Thalami', 'Midbrain', 'Palate', '4th ventricle', 'Cisterna magna', 'Nuchal translucency (NT)', 'Nasal tip', 'Nasal skin', 'Nasal bone']

    for i in range(9):
        metrics.append([diseases[i], sensitivity[i], specificity[i], ppv[i], npv[i], acc[i]])

    for i in range(9):
        for j in range(1, 5):
            if math.isnan(metrics[i][j]):
                metrics[i][j] = 0
        if math.isnan(ppv[i]):
            ppv[i] = 0

    values = sklearn.metrics.precision_recall_fscore_support(y_test, classes, average='micro')
    overall_acc = (sum(TP) + sum(TN)) / (sum(TP) + sum(FP) + sum(FN) + sum(TN))
    overall_npv = sum(TN) / (sum(TN) + sum(FN))
    overall_speci = sum(TN) / (sum(TN) + sum(FP))

    metrics.append(['Micro-averaging measurements:', values[0], overall_speci, values[1], overall_npv, overall_acc])

    print(tabulate(metrics, header, tablefmt="github", numalign="center"))

    save_results_to_csv('metrics.csv', [exp_id, exp['layers'], exp['epochs_learner'], exp['epochs_al'], exp['sampling_strategy'].__name__, exp['instances_to_train'], exp['queries'], f"{train_acc * 100:.2f}%", f"{test_acc * 100:.2f}%", f"{auc:.2f}", f"{overall_acc * 100:.2f}%"])


accuracy_historial

""" Test with new images """

new_images = load_new_images("Dataset\\Internal Test Set\\Standard")
new_predictions = learner.predict(new_images)

# # Decode the predicted labels
new_predictions_decoded = label_encoder.inverse_transform(new_predictions)

print("Predictions for new images:")
for i, prediction in enumerate(new_predictions_decoded):
    print(f"Image {i+1}: {prediction}")