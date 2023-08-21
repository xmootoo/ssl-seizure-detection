import os
from numpy import squeeze, mean, array, argsort
from tensorflow.keras import backend as K

import matplotlib.pyplot as plt


# plot and save figures for training stats
def training_curves(train_loss, val_loss, train_acc, val_acc, log_dir):
    # set font and figures params
    plt.rcParams["font.size"] = 18
    plt.rcParams["font.family"] = "serif"
    plt.rcParams["font.serif"] = "Arial Narrow"
    plt.rcParams["figure.dpi"] = 300
    plt.rcParams["figure.figsize"] = (7, 5)

    plt.figure()
    plt.plot(train_loss, 'b')
    plt.plot(val_loss, 'g')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend(["Training Loss", "Validation Loss"])
    plt.savefig(os.path.join(log_dir, "training curve loss.jpg"))

    plt.figure()
    plt.plot(train_acc, 'b')
    plt.plot(val_acc, 'g')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend(["Training Accuracy", "Validation Accuracy"])
    plt.savefig(os.path.join(log_dir, "training curve acc.jpg"))


def recall_m(y_true, y_pred):
    TP = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    Positives = K.sum(K.round(K.clip(y_true, 0, 1)))

    recall = TP / (Positives+K.epsilon())
    return recall


def precision_m(y_true, y_pred):
    TP = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    Pred_Positives = K.sum(K.round(K.clip(y_pred, 0, 1)))

    precision = TP / (Pred_Positives+K.epsilon())
    return precision


def f1_score(y_true, y_pred):
    precision, recall = precision_m(y_true, y_pred), recall_m(y_true, y_pred)
    return 2*((precision*recall)/(precision+recall+K.epsilon()))


# Evaluate model on test set
def eval(model, test_loader):
    metrics = None
    attns = []
    test_loss = []
    test_acc = []
    test_auc = []
    test_f1_score = []
    for inputs, labels in test_loader:
        inputs = [inp.numpy() for inp in inputs]
        labels = labels.numpy()
        # get evaluation stats
        outs = model.evaluate(inputs, labels, workers=3)
        for i, k in enumerate(model.metrics_names):
            if metrics is None:
                metrics = {k: 0 for k in model.metrics_names}
            metrics[k] += outs[i] / len(test_loader)
        test_loss.append(metrics["loss"])
        test_acc.append(metrics["accuracy"])
        test_auc.append(metrics["auc"])
        test_f1_score.append(metrics["f1_score"])

    test_stats = {
        'test_loss': test_loss[-1],
        'test_acc': test_acc[-1],
        'test_auc': test_auc[-1],
        'test_f1_score': test_f1_score[-1],
    }

    # stats on test set
    desc = "Evaluation"
    for k in model.metrics_names:
        desc_metrics = " - {}: " + ("{:.4f}" if k == "loss" else "{:.2f}")
        desc += desc_metrics.format(k, metrics[k])
    print(desc)

    return test_stats

