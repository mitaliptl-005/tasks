# Cell 1: Import Libraries

import argparse
import os
import re
import scipy.io as sio
import numpy as np
from sklearn.metrics import cohen_kappa_score
from sklearn.metrics import confusion_matrix, f1_score

# Cell 2: Define Constants and Functions

W = 0
N1 = 1
N2 = 2
N3 = 3
REM = 4
classes = ['W', 'N1', 'N2', 'N3', 'REM']
n_classes = len(classes)

def evaluate_metrics(cm):
    print("Confusion matrix:")
    print(cm)

    cm = cm.astype(np.float32)
    FP = cm.sum(axis=0) - np.diag(cm)
    FN = cm.sum(axis=1) - np.diag(cm)
    TP = np.diag(cm)
    TN = cm.sum() - (FP + FN + TP)
    # Sensitivity, hit rate, recall, or true positive rate
    TPR = TP / (TP + FN)
    # Specificity or true negative rate
    TNR = TN / (TN + FP)
    # Precision or positive predictive value
    PPV = TP / (TP + FP)
    # Negative predictive value
    NPV = TN / (TN + FN)
    # Fall out or false positive rate
    FPR = FP / (FP + TN)
    # False negative rate
    FNR = FN / (TP + FN)
    # False discovery rate
    FDR = FP / (TP + FP)

    # Overall accuracy
    ACC = (TP + TN) / (TP + FP + FN + TN)
    ACC_macro = np.mean(ACC) # Macro-average accuracy

    F1 = (2 * PPV * TPR) / (PPV + TPR)
    F1_macro = np.mean(F1)

    print("Sample: {}".format(int(np.sum(cm))))
    for index_ in range(n_classes):
        print("{}: {}".format(classes[index_], int(TP[index_] + FN[index_])))

    return ACC_macro, ACC, F1_macro, F1, TPR, TNR, PPV

def print_performance(cm, y_true=[], y_pred=[]):
    tp = np.diagonal(cm).astype(float)  # Use float instead of np.float
    tpfp = np.sum(cm, axis=0).astype(float) # Sum of each column
    tpfn = np.sum(cm, axis=1).astype(float) # Sum of each row
    acc = np.sum(tp) / np.sum(cm)
    precision = tp / tpfp
    recall = tp / tpfn
    f1 = (2 * precision * recall) / (precision + recall)

    FP = cm.sum(axis=0).astype(float) - np.diag(cm)
    FN = cm.sum(axis=1).astype(float) - np.diag(cm)
    TP = np.diag(cm).astype(float)
    TN = cm.sum().astype(float) - (FP + FN + TP)
    specificity = TN / (TN + FP) # TNR

    mf1 = np.mean(f1)

    print("Sample: {}".format(np.sum(cm)))
    print("W: {}".format(tpfn[W]))
    print("N1: {}".format(tpfn[N1]))
    print("N2: {}".format(tpfn[N2]))
    print("N3: {}".format(tpfn[N3]))
    print("REM: {}".format(tpfn[REM]))
    print("Confusion matrix:")
    print(cm)
    print("Precision(PPV): {}".format(precision))
    print("Recall(Sensitivity): {}".format(recall))
    print("Specificity: {}".format(specificity))
    print("F1: {}".format(f1))
    if len(y_true) > 0:
        print("Overall accuracy: {}".format(np.mean(y_true == y_pred)))
        print("Cohen's kappa score: {}".format(cohen_kappa_score(y_true, y_pred)))
    else:
        print("Overall accuracy: {}".format(acc))
    print("Macro-F1 accuracy: {}".format(mf1))

    return acc, precision, recall, specificity, f1, mf1

def perf_overall(data_dir):
    # Remove non-output files, and perform ascending sort
    allfiles = os.listdir(data_dir)
    outputfiles = [os.path.join(data_dir, f) for f in allfiles if re.match(r"^output_.+\d+\.npz", f)]
    outputfiles.sort()

    y_true = []
    y_pred = []
    all_acc = []
    all_precision = []
    all_recall = []
    all_specificity = []
    all_f1 = []
    all_mf1 = []

    for fpath in outputfiles:
        with np.load(fpath) as f:
            print(f["y_true"].shape)
            if len(f["y_true"].shape) == 1:
                if len(f["y_true"]) < 10:
                    f_y_true = np.hstack(f["y_true"])
                    f_y_pred = np.hstack(f["y_pred"])
                else:
                    f_y_true = f["y_true"]
                    f_y_pred = f["y_pred"]
            else:
                f_y_true = f["y_true"].flatten()
                f_y_pred = f["y_pred"].flatten()

            y_true.extend(f_y_true)
            y_pred.extend(f_y_pred)

            print("File: {}".format(fpath))
            cm = confusion_matrix(f_y_true, f_y_pred, labels=[0, 1, 2, 3, 4])
            acc, precision, recall, specificity, f1, mf1 = print_performance(cm)
            all_acc.append(acc)
            all_precision.append(precision)
            all_recall.append(recall)
            all_specificity.append(specificity)
            all_f1.append(f1)
            all_mf1.append(mf1)
    print(" ")

    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    sio.savemat('con_matrix_sleep.mat', {'y_true': y_true, 'y_pred': y_pred})
    cm = confusion_matrix(y_true, y_pred, labels=range(n_classes))
    acc = np.mean(y_true == y_pred)
    mf1 = f1_score(y_true, y_pred, average="macro")

    total = np.sum(cm, axis=1)

    print("Ours:")
    acc, precision, recall, specificity, f1, mf1 = print_performance(cm, y_true, y_pred)

    return {
        "max_acc": np.max(all_acc),
        "max_precision": np.max([np.max(p) for p in all_precision]),
        "max_recall": np.max([np.max(r) for r in all_recall]),
        "max_specificity": np.max([np.max(s) for s in all_specificity]),
        "max_f1": np.max([np.max(f) for f in all_f1]),
        "max_mf1": np.max(all_mf1)
    }

# Cell 3: Define Visualization Functions (if needed)

# Cell 4: Execute the Analysis

# For this notebook, we won't use argparse. Set the directory manually.
data_dir = "./outputs_2013/outputs_eeg_fpz_cz"

# Run the performance evaluation and get max values
max_values = perf_overall(data_dir)
print("Maximum Values:")
print(max_values)
