import numpy as np
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score


def compute_tp_tn_fn_fp(y_true, y_pred):
    '''
    True positive - actual = 1, predicted = 1
    False positive - actual = 1, predicted = 0
    False negative - actual = 0, predicted = 1
    True negative - actual = 0, predicted = 0
    '''
    tp = sum((y_true == 1) & (y_pred == 1))
    tn = sum((y_true == 0) & (y_pred == 0))
    fn = sum((y_true == 1) & (y_pred == 0))
    fp = sum((y_true == 0) & (y_pred == 1))
    return tp, tn, fp, fn


def acc_score(y_true, y_pred):
    '''
    Accuracy = TP + TN / FP + FN + TP + TN
    '''
    tp,tn,fp,fn = compute_tp_tn_fn_fp(y_true,y_pred)
    return (tp + tn)/float(tp + tn + fn + fp)


def prec_score(y_true, y_pred):
    '''
    Precision = TP  / FP + TP 
    '''
    tp,tn,fp,fn = compute_tp_tn_fn_fp(y_true,y_pred)
    return tp/float(tp + fp)


def rec_score(y_true, y_pred):
    '''
    Recall = TP /FN + TP 
    '''
    tp,tn,fp,fn = compute_tp_tn_fn_fp(y_true,y_pred)
    return tp/float(tp + fn)


def f1(y_true, y_pred):
    '''
    F1 = 2*((precision * recall)/(precision + recall))
    F1 = TP/(TP+ 0.5* (FP + FN))
    '''
    precision = prec_score(y_true, y_pred)
    recall = rec_score(y_true, y_pred)
    f1_score = (2  * (precision * recall)/(precision + recall))
    return f1_score


y_t = np.array([1, 1, 1, 1, 1, 0, 1, 0, 0, 1])
y_p = np.array([1, 1, 1, 0, 0, 1, 0, 1, 0, 0])

print("scratch")
print(compute_tp_tn_fn_fp(y_t, y_p))
print(acc_score(y_t, y_p))
print(prec_score(y_t, y_p))
print(rec_score(y_t, y_p))
print(f1(y_t, y_p))

print("\nsklearn")
print(accuracy_score(y_t, y_p))
print(precision_score(y_t, y_p))
print(recall_score(y_t, y_p))
print(f1_score(y_t, y_p))

