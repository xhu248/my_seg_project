from sklearn import metrics
from sklearn.metrics import (accuracy_score, confusion_matrix, precision_score,
                             recall_score, f1_score, classification_report)


def print_metrices_out(y_predicted, y_test, y_prob):
    print("Accuracy is %f (in percentage)" %
          (accuracy_score(y_test, y_predicted) * 100))
    matrix = confusion_matrix(y_test, y_predicted)
    print("Confusion Matrix: \n" + str(confusion_matrix(y_test, y_predicted)))
    print("Recall score is %f." % recall_score(y_test, y_predicted))
    print("Precision score is %f." %
          precision_score(y_test, y_predicted))
    print("F1 score is %f." % f1_score(y_test, y_predicted))
    test_auc = metrics.roc_auc_score(y_test, y_prob)
    print("AUC score is %f." % test_auc)
    print("classification Report: \n" +
          str(classification_report(y_test, y_predicted)))
    print("-----------------------------------\n")
    return matrix

