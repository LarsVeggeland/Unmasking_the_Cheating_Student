#---------- Imported Libraries ----------

import numpy as np
from sklearn.svm import SVC, OneClassSVM
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score, confusion_matrix



#---------- Curve_classification_class ----------

class CurveClassification:
    
    def __init__(self, conf):
        """
        Constructor for the Curve_classification class.

        Parameters
        ----------
        conf : dict
            A dictionary containing configuration information for the class.
        """
        self.C = conf["C"]
        kernel_type = conf["kernel_type"]

        if conf["model"] == "two-class":
            self.model = SVC(C=self.C, kernel=kernel_type)
        else:
            self.model = OneClassSVM(nu=self.C, kernel=kernel_type)
            self._class = conf["class"]

        
    def classify_curves(self, author_curves, labels):
        """
        Train and test an SVM model using 5-fold cross-validation.
        
        Parameters
        ----------
        author_curves : numpy array
            The array of author curves to use as input features (X).
        labels : numpy array
            The array of labels to use as output (Y).

        Returns
        -------
        float
            The average accuracy across the 5 folds.
        """
        k_fold = KFold(n_splits=5, shuffle=True, random_state=42)
        accuracy_values = []
        fold_results = []

        for train_index, test_index in k_fold.split(author_curves):
            X_train, X_test = author_curves[train_index], author_curves[test_index]
            y_train, y_test = labels[train_index], labels[test_index]

            self.model.fit(X_train, y_train)
            y_pred = self.model.predict(X_test)

            accuracy = accuracy_score(y_test, y_pred)
            accuracy_values.append(accuracy)
            fold_results.append((y_test, y_pred))

        tp, fp, tn, fn = self.calculate_metrics(fold_results)
        print("Aggregate True Positives:", tp)
        print("Aggregate False Positives:", fp)
        print("Aggregate True Negatives:", tn)
        print("Aggregate False Negatives:", fn)

        return np.mean(accuracy_values)

    # ... (calculate_metrics method here)


    def calculate_metrics(self, fold_results):
        """
        Calculate the aggregate true positives (TP), false positives (FP),
        true negatives (TN), and false negatives (FN) across the folds.

        Parameters
        ----------
        fold_results : list of tuples
            A list containing tuples of (y_test, y_pred) for each fold,
            where y_test is the true labels and y_pred is the predicted labels.

        Returns
        -------
        tuple
            A tuple containing the aggregate values of TP, FP, TN, and FN, respectively.
        """
        aggregate_tp = 0
        aggregate_fp = 0
        aggregate_tn = 0
        aggregate_fn = 0

        for y_test, y_pred in fold_results:
            cm = confusion_matrix(y_test, y_pred)
            tn, fp, fn, tp = cm.ravel()

            aggregate_tp += tp
            aggregate_fp += fp
            aggregate_tn += tn
            aggregate_fn += fn

        return aggregate_tp, aggregate_fp, aggregate_tn, aggregate_fn