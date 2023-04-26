#---------- Imported Libraries ----------

import numpy as np
from sklearn.svm import SVC, OneClassSVM
from sklearn.model_selection import KFold, train_test_split
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
        self.isBinary = conf["model"] == "two-class"
        self.kernel_type = conf["kernel_type"]
        self._class = conf["class"]
        self.model = None

        
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
        invert = False

        if not self.isBinary:
            # We must alter the labels based on the target class
            if self._class == "different":
                # If different we must flip the labels
                labels = np.array([int(i==0) for i in labels], dtype=np.float32)
                invert = True



        for train_index, test_index in k_fold.split(author_curves):
            # Split data
            X_train, X_test = author_curves[train_index], author_curves[test_index]
            y_train, y_test = labels[train_index], labels[test_index]

            # Set the model
            if self.isBinary:
                self.model = SVC(C=self.C, kernel=self.kernel_type)
            else:
                self.model = OneClassSVM(nu=self.C, kernel=self.kernel_type, gamma="auto")

            # Train and test the model on this fold
            self.model.fit(X_train, y_train)
            y_pred = self.model.predict(X_test)
            y_pred = [int(i==1) for i in y_pred]


            if invert:
                y_test = self.invert_list(y_test)
                y_pred = self.invert_list(y_pred)

            accuracy = accuracy_score(y_test, y_pred)
            accuracy_values.append(accuracy)
            fold_results.append((y_test, y_pred))


        tp, fp, tn, fn = self.calculate_metrics(fold_results)
        print("Aggregate True Positives:", tp)
        print("Aggregate False Positives:", fp)
        print("Aggregate True Negatives:", tn)
        print("Aggregate False Negatives:", fn)

        return np.mean(accuracy_values)


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
    

    def invert_list(self, input_list : np.array) -> np.array:
        """
        Inverts all entries in a list of 1's and 0's.
    
        Given a list of 1's and 0's, this function creates a new list with the inverted values.
        In the new list, 1's are replaced with 0's and 0's are replaced with 1's.

        :param input_list: A np.arrat of 1's and 0's to be inverted.
        :return: A new list with inverted values.
        :rtype: np.array
        """
        return np.array([1 - i for i in input_list])