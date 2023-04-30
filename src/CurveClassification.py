#---------- Imported Libraries ----------

import numpy as np
from sklearn.svm import SVC, OneClassSVM
from sklearn.model_selection import KFold, train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.decomposition import PCA
from statsmodels.stats.outliers_influence import variance_inflation_factor



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
        self.confidence = conf["confidence"]
        if self.confidence is not None:
            self.confidence /= 2
        self.model = None

        self.grid_search = conf["grid_search"]
        self.C = conf["C"]
        self.isBinary = conf["model"] == "two-class"
        self.kernel_type = conf["kernel_type"]
        self.gamma = "auto"
        self._class = conf["class"]
        self.max_dims = conf["max_dims"]
        self.confidence = conf["confidence"]
        self.imbalance_ratio = conf["imbalance_ratio"]


        
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
        #vif = self.calculate_vif(author_curves)
        #print(vif)
        k_fold = KFold(n_splits=5, shuffle=True, random_state=42)
        accuracy_values = []
        fold_results = []
        invert = False

        if self.imbalance_ratio is not None:
            labels, author_curves = self.create_imbalance(labels, author_curves)

        if self.max_dims is not None:
            author_curves = self.reduce_features_with_pca(author_curves)
        
        if self.grid_search:
            params = self.grid_search_svm(author_curves, labels)[0]
            print(params)
            self.C = params["C"]
            self.kernel_type = params["kernel"]
            self.gamma = params["gamma"]


        for train_index, test_index in k_fold.split(author_curves):
            # Split data
            X_train, X_test = author_curves[train_index], author_curves[test_index]
            y_train, y_test = labels[train_index], labels[test_index]

           
            if self.isBinary:
                y_pred = self.two_class_svm(X_train, y_train, X_test)
            else:
                y_pred = self.one_class_svm(X_train, y_train, X_test)


            accuracy = accuracy_score(y_test, y_pred)
            accuracy_values.append(accuracy)
            fold_results.append((y_test, y_pred))


        tp, fp, tn, fn = self.calculate_metrics(fold_results)
        #print("Aggregate True Positives:", tp)
        #print("Aggregate False Positives:", fp)
        #print("Aggregate True Negatives:", tn)
        #print("Aggregate False Negatives:", fn)

        accuracy = np.mean(accuracy_values)
        print(fr"{self.imbalance_ratio} & {int((accuracy*10000))/10000} & {tp} & {fp} & {tn} & {fn}\\")
        return accuracy


    def two_class_svm(self, X_train, y_train, X_test):
        # Specify and train the model
        if self.confidence is None:
            clf = SVC(kernel=self.kernel_type, C=self.C, gamma=self.gamma)
        else:
            clf = SVC(kernel=self.kernel_type, C=self.C, gamma=self.gamma, probability=True)
        clf.fit(X_train, y_train)

        # Make predictions on the test data
        if self.confidence is None:
            y_pred = clf.predict(X_test)
        else:
            y_prob = clf.predict_proba(X_test)
            y_pred =  np.where(y_prob[:, 0] <= 0.5 + self.confidence, 1, 0)
        
        return y_pred

        
    def one_class_svm(self, X_train, y_train, X_test):
        # Extract the indices of the normal class (labeled as 1) from the training data
        normal_class_indices = np.where(y_train == 1)[0]

        # Extract the features of the normal class from the training data
        normal_class_features = X_train[normal_class_indices]

        # Train the one-class SVM on the normal class features
        clf = OneClassSVM(kernel=self.kernel_type, nu=self.C, gamma=self.gamma)
        clf.fit(normal_class_features)

        # Make predictions on the test data
        predictions = clf.predict(X_test)

        # Convert -1 predictions to 0
        predictions[predictions == -1] = 0

        return predictions

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
    

    def svm_threshold_classifier(self, X_train, y_train, X_test):
        """
        Trains an SVM classifier and makes predictions using a threshold.
        
        Parameters:
            X_train (numpy array): The feature matrix for training data.
            y_train (numpy array): The label vector for training data.
            X_test (numpy array): The feature matrix for testing data.
            threshold (float): The decision threshold for classifying predictions as 0.
            
        Returns:
            numpy array: The predicted labels for the testing data.
        """
        # Train the SVM classifier
        svm = SVC(probability=True, kernel=self.kernel_type, C=self.C, gamma=self.gamma)
        svm.fit(X_train, y_train)
        
        # Get the predicted probabilities for the test data
        probabilities = svm.predict_proba(X_test)
        
        # Apply the threshold to the probabilities and classify the predictions
        predictions = np.where(probabilities[:, 0] <= 0.5 + self.confidence, 1, 0)
        
        return predictions

    
    def grid_search_svm(self, author_curves, labels, test_size=0.2, random_state=None):
        # Split the data into training and test sets
        X_train, X_test, y_train, y_test = train_test_split(author_curves, labels, test_size=test_size, random_state=random_state)

        # Define the hyperparameter search space
        param_grid = {
            'C': np.logspace(-3, 3, 7),
            'kernel': ['linear', 'poly', 'rbf', 'sigmoid'],
            'degree': [2, 3, 4],
            'gamma': ['scale', 'auto'] + list(np.logspace(-3, 2, 6)),
            'coef0': np.linspace(-1, 1, 21),
        }

        # Create a binary SVM classifier
        svm = SVC()

        # Create the grid search object
        grid_search = GridSearchCV(estimator=svm, param_grid=param_grid, scoring='accuracy', n_jobs=-1, cv=5, verbose=1)

        # Fit the grid search object to the training data
        grid_search.fit(X_train, y_train)

        # Get the best parameters and the corresponding classifier
        best_params = grid_search.best_params_
        best_svm = grid_search.best_estimator_

        # Make predictions on the test set using the best classifier
        y_pred = best_svm.predict(X_test)

        # Calculate the accuracy of the classifier
        accuracy = accuracy_score(y_test, y_pred)

        return best_params, accuracy
    

    def create_imbalance(self, labels, features):
        pos_indices = np.where(labels == 1)[0]
        neg_indices = np.where(labels == 0)[0]
        
        num_pos = len(pos_indices)
        num_neg = len(neg_indices)
        
        desired_neg = int(num_pos / self.imbalance_ratio)
        
        if desired_neg >= num_neg:
            #print("Desired imbalance_ratio is too high, cannot be achieved.")
            return labels, features
        
        np.random.seed(42)
        indices_to_remove = np.random.choice(neg_indices, size=(num_neg - desired_neg), replace=False)
        
        new_labels = np.delete(labels, indices_to_remove)
        new_features = np.delete(features, indices_to_remove, axis=0)
        
        return new_labels, new_features
        

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
    

    def reduce_features_with_pca(self, features):
        pca = PCA(n_components=self.max_dims)
        reduced_features = pca.fit_transform(features)
        return reduced_features
    

    def calculate_vif(self, features):
        vif = [variance_inflation_factor(features, i) for i in range(features.shape[1])]
        return vif
