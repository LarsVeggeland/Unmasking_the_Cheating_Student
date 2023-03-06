#---------- Imported libraries ----------

from Chunking import Chunking
from FeatureExtractor import FeatureExtractor
import numpy as np
from sklearn.model_selection import cross_val_predict, cross_val_score, StratifiedKFold, KFold
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, accuracy_score
from random import shuffle
import pandas as pd
from matplotlib import pyplot as plt
from collections import Counter
from os import listdir
from DataPreProcessing import PreProcessor


#---------- Unmasking Class ----------

class Unmasking:

    def __init__(
                self,
                chunker : Chunking,
                normalized_features : bool,
                number_of_features : int,
                C_parameter_curve_construction : float,
                C_parameter_curve_classification : float,
                kernel_type_curve_classification : str,
                build_author_curves : bool,
                save_author_curves : str,
                load_author_curves : str
            ) -> None:

        self.chunker = chunker
        self.normalized_features = normalized_features
        self.number_of_features=number_of_features
        self.C_parameter_curve_construction = C_parameter_curve_construction
        self.C_parameter_curve_classification = C_parameter_curve_classification
        self.kernel_type_classification = kernel_type_curve_classification
        self.build_author_curves = build_author_curves
        self.save_author_curves = save_author_curves
        self.load_author_curves = load_author_curves


    def handle_X_A_pair(
                    self,  
                    X : str,
                    A : list,
                    chunks_X : list,
                ) -> np.ndarray:
        
        # Get the chunks from X and A
        chunks_X = chunks_X #self.chunk_samples(X, chunk_size=500)
        chunks_A = self.chunker.chunk_file(A)

        # Ensure that there are equally many chunks from X and A
        if len(chunks_A) > len(chunks_X):
            shuffle(chunks_A)
            chunks_A = chunks_A[:len(chunks_X)]
        elif len(chunks_X) > len(chunks_A):
            shuffle(chunks_X)
            chunks_X = chunks_X[:len(chunks_A)]

        # Get the features from the chunks
        features_X, features_A = self.extract_features(
                    X=X, A=A,
                    chunks_X=chunks_X,
                    chunks_A=chunks_A
                )

        # Perform chunk classification and retrieve results
        results = self.train_chunk_classifiers(
                    features_to_be_eliminated=3,
                    features_X=features_X,
                    features_A=features_A
                )

        # Use chunk classification results to coonstruct author curve
        author_curve = self.construct_author_curve(results=results)

        return author_curve
    

    def chunk_samples(self, data : list, chunk_size : int) -> list:
        """
        Chunks all text samples in accordance with the specified approach
        """
        chunker = Chunking(chunk_size)
        # Make this more configureable, i.e., utilize a similar approach as the preprocessor
        #return chunker.sample_by_words(data)
        return chunker.sample_by_words(data)

    def extract_features(self, X : str, A : list, chunks_X : list, chunks_A : list) -> tuple:
        """
        Retrieves the specified features from the provided data
        """
        extractor = FeatureExtractor()
        return extractor.vectorize_chunks_by_n_most_frequent_weighted_words(
                    n=self.number_of_features,
                    X=X, 
                    A=A, 
                    chunks_X=chunks_X, 
                    chunks_A=chunks_A,
                    normalized=self.normalized_features
                )


    def train_chunk_classifiers(self, features_to_be_eliminated : int, features_X : np.ndarray, features_A : np.ndarray) -> list:
        """
        Trains a classifier for each of the 5 folds. Elinates the most positive and negative features for each elimination round
        """
        # Get the combined chunks from X and A
        chunks = np.concatenate((features_X, features_A), axis=0)

        # Get the labels for each chunk X=1, A=0
        labels_X = np.ones(features_X.shape[0])
        labels_A = np.zeros(features_A.shape[0])
        labels = np.concatenate((labels_X, labels_A), axis=None)

        # The number of elimiation rounds required for exhausting the feature set
        elimination_rounds = int(chunks.shape[1]/(features_to_be_eliminated*2))

        # The results array holding the classification performance for each round
        all_results = np.zeros(shape=(5, elimination_rounds))

        # Create the indexes for training and test data for each fold
        folds = KFold(n_splits=5, random_state=42, shuffle=True)

        for fold_index, (train_index, test_index) in enumerate(folds.split(chunks)):
            # A copy of the chunks used for this fold
            fold_chunks = np.array(chunks)

            for elim_round in range(elimination_rounds):
                if chunks.shape[1] <= features_to_be_eliminated:
                    break

                # Get train and test data
                X_train = fold_chunks[train_index]
                X_test = fold_chunks[test_index]
                y_train = labels[train_index]
                y_test = labels[test_index]

                # Specify, train, and test the model
                clf = SVC(kernel="linear")
                clf.fit(X_train, y_train)
                pred = clf.predict(X_test)
                score = accuracy_score(y_test, pred)
                all_results[fold_index, elim_round] = score

                # Eliminate the most positive and negatve features for the model from fold_chunks
                fold_chunks = self.feature_elimination(n_features=features_to_be_eliminated, 
                                                  chunks=fold_chunks, 
                                                  clf=clf
                                                )
        # Calculate and return the mean accuracy for each elimination round for all folds
        results = np.mean(all_results, axis=0)
        return results

    
    def feature_elimination(self, n_features : int, chunks : np.ndarray, clf) -> np.ndarray:
        """
        Removes the n first/most positively weighted features from each chunk
        """
        # Find the most important features for the provided model
        coefs = clf.coef_
        features = np.argsort(coefs)
        most_negative = features[:,:n_features]
        most_positive = features[:,features.shape[1]-n_features:]

        # Determines the index of the most negative and positive coeficients
        features_to_be_removed = np.concatenate((most_negative, most_positive), axis=None)
        features_to_be_removed = np.sort(features_to_be_removed, axis=None)[::-1]

        # Delete the features/columns from all chunks 
        for feature in features_to_be_removed:
            chunks = np.delete(chunks, feature, 1)
        
        return chunks


    def construct_author_curve(self, results : np.ndarray) -> np.ndarray:
        """
        Constructs an author curve from the provided chunk classification results
        """
        # Get the difference between neighbors
        neighbor_diff = - np.diff(results)

        # Get difference between next neighbours
        l = results.tolist()
        next_diff = []   
        for i, res in enumerate(l):
            if i+2 >= len(l):
                break
            next_diff.append(res - l[i+2])
        
        next_neighbor_diff = np.array(next_diff)

        # Get the biggest drop between chunk classification iterations
        max_neighbor_drop = np.amin(neighbor_diff)
        max_next_neighbor_drop = np.amin(next_neighbor_diff)
        drops = np.array([max_neighbor_drop, max_next_neighbor_drop])

        # Constuct the author curve
        author_curve = np.concatenate((results, neighbor_diff, next_neighbor_diff, drops))

        return author_curve