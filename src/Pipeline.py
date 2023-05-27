#---------- Imported libraries ----------

import pandas as pd
import numpy as np
import random
from random import shuffle
from copy import deepcopy
from multiprocessing import Process, Queue
from queue import Empty
import json
from nltk.tokenize import word_tokenize
from itertools import chain
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score
from Chunking import Chunking
from FeatureExtractor import FeatureExtractor
from Unmasking import Unmasking
from Utils import get_time, print_progressbar
from CurveClassification import CurveClassification



#---------- Pipeline ----------

class Pipeline:
     
     def __init__(self, settings_file : str) -> None:
        # Load settings from config file
        with open(settings_file, "r") as file:
            self.settings = json.load(file)
        
        if self.settings["build_author_curves"] == True:
            # Load dataset from disk
            print(f"{get_time()} - Getting data from disk...")
            self.word_cap = self.settings["word_cap"]
            self.file_partitions = self.settings["file_partitions"]
            dataset = self.get_data(self.settings["dataset"])

            # Configure chunking scheme
            self.chunker = Chunking(conf=self.settings["chunk_config"])

                        # Configure feature extraction
            self.feature_extractor = FeatureExtractor(conf=self.settings["feature_config"])

            try:
                tags = dataset["article_partition_tags"]
            except KeyError:
                tags = None

            authors = dataset["authors"].to_list()
            files = dataset["articles"].to_list()

            print(f"{get_time()} - Constructing text pairs...")
           
            file_lengths = np.array([len(word_tokenize(file)) for file in files])

            # Chunk all files now to avoid duplicate work
            files = [self.chunker.chunk_files(file) for file in files]

            # Get X and A pairs
            grouped_by_author = self.group_files_and_authors(authors, files, tags)
            
            print(f"\n{'#'*8} Dataset Metadata {'#'*8}")
            print(f"Number of authors: {len(grouped_by_author.keys())}\nNumber of files: {len(list(chain.from_iterable(grouped_by_author.values())))}")
            print(f"Average file length: {int(np.mean(file_lengths))}\nstd: {round(np.std(file_lengths), 2)}")
            print("#"*34, "\n")
        
            print(f"{get_time()} - Constructing X,A pairs...")


            if self.settings["balanced_classes"] == True:
                X_A = self.create_X_A_pairs_balanced(grouped_by_author)
            else:
                X_A = self.create_X_A_pairs(grouped_by_author)

            print(f"The proportion of same X, A pairs is {round(100*len([i for i in X_A if i[0]])/len(X_A), 2)} %")

            # Specify the Unmasking model
            clf = Unmasking(
                        chunker=self.chunker,
                        feature_extractor=self.feature_extractor,
                        features_eliminated=self.settings["features_eliminated"],
                        C_parameter_curve_construction=self.settings["C_parameter_curve_construction"],
                        model=self.settings["chunk_classifier_type"]
                    )
            
            self.author_curves = [None]*len(X_A)

            # Partition the X_A pairs into 10 blocks of equal length
            blocks = [X_A[i::10] for i in range(10)]

            # Use the blocks to run the author curve generation in parallell
            print(f"{get_time()} - Building author curves...")
            print_progressbar(current_position=-1, length=len(X_A))
            processes = []
            queue = Queue()

            for i in range(10):
                proc = Process(target=self.handle_block, args=(queue, blocks[i], clf))
                processes.append(proc)
            
            for proc in processes:
                proc.start()
            
            counter = 0
            self.author_curves = [None]*len(X_A)
            self.labels = np.zeros(len(X_A))
        
            while counter < len(X_A):
                try:
                    label_and_curve = queue.get(timeout=3)
                    if label_and_curve is not None:
                        self.labels[counter] = int(label_and_curve[0])
                        self.author_curves[counter] = label_and_curve[1]
                        print_progressbar(current_position=counter, length=len(X_A))
                        counter += 1
                except Empty:
                    # The queue is empty and has not been provided with new results
                    # This is ignored
                    pass
            
            self.author_curves = self.normalize_author_curves(self.author_curves)

            print(f"{get_time()} - Author curves have been sucesfully constructed")

        if self.settings["save_author_curves"] and self.settings['build_author_curves']:
            print(f"{get_time()} - Saved the created author curves to {self.settings['save_author_curves']}")
            
            self.save_author_curves(self.settings["save_author_curves"], self.labels, np.array(self.author_curves))
        
        if self.settings["load_author_curves"] is not None:
            if self.settings["load_author_curves"] != self.settings['save_author_curves'] and self.settings['build_author_curves']:
                print("WARNING!\nYou are currently not verifying the performance of the author curves just constructed.")
                print(f"You are saving the curves to {self.settings['save_author_curves']} but loading curves from another file {self.settings['load_author_curves']} for curve classification")
                exit(1)
        

        # Load the curves
        print(f"{get_time()} - Author curves loaded from {self.settings['load_author_curves']}")
        self.labels, self.author_curves = self.load_author_curves(self.settings["load_author_curves"])
        
        # Train the author curve verification model
        curve_classifier = CurveClassification(conf={
                                                        "C" : self.settings["C_parameter_curve_classification"],
                                                            "kernel_type" : self.settings["kernel_type_curve_classification"],
                                                            "model" : self.settings["model"],
                                                            "class" : self.settings["class"],
                                                            "max_dims" : self.settings["max_dims"],
                                                            "confidence" : self.settings["confidence"],
                                                            "imbalance_ratio" : self.settings["imbalance_ratio"],
                                                            "grid_search" : self.settings["grid_search"]
                                                        }
                                                    )
            
        accuracy = curve_classifier.classify_curves(author_curves=self.author_curves,
                                                                labels=self.labels)
                
        #print(f"Accuracy for the author curve classifier: {round(accuracy*100, 2)} %")
        


     def get_data(self, filename : str) -> pd.DataFrame:
         try:
            # Get the different columns of the dataset
            authors, articles = self.read_and_filter_data(filename)

            # If there is no word cap the dataset is just returned as is
            if self.word_cap is None:
                data = pd.DataFrame({"authors" : authors, "articles" : articles})
                return data
        

            # If no file partition is defined each file is simply capped at the specified word count
            if not self.file_partitions:
                articles = [" ".join(word_tokenize(article)[:self.word_cap]) for article in articles]
                data = pd.DataFrame({"authors" : authors, "articles" : articles})
                return data
            
            # Define new columns for the partitioned dataset
            partitoned_articles = []
            partitoned_authors = []
            article_partition_tags = []
            article_tag = 0
            for i, article in enumerate(articles):
                author = authors[i]
                article_parts = []
                words = word_tokenize(article)
                for j in range(0, len(words), self.word_cap):

                    # Retrieve a partition of the article
                    partition = words[j:j+self.word_cap]
                    if len(partition) != self.word_cap:
                        continue
                    article_parts.append(" ".join(partition))
                    article_partition_tags.append(article_tag)
                # Add all file partitions and the author to the new columns
                partitoned_articles += article_parts
                partitoned_authors += [author]*len(article_parts)
                article_tag += 1
            
            data = pd.DataFrame({"authors" : partitoned_authors, 
                                 "articles" : partitoned_articles,
                                 "article_partition_tags" : article_partition_tags
                                 })
            return data
         
         except FileNotFoundError:
            print(f"The file {filename} could not be found")
            exit(1)
    

     def read_and_filter_data(self, filename):
        data = pd.read_csv(filename, encoding="utf-8", encoding_errors="ignore")

        # Get the different columns of the dataset
        authors = data["authors"].tolist()
        articles = data['articles'].tolist()

        # Remove any corrupted entries
        uncorrupted_files = [i for i, article in enumerate(articles) if type(article) is str]
        authors = [author for i, author in enumerate(authors) if i in uncorrupted_files]
        articles = [article for i, article in enumerate(articles) if i in uncorrupted_files]

        filtered_authors = []
        filtered_articles = []

        for author, article in zip(authors, articles):
            article_length = len(word_tokenize(article))
            min_length = self.settings["minimum_file_length"]
            max_length = self.settings["maximum_file_length"]

            if min_length is not None and article_length < min_length:
                continue

            if max_length is not None and article_length > max_length:
                continue

            filtered_authors.append(author)
            filtered_articles.append(article)
        
        return filtered_authors, filtered_articles


     def group_files_and_authors(self, authors: list, files: list, tags: list = None) -> dict:
        """
        Returns a dictionary where each author is mapped to all files (s)he has written
        """
        grouped_by_author = {author: [] for author in authors}

        for i, author in enumerate(authors):
            if tags is None:
                grouped_by_author[author].append(files[i])
            else:
                grouped_by_author[author].append((files[i], tags[i]))

        if self.settings["minimum_file_count"]:
            authors_to_remove = [
                author for author, files in grouped_by_author.items()
                if len(files) < self.settings["minimum_file_count"]
            ]

            for author in authors_to_remove:
                del grouped_by_author[author]

        if self.settings["maximum_file_count"]:
            for author, files in grouped_by_author.items():
                if len(files) > self.settings["maximum_file_count"]:
                    files_to_keep = random.sample(files, self.settings["maximum_file_count"])
                    grouped_by_author[author] = files_to_keep

        return grouped_by_author

     
     

     def create_X_A_pairs(self, grouped_by_author : dict) -> list:
        """
        Creates all possible book author pairs from the corpus
        """
        X_A = []
        for author in grouped_by_author.keys():
            files = grouped_by_author[author]
            #grouped_by_author[author] = [self.chunker.chunk_files(file) for file in]
            for i, X in enumerate(files):
                # Chunk the file (X) here to avoid duplicate work later
                #chunked_file = self.chunker.chunk_files(file)
                for a in grouped_by_author.keys():
                    if a == author:
                        if len(files) > 1:
                            A = deepcopy(files)
                            A.pop(i)
                            X_A.append([True, X, list(chain.from_iterable(A))])
                    else:
                        X_A.append([False, X, list(chain.from_iterable(grouped_by_author[a]))])
        
        return X_A


     def create_X_A_pairs_balanced(self, grouped_by_author : dict) -> list:
        """
        Does the same as the above but the number of same and different author curves are equal
        """
        same = []
        different = []
        tagged = self.settings["file_partitions"]

        for author in grouped_by_author.keys():
            files = grouped_by_author[author]

            for i, X in enumerate(files):
                if tagged:
                    X = X[0]

                for a in grouped_by_author.keys():
                    if a == author:
                        # X is written by the author
                        if len(files) > 1:
                            A = deepcopy(files)

                            # Remove X from A
                            tag = A.pop(i)[1]
                            if tagged:
                                # Remove all fractions from same origin as X from A
                                #print(f"A length is {len(A)} yet to remove fractions with tag {tag}")
                                A = self.handle_tagged_files(A, tag)
                                #print(f"A length is {len(A)} after removing fractions")
                                if len(A) == 0:
                                    continue

                            #print(f"Same pair. A consist of {len(A)} files and {len(list(chain.from_iterable([word_tokenize(file) for file in A])))}")
                            same.append([True, X, list(chain.from_iterable(A))])
                            
                    else:
                        # X is not written by the author
                        A = [file[0] if tagged else file for file in grouped_by_author[a]]
                        #print(f"Different pair. A consist of {len(A)} files and {len(list(chain.from_iterable([word_tokenize(file) for file in A])))}")
                        different.append([False, X, list(chain.from_iterable(A))])

        # Ensure that there are equally many samples of each class
        if len(different) > len(same):
            shuffle(different)
            different = different[:len(same)]
        else:
            shuffle(same)
            same = same[:len(different)]

        X_A = same + different
        shuffle(X_A)
        return X_A
     

     def handle_tagged_files(self, A_files, file_tag) -> list:
         """
         Handles the creation of X, A pairs when file tags are used. 
         This is occurs when single texts have been broken up into smaller
         chunks and treated as individual files. This method ensures that no fragments
         from the same file as the one used as X ends up in A.
         """
         # Finds all fragments from a different source file than X
         allowed_files = [file[0] for file in A_files if file[1] != file_tag]
         return allowed_files
        

     def handle_block(self, queue : Queue, block : list, clf : Unmasking) -> None:
        """
        Constructs the author curves from the provided block of X A pairs 
        """
        for X_A_pair in block:
            # Get the different elements in the X_A pair
            label = X_A_pair[0]
            X = X_A_pair[1]
            A = X_A_pair[2]
                
            # Get the author curve
            if self.chunker.method == "original1":
                author_curve = clf.handle_X_A_pair(X=X, A=A)
            else:
                author_curves = []
                for _ in range(10):
                    author_curves.append(clf.handle_X_A_pair(X=X, A=A))
                author_curve = np.mean(author_curves, axis=0)

            # Now push the label and finished author curve to the queue
            queue.put([label, author_curve])

    
     def save_author_curves(self, filename : str, labels : np.ndarray, author_curves : np.ndarray) -> None:
        """
        Saves the provided author curves to a .csv file
        """
        # Combine the labels and author curves into one array and save it to a .csv file
        c = np.concatenate((labels.reshape(-1, 1), author_curves), axis=1)
        columns = ["label"] + [f"feature{i+1}" for i in range(author_curves.shape[1])]
        df = pd.DataFrame({columns[i] : c[:,i] for i in range(len(columns))})
        df.to_csv(filename, index=False)


     def load_author_curves(self, filename : str) -> tuple:
         """
         Loads the labels and author curves from the specified .csv file
         """
         df = pd.read_csv(filename)
         labels = df['label'].to_numpy()
         author_curves = df.iloc[:,1:].to_numpy()
         
         return (labels, author_curves)
     

     def normalize_author_curves(self, curves : list) -> np.array:
         """
         Normalizes curve shapes by reverse engineering the author curves with more than the smalles amount of features
         This means that data obtained from some of the last elimination rounds for some curves are droppped
         """
         shapes = [curve.shape[0] for curve in curves]
         if not max(shapes) - min(shapes):
             return curves
         
         min_shape = min(shapes)

         features_specified = self.feature_extractor.number
         print(f"WARNING! - There are text pairs where the total number of feauter are less than the {features_specified} features specifified in the config file.")
         print(f"The smallest number features extracted from a text pair is : {int((min_shape + 3)/3)}.")
         print("An effort will be made to normalize the created author curves. This means that data from some elimination rounds will be dropped.")
         for i, curve in enumerate(curves):
            delta = int((curve.shape[0]+3)/3) - int((min_shape+3)/3)
            if delta:
                acc_len = int((curve.shape[0] + 3)/3)
                n1_len = 2*acc_len - 1
                n2_len = 3*acc_len - 3

                acc_part = curve[0:acc_len-delta]
                n1_part = curve[acc_len:n1_len-delta]
                n2_part = curve[n1_len:n2_len-delta]
                drops = curve[-2:]
                curves[i] = np.concatenate([acc_part, n1_part, n2_part, drops])
    
         return curves

    
     def classify_curves(self, kernel : str, C : float) -> None:
        """
        Classifies the author curves and returns the average performance
        from 5-fold cross-validation
        """
        # Specify, train, and test the model
        clf = SVC(kernel=kernel, C=C)
        scores = cross_val_score(clf, self.author_curves, self.labels, cv=5)
        return scores.mean()
     