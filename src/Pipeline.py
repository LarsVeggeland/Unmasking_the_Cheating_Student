#---------- Imported libraries ----------

import pandas as pd
import numpy as np
from random import shuffle
from copy import deepcopy
from multiprocessing import Process, Queue
from queue import Empty
import json
from nltk.tokenize import word_tokenize
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score
from DataPreProcessing import PreProcessor
from Chunking import Chunking
from FeatureExtractor import FeatureExtractor
from Unmasking import Unmasking
from Utils import get_time, print_progressbar


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
            authors = dataset["authors"].to_list()
            files = dataset["articles"].to_list()

            # Preprocess the loaded data
            print(f"{get_time()} - Pre-processing files...")
            proc = PreProcessor(jobs=self.settings["preprocessing_jobs"])
            files = self.pre_process_data(data=files, proc=proc)

            # Get X and A pairs
            grouped_by_author = self.group_files_and_authors(authors, files)

            # Configure chunking scheme
            self.chunker = Chunking(conf=self.settings["chunk_config"])

            # Configure feature extraction
            self.feature_extractor = FeatureExtractor(conf=self.settings["feature_config"])

            if self.settings["balanced_classes"] == True:
                X_A = self.create_X_A_pairs_balanced(grouped_by_author)
            else:
                X_A = self.create_X_A_pairs(grouped_by_author)

            # Specify the Unmasking model
            clf = Unmasking(
                        chunker=self.chunker,
                        feature_extractor=self.feature_extractor,
                        features_eliminated=self.settings["features_eliminated"],
                        C_parameter_curve_construction=self.settings["C_parameter_curve_construction"]
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
            
            print(f"{get_time()} - Author curves loaded from {self.settings['load_author_curves']}")
            self.labels, self.author_curves = self.load_author_curves(self.settings["load_author_curves"])


            # Train the author curve verification model
            accuracy = self.classify_curves(
                    kernel=self.settings["kernel_type_curve_classification"],
                    C=self.settings["C_parameter_curve_classification"]
                )
            print(f"Accuracy for the author curve classifier: {round(accuracy*100, 2)} %")
        




     def get_data(self, filename : str) -> pd.DataFrame:
        try:
            data = pd.read_csv(filename, encoding="utf-8", encoding_errors="ignore")

            # If there is no word cap the dataset is just returned as is
            if self.word_cap is None:
                return data
            
            # Get the different columns of the dataset
            authors = data["authors"].to_list()
            articles = data['articles'].to_list()

            # If no file partition is defined each file is simply capped at the specified word count
            if not self.file_partitions:
                articles = [" ".join(word_tokenize(article)[:self.word_cap]) for article in articles]
                data = pd.DataFrame({"authors" : authors, "articles" : articles})
                return data
            
            # Define new columns for the partitioned dataset
            partitoned_articles = []
            partitoned_authors = []

            for i, article in enumerate(articles):
                author = authors[i]
                article_parts = []
                for j in range(0, len(article), self.word_cap):
                    if j + self.file_partitions >= len(article):
                        break
                    # Retrieve a partition of the article
                    article_parts.append(article[j:j+self.word_cap])

                # Add all file partitions and the author to the new columns
                partitoned_articles += article_parts
                partitoned_authors += [author]*len(article_parts)
            
            data = pd.DataFrame({"authors" : partitoned_authors, "articles" : partitoned_articles})
            return data
                

        except FileNotFoundError:
            print(f"The file {filename} could not be found")
            exit(1)


     def pre_process_data(self, data : list, proc : PreProcessor) -> list:
         """
         Perform the specified preprocessing on all files
         """
         processed_files = [proc.process_object(file) for file in data] 
         return processed_files
     

     def group_files_and_authors(self, authors : list, files : list) -> dict:
        """
        Returns a dictionary where each author is mapped to all files (s)he has written
        """
        grouped_by_author = {author : [] for author in authors}
        for i, author in enumerate(authors): grouped_by_author[author].append(files[i])
        return grouped_by_author
     

     def create_X_A_pairs(self, grouped_by_author : dict) -> list:
        """
        Creates all possible book author pairs from the corpus
        """
        X_A = []
        for author in grouped_by_author.keys():
            files = grouped_by_author[author]
            for i, file in enumerate(files):
                # Chunk the file (X) here to avoid duplicate work later
                chunked_file = self.chunker.chunk_files(file)
                for a in grouped_by_author.keys():
                    if a == author:
                        if len(files) > 1:
                            copied_files = deepcopy(files)
                            copied_files.pop(i)
                            X_A.append([True, file, chunked_file, copied_files])
                    else:
                        X_A.append([False, file, chunked_file, grouped_by_author[a]])
        
        return X_A


     def create_X_A_pairs_balanced(self, grouped_by_author : dict) -> list:
        same = []
        different = []
        for author in grouped_by_author.keys():
            files = grouped_by_author[author]
            for i, file in enumerate(files):
                # Chunk the file (X) here to avoid duplicate work later
                chunked_file = self.chunker.chunk_files(file)
                for a in grouped_by_author.keys():
                    if a == author:
                        if len(files) > 1:
                            copied_files = deepcopy(files)
                            copied_files.pop(i)
                            same.append([True, file, chunked_file, copied_files])
                    else:
                        different.append([False, file, chunked_file, grouped_by_author[a]])

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
        

     def handle_block(self, queue : Queue, block : list, clf : Unmasking) -> None:
        """
        Constructs the author curves from the provided block of X A pairs 
        """
        for X_A_pair in block:
            # Get the different elements in the X_A pair
            label = X_A_pair[0]
            X = X_A_pair[1]
            chunks_X = X_A_pair[2]
            A = X_A_pair[3]

            # Get the author curve
            author_curve = clf.handle_X_A_pair(X=X, A=A, chunks_X=chunks_X)

            # Now push the label and finished author cure to the que
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
     