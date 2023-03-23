#---------- Imported libraries ----------

from collections import Counter
from nltk.tokenize import word_tokenize
from nltk.probability import FreqDist
import numpy as np


#---------- FeatureExtractor class ----------

class FeatureExtractor:

    def __init__(self) -> None:
        pass

    def get_token_counters (self, tokens_X : list, tokens_A : list) -> tuple:
        """
        Returns a counter for X and A showing the counts for all unique tokens in each 
        """
        # Count the tokens in X and A
        counter_x = FreqDist(tokens_X)
        counter_a = FreqDist(tokens_A)

        return (counter_x, counter_a)
    
    def normalize_features(self, features_X, features_A):
        """
        Min-Max normalizes the features collected from X and A with respect to each other
        """
        combined = np.concatenate((features_X, features_A), axis=0)
        #print(combined)
        norm_X = np.zeros(shape=features_X.shape)
        norm_A = np.zeros(shape=features_A.shape)

        for col in range(combined.shape[1]):
            # Perform min-max normalization on a column basis
            min = np.amin(combined[:,col])
            max = np.amax(combined[:,col])
            if max > min:
                norm_X[:,col] = (features_X[:,col]-min)/(max-min)
                norm_A[:,col] = (features_A[:,col]-min)/(max-min)
        
        return (norm_X, norm_A)



    def vectorize_chunks(self, chunks : list, feature_set : list) -> np.ndarray:
        """
        Vectorizes the chunks in accordance with the specified feature set
        """
        #vectorized_chunks = [None]*len(chunks)
        vectorized_chunks = np.zeros(shape=(len(chunks), len(feature_set)))

        for i, chunk in enumerate(chunks):
            # Count all the unique tokens in the chunk
            chunk_token_freqs = Counter(chunk)

            # Determine the frequency of the features in the chunk
            for j, feature in enumerate(feature_set):
                if feature in chunk_token_freqs.keys():
                    vectorized_chunks[i,j] = chunk_token_freqs[feature]
            
        return vectorized_chunks


    def get_n_most_frequent_weighted_tokens(self, n : int, tokens_X : list, tokens_A : list, counter_x : Counter, counter_a : Counter) -> list:
        """
        Counts all tokens in both X and A and returns an ordered set of the most frequent tokens
        weighted evenly between X and A
        """
        # Get all the unique tokens shared by A and X
        total_tokens = set(list(counter_x.keys()) + list(counter_a.keys()))
        token_counts = [None] * len(total_tokens)

        # Iterate over the tokens and find their frequency in X and A
        for i, token in enumerate(total_tokens):
            counts = 0,
            if token in counter_x.keys():
                counts += counter_x[token]/len(tokens_X)
            if token in counter_a.keys():
                counts += counter_a[token]/len(tokens_A)
            token_counts[i] = (token, counts)
        
        # Sort the tokens based on the absolute difference in frequency between X and A
        token_counts = sorted(token_counts, reverse=True, key=lambda x : x[1])
        #print("Most frequent tokens:", [count[0] for count in token_counts[:n]])

        return [count[0] for count in token_counts[:n]]

    
    def vectorize_chunks_by_n_most_frequent_weighted_words(self, n : int, X : str, A : list, chunks_X : list, chunks_A : list, normalized : bool) -> tuple:
        """
        Returns the n most frequent words in X and A weighted equally
        """
        # Flatten A into the same shape as X
        A = " ".join(A)

        # Get the words from X and A
        words_X : list = word_tokenize(X)
        words_A : list = word_tokenize(A)

        # Get the counters for X and A
        counter_x, counter_a = self.get_token_counters(tokens_X=words_X, tokens_A=words_A)
        
        # Get the n most frequent words
        most_frequent_words : list = self.get_n_most_frequent_weighted_tokens(
                n=n,
                tokens_X=words_X,
                tokens_A=words_A,
                counter_x=counter_x,
                counter_a=counter_a
            )
        
        # Use the most frequent words to vectorize the chunks sampled from X and A
        features_X = self.vectorize_chunks(chunks=chunks_X, feature_set=most_frequent_words)
        features_A = self.vectorize_chunks(chunks=chunks_A, feature_set=most_frequent_words)

        if normalized:
            return self.normalize_features(features_X, features_A)
        
        return (features_X, features_A)