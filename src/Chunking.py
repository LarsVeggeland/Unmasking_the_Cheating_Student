#---------- Imported libraries ----------

from nltk.tokenize import word_tokenize
from nltk import ngrams, pos_tag
import random
import math
import numpy as np



#---------- Chunking class ---------

class Chunking:

    def __init__(self, conf : dict) -> None:
        errors = []
        try:
            self.size = conf["size"]
            assert(self.size > 0)
        except AssertionError:
            errors.append(f"Chunk size must be a positive integer not {self.size}")

        try:
            self.type = conf['type']
            assert(self.type in ["words", "ngrams", "pos_tags", "lex_pos"])

            if self.type == "ngram":
                try:
                    self.ngram_size = conf["ngram_size"]
                    assert(self.ngram_size > 0 and isinstance(self.ngram_size, int))
                except AssertionError:
                    errors.append(f"ngram size must be a positive integer not {self.ngram_size}")
                except KeyError:
                        errors.append("Chunk type set to ngrams but ngram size is not defined in configuration file")
        except AssertionError:
            errors.append(f"The provided chunk type {self.type} does not exist")

        try:
            self.method = conf["method"]
            assert(self.method in ["original", "bootstrap", "sliding_window"])
        except AssertionError:
            errors.append(f"The provided chunk method {self.method} does not exits")
        
        if len(errors) > 0:
            error_msg = "\n".join(errors)
            raise AttributeError(f"There are issues with the provided configuration file:\n{error_msg}")
        

    def get_tokens(self, files :list) -> list:
        """
        Get the specified type of tokens from the file
        """
        token_function = eval(f"self.{self.type}")
        return token_function(files)
    
    
    def chunk_files(self, files : list) -> list:
        """
        Returns chunked versions of the provided files
        """
        tokens = self.get_tokens(files)
        
        if self.method == "original":
            return self.original(tokens)

        elif self.method == "bootstrap":
            # TODO
            return self.bootstrap(tokens, 25)


    def words(self, data : list) -> list:
        """
        Partitions every provided file into chunks of a specified length
        """
        # First check if the provided is a single or list of files
        filestream = self.get_filestream(data)
        
        # Get all the words in the stream in a list format
        words = word_tokenize(filestream)

        # Get the chunks
        #chunks = self.chunk_tokens(tokens=words)

        return words


    def ngrams(self, data : list) -> list:
        """
        Partitions the file into n-gram chunks
        """
        # First check if the provided is a single or list of files
        filestream = self.get_filestream(data)

        # Get all the character n-grams from the filestream
        grams = list(ngrams(filestream, self.ngram_size))[::self.ngram_size]

        #grams = [None]*math.ceil(len(filestream)/ngram_length)
        #for index, i in enumerate(range(0, len(filestream), ngram_length)):
            #if i + ngram_length < len(filestream):
                #grams[index] = filestream[i:i+ngram_length]
            #else:
                #grams[index] = filestream[i:] + " "*(len(filestream)-i)

        # Get the chunks 
        #chunks = self.chunk_tokens(tokens=grams)

        return grams
    

    def post_tags(self, data : list) -> list:
        """
        Partitions the file into pos tags
        """
        # First check if the provided is a single or list of files
        filestream = self.get_filestream(data)

        # Get all the pos tags from the filestream
        tags = pos_tag(filestream)

        # Get the chunks
        #chunks = self.chunk_tokens(tags)


    def original(self, tokens : list) -> list:
        """
        Partitions the tokens into chunks of the specified length
        """
         # Calculate the number of chunks required
        number_of_chunks = math.floor(len(tokens)/self.size) + bool(len(tokens) % self.size)
        res = [list() for c in range(number_of_chunks)]

        # Add each word to an appropriate chunk
        for i, token in enumerate(tokens):
            index = math.floor(i/self.size)
            res[index].append(token)
        
        return res

    def bootstrap(self, data : list, chunk_count : int) -> list:
        """
        Uses the randomized bootstrap method for smapling chunks
        """
        res = [None]*len(data)
        for i, file in enumerate(data):
            # Get all the words in the file
            words = word_tokenize(file)
            chunks = []

            for _ in range(chunk_count):
                # Create a new chunk
                chunk = [None]*self.size

                for j in range(self.size):
                    # If the pool of words have been exhausted simply replinish it
                    if len(words) == 0:
                        words += word_tokenize(file)
                        
                    # Calculate a random index and pop the element at that index from the pool of words
                    index = random.randrange(0,len(words))
                    chunk[j] = words.pop(index)
                
                chunks.append(chunk)
            res[i] = chunks
        return res
    

    def get_filestream(self, data : list) -> str:
        """
        Returns the provided data as a single string
        """
        if type(data) == list:
            file_stream = ""
            for file in data:
                file_stream += file + " "
        else:
            file_stream = data

        return file_stream