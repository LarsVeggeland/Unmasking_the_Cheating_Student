#---------- Imported libraries ----------

from nltk.tokenize import word_tokenize
from nltk import ngrams, pos_tag
from random import shuffle
import math
import numpy as np
import regex as re
from copy import deepcopy



#---------- Chunking class ---------

class Chunking:

    def __init__(self, conf : dict) -> None:
        
        self.size = conf["size"]
        self.type = conf['type']
        self.method = conf["method"]

        if self.type == "ngrams":
            self.ngram_size = conf["ngram_size"]
        
        if self.method != "original":
            self.chunk_count = conf["chunk_count"]

        if self.method == "sliding_window":
            self.window_movement = conf["window_movement"]
        
        
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

        if self.method == "bootstrap":
            return self.bootstrap(tokens=tokens)
        
        if self.method == "sliding_window":
            # TODO 
            return self.sliding_window(tokens)


    def words(self, data : list) -> list:
        """
        Partitions every provided file into chunks of a specified length
        """
        # First check if the provided is a single or list of files
        filestream = data#self.get_filestream(data)
        
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
    

    def pos_tags(self, data : list) -> list:
        """
        Partitions the file into PoS tags
        """
        # First check if the provided is a single or list of files
        filestream = self.get_filestream(data)

        # Get all the pos tags from the filestream
        tags = [f"{tag[1]}" for tag in pos_tag(word_tokenize(filestream))]

        return tags

        # Get the chunks
        #chunks = self.chunk_tokens(tags)

    
    def lex_pos(self, data : list) -> list:
        """
        Partitions the file into lexpos tags
        """
        # First check if the provided is a single or list of files
        filestream = self.get_filestream(data)

        # Get all the pos tags from the filestream
        tags =  pos_tag(word_tokenize(filestream))
        lexpos = [f"{tag[0]}_{tag[1]}" for tag in tags]

        return lexpos


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

    def bootstrap(self, tokens : list) -> list:
        """
        Uses the randomized bootstrap method for smapling chunks
        """
        # Create the 2d chunk array
        chunks = [[None for j in range(self.size)] for i in range(self.chunk_count)]

        # Initialize random indices
        indices = [i for i in range(len(tokens))]
        shuffle(indices)

        for _, chunk in enumerate(chunks):

            for j in range(len(chunk)):
                 # Repopulate the random indicies if exhausted
                if not len(indices):
                    indices = [i for i in range(len(tokens))]
                    shuffle(indices)
                
                # Get an index from the list of shuffled indicies
                random_index = indices.pop()

                # Insert the token at the random index into the chunk
                chunk[j] = tokens[random_index]



        return chunks
    

    def sliding_window(self, tokens : list) -> list:
        """
        Uses the sliding window method for populating chunks
        """
        # Create the 2d chunk array
        chunks = [[None for j in range(self.size)] for i in range(self.chunk_count)]

        # Determine the start and end indicies for each window
        start_indicies = [(i*self.window_movement)%len(tokens) for i in range(self.chunk_count)]
        end_indicies =  [(i*self.window_movement + self.size)%len(tokens) for i in range(self.chunk_count)]

        # Create all the chunks using the start and end indicies
        for i in range(len(chunks)):
            start = start_indicies[i]
            end = end_indicies[i]

            if start < end:
                chunks[i] = tokens[start:end]
            # If end < start then a full circle of the tokens have been made. This must be handled differently
            else:
                chunks[i] = tokens[start:] + tokens[:end]
        
        return chunks



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
