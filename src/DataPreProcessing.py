#------------ Imported libraries ----------

import regex as re
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.stem import PorterStemmer, WordNetLemmatizer
from collections import Counter



#---------- PreProcessor class ------------

class PreProcessor:

    
    def __init__(self, jobs : list = None) -> None:

        existing_jobs = [
                'lowercase',
                #'no_urls',
                'no_punctuations',
                #'no_nums',
                'remove_special_chars',
                #'stopwords',
                #'spelling',
                #'lemmatize',
                #'stem',
                #'fix_ws'
            ]

        if jobs is None: jobs = existing_jobs
        
        for i in range(len(jobs)):
            job = jobs[i]
            if job in existing_jobs:
                jobs[i] = eval(f'self.{job}')
            else:
                raise Exception(f"The job {job} is not available")

        self.jobs = jobs
        self.stopwords_map = Counter(stopwords.words('english')+['urllink', 'nbsp'])         
        self.lemmatizer = WordNetLemmatizer()
        self.stemmer = PorterStemmer()
        self.cut_sequence = re.compile(r"(.)\1{2,}")
        self.url = re.compile(r"https?://www[a-z0-9#!$%&@\./<>*=_\+()\\]+")
        self.punctuations = re.compile(r"[\.,\?!]+")
        self.nums = re.compile(r'\d+')
        self.special_chars = re.compile(r"[#$%\^%&*';:|\-_\=`~<>/\(\)]+")
        self.leftovers = re.compile(r"\s+\w\s+")
        self.ws = re.compile(r"\s+")
        self.quote = re.compile(r'"[\w\s]"')


    def process_object(self, file : str) -> str:
        """
        Performs the predefined set of jobs on the file before returning it
        """
        for job in self.jobs:
            file = job(file)
        return file



    #---------- Available Normalization Jobs ----------

    def lowercase(self, file : str) -> str:
        """ Converts all characters to lowercase """
        return file.lower()


    def stopwords(self, file)-> str:
        """ Filters out all english stopwords and other undesierable tokens"""
        return " ".join(word for word in file.split(" ") if word not in self.stopwords_map)
    

    def no_punctuations(self, file) -> str:
        """ Removes all periods, commas, exclamation- , and question marks """
        return self.punctuations.sub(" ", file)
     

    def no_urls(self, file : str) -> str:
        """ Removes all hyperlinks """
        return self.url.sub(" ", file)

    
    def no_nums(self, file : str) -> str:
        """ Removes all digits """
        return self.nums.sub(" ", file)

    
    def remove_special_chars(self, file : str) -> str:
        return self.special_chars.sub(" ", file)


    def spelling(self, file) -> str:
        """ Shortens all one-char sequences longer than 2 to 2 """
        fixed_length = self.cut_sequence.sub(r"\1\1", file)
        return fixed_length


    def lemmatize(self, file) -> str:
        """ Converts all words to their lemma """
        return " ".join(self.lemmatizer.lemmatize(word) for word in file.split(" "))


    def stem(self, file) -> str:
        """ Stems all words in file """
        return " ".join(self.stemmer.stem(word) for word in file.split(" "))


    def fix_ws(self, file : str) -> str:
        """ Replaces all sequences of whitespace characters with a single whitespace """
        return self.ws.sub(" ", file)
    
    
    def normalize_quotes():
        # TODO 
        pass