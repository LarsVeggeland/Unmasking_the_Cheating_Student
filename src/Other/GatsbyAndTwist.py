# ---------- Imported libraries ---------

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from collections import Counter


# ---------- Util functions -----------

def retrieve_stopword_frequency(book : str) -> list:
    sw = stopwords.words('english')
    book = book.lower()
    tokens = word_tokenize(book)
    book_length = len(tokens)
    counts = Counter(tokens)
    freqs = [(w, counts[w]/book_length)  for w in counts.keys() if w in sw]
    freqs = {freq[0] : freq[1] for freq in freqs}
    for word in sw:
        if word not in freqs.keys():
            freqs[word] = 0
    return freqs


def open_book(filename : str) -> str:
    with open(filename, errors="ignore") as book:
        return book.read()


file1 = "D:\Downloads\The Great Gatsby.txt"
book1 = open_book(file1)
freqs1 = retrieve_stopword_frequency(book1)

file2 = "D:\Downloads\CharlesDickens-OliverTwist.txt"
book2 = open_book(file2)
freqs2 = retrieve_stopword_frequency(book2)

delta = [(word, freqs1[word], freqs2[word]) for word in freqs1.keys()]

delta = sorted(delta, reverse=True, key=lambda x : abs(x[1]-x[2]))[:10]

for i in delta:
    print(i)