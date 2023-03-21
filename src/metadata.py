#---------- imported libraries ----------

import pandas as pd
from matplotlib import pyplot as plt
from collections import Counter
from nltk.tokenize import word_tokenize
import numpy as np
from math import floor, log2

#---------- util funtions ----------

def get_data(filename : str) -> pd.DataFrame:
    return pd.read_csv(filename, encoding="utf-8", encoding_errors="ignore")


def get_article_count_per_author(data : pd.DataFrame) -> Counter:
    authors = data["authors"].to_list()
    return Counter(authors)


def get_words_per_article(data : pd.DataFrame) -> np.ndarray:
    articles = data["articles"].to_list()
    word_counts = np.zeros(len(articles))

    for i, article in enumerate(articles):
        word_counts[i] = len(word_tokenize(article))
    
    return word_counts


def get_avg_words_per_article(data : pd.DataFrame) -> float:
    word_counts = get_words_per_article(data)
    return np.sum(word_counts)/len(word_counts)


def get_article_length_distribution(word_counts : np.ndarray) -> tuple:
    mean = np.mean(word_counts)
    std = np.std(word_counts)
    return (mean, std)

def authors_grouped_by_article_count(data : pd.DataFrame) -> dict:
    counts = get_article_count_per_author(data)
    groups = {(i)*10 : 0 for i in range(21)}
    print(groups)

    for value in counts.values():
        index = 10 * floor(value/10) if value < 200 else 200
        groups[index] += 1
    
    for key, value in groups.items():
        if value != 0: groups[key] = value

    return groups


def group_articles_by_word_count(data : pd.DataFrame) -> dict:
    counts = get_words_per_article(data)
    groups = {(i)*100 : 0 for i in range(31)}
    print(groups)

    for value in counts:
        index = 100 * floor(value/100) if value < 3000 else 3000
        groups[index] += 1
    
    for key, value in groups.items():
        if value != 0: groups[key] = value

    return groups


def authors_above_threshold(data : pd.DataFrame, words_threshold : int, article_threshold : int) -> list:
    authors = data["authors"]

    word_counts = get_words_per_article(data)

    above_threshold = dict()

    for author in list(set(authors)):
        above_threshold[author] = 0

    for i, author in enumerate(authors):
        count = word_counts[i]
        above_threshold[author] += count >= words_threshold
    
    acceptable_authors = []
    for author, count in above_threshold.items():
        if count >= article_threshold:
            acceptable_authors.append((author, count))
    
    return acceptable_authors


def plot_author_article_count_groups(data : pd.DataFrame, log : bool, bins : int) -> None:
    x = [value for value in get_article_count_per_author(data).values()]
    plt.xlabel("Number of books written")
    plt.ylabel("Number of authors")

    plt.hist(x=x, bins=bins, log=log)
    plt.show()


def plot_article_word_count_groups(data : pd.DataFrame, log : bool, bins : int) -> None:
    x = get_words_per_article(data)
    #x = x[x < 3000]
    plt.xlabel("Book length in words")
    plt.ylabel("Number of articles")

    plt.hist(x=x, bins=bins, log=log)
    plt.show()

data = get_data("data/datasets/books.csv")
#plot_author_article_count_groups(data, False, 3)
plot_article_word_count_groups(data, False, 8)