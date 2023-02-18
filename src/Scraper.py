# ---------- Imported libraries ----------
import requests
import regex as re
from time import sleep
from random import random
from datetime import datetime
from Utils import get_time, print_progressbar
from bs4 import BeautifulSoup
import hashlib
from time import sleep



# ---------- The base scrapper class ----------

class Scraper:

    def __init__(self, base_urls : list, regex_patterns : list, GET : bool) -> None:
        self.base_urls = base_urls

        if GET:
            # Get links to newsarticles
            self.urls = self.get_article_urls(base_urls, regex_patterns)
            print(len(self.urls))
            # Get news articles 
            self.articles = self.get_articles(self.urls)


    def get_article_urls(self, base_urls : list, regex_patterns : list) -> list:
        """
        Returns all the links to news articles listed on the sitemap page
        """
        print(f"{get_time()} - Getting sitemap pages and retreiving article urls...")
        result = []

        # Get each site map page for the provider
        for i, base_url in enumerate(base_urls):
            sitemap = requests.get(base_url).text

            # Some simple cleaining/normalization of the sitemap page
            #sitemap = re.sub(pattern=r"\s+", string=sitemap, repl="")
            
            # Iterate over all the provided patterns and retrive the matching urls
            for regex_pattern in regex_patterns:
                result += re.findall(pattern=regex_pattern, string=sitemap)
            
            print_progressbar(current_position=i, length=len(base_urls))

        print(f"{get_time()} - All article urls retrieved")

        # Return a list of unique urls to avoid duplicates
        return list(set(result))

    
    def get_article(self, url) -> str:
        """
        Get a single article while sleeping 0-1 seconds to throttle traffic
        """
        return requests.get(url).text

    
    def get_articles(self, article_links : list) -> list:
        """
        Fetches all the articles found on the sitemap page
        """
        length = len(self.urls)
        articles = [""]*length

        print(f"{get_time()} - Getting {length} article{'s' if length > 1 else ''}...")
        
        for i in range(length):
            try:
                if i % 100 == 0:
                    sleep(3)
                articles[i] = self.get_article(self.urls[i])
            except requests.exceptions.MissingSchema:
                # This occurs when there is something wrong with the parsed URl.
                # These URLs are just ignores
                pass
            print_progressbar(current_position=i, length=length)
        
        print(f"{get_time()} - Articles collected")

        return articles


    def store_data(self, PATH : str, author_patterns : list, text_tags : list) -> None:
        """
        Save the collected news articles
        """
        print(f"{get_time()} - Saving all {len(self.articles)} collected articles to disk...")
        na = 0
        na_list = []
        for i, article in enumerate(self.articles):
            try:
                    # Get the name of the author
                    author = self.get_author(article, author_patterns)

                    # Stops if no authpr could be found
                    if not len(author): raise AttributeError

                    # Get all the text elements in the article
                    text = self.get_article_text(article, text_tags)
                    
                    # Stops if the article is empty/no text could be retrieved
                    if not len(text): raise AttributeError

                    # Get filehash
                    title = hashlib.sha256(bytes(text, encoding="utf-8")).hexdigest()
                    # Writes author and text to file
                    with open(f"{PATH}/{title}.txt", "w+", encoding="utf-8", errors="ignore") as file:
                            file.write(f"{author}\n{text}")
                    
            # IO-errors
            except IOError or FileNotFoundError:
                print("Error when writing file nr. {i}: {title} to disk")

            # No authorname could be retrieved or the text is empty
            except AttributeError:
                na += 1
                na_list.append(i)

            print_progressbar(i, len(self.articles))

        print(f"{get_time()} - Articles saved to disk")
        print("Number of articles with no author:", na)
        try:
            print("Sample of article with no author: ", self.urls[na_list[0]])
        except Exception:
            print("error when printing na_list. na count:", na)

                        
    def get_author(self, article : str, author_patterns : list) -> str:
        """
        Loops over the provided author patterns to try and find the 
        name of the author in the article
        """
        author = ""
        for pattern in author_patterns:
            author = re.findall(pattern=pattern, string=article)
            if len(author):
                author = author[0]
                break
        return author


    def get_article_text(self, article, text_tags):
        """
        Finds all the specified text tags in the article and combines them into a single string
        """
        soup = BeautifulSoup(article, 'lxml')
        # Get all the text elements in the article
        text_elements = [element for element in soup.find_all(text=True) if element.parent.name in text_tags]
        text = "\n".join(text_elements)

        return text