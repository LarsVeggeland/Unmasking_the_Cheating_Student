# ---------- Imported libraries ----------

import regex as re
from Scraper import Scraper
import requests 


# ---------- CnnScrapper class ----------

class CnnScraper(Scraper):

    def __init__(self, GET: bool) -> None:
        
        # Base urls
        base_urls = self.get_sitemaps()
            
        # Regex pattern for parsing the article urls
        regex_patterns = [r"https:\/\/edition.cnn.com\/[\d]{4}\/[\d]{2}\/[\d]{2}\/[\w\/-]{10,60}\/index.html"]

        # Initialize super class
        super().__init__(base_urls, regex_patterns, GET)

        print(len(self.articles))

        author_patterns = [r'(?<=<span class="byline__name">)[\w\s]{4,20}(?=<\/span>)']
        text_tags = ["p"]

        self.store_data(PATH="data\\news_articles\\cnn", 
                        author_patterns=author_patterns, 
                        text_tags=text_tags
                        )
    

    def get_sitemaps(self) -> list:
        """
        This method finds the sitemap pages containg links to articles
        """
        url = "https://edition.cnn.com/sitemaps/cnn/index.xml"
        root_sitemap = requests.get(url).text
        patterns = [r"https:\/\/www.cnn.com\/sitemaps\/article-[\d]{4}-[\d]{2}.xml",
                    r"https:\/\/edition.cnn.com\/sitemaps\/article-[\d]{4}-[\d]{2}.xml"]

        article_sitemaps = []

        for pattern in patterns:
            article_sitemaps += re.findall(pattern=pattern, string=root_sitemap)

        # Restricting collection to the first 10 relevant sitemaps
        print(article_sitemaps[1])
        return article_sitemaps[10:30]

CnnScraper(GET=True)