# ---------- Imported libraries ----------

import regex as re
from Scraper import Scraper



# ---------- CnnScrapper class ----------

class CnnScraper(Scraper):

    def __init__(self, GET: bool) -> None:
        
        # Base urls
        stem = "https://www.bbc.com/sitemaps/https-sitemap-com"
        base_urls = ["https://www.cnn.com/sitemaps/cnn/news.xml"]
            
        # Regex pattern for parsing the article urls
        regex_patterns = [r"https:\/\/www.bbc.com\/[\w\/-]{5,35}(?=[\w\s\-\<\>:\/]{110,150}>en<\/)",
                        r"https:\/\/www.bbc.com\/(sport|news|business)\/([\a-zA-Z\/\-]{4,25})?[\d]{8}"
                        ]

        # Initialize super class
        super().__init__(base_urls, regex_patterns, GET)

        print(len(self.articles))

        author_patterns = [r'(?<=">By )[\w\s]{4,25}(?=<\/div)']
        text_tags = ["p"]

        self.store_data(PATH="data\\news_articles\\the_bbc", 
                        author_patterns=author_patterns, 
                        text_tags=text_tags
                        )
    
