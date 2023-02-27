# ---------- Imported libraries ----------

import regex as re
from Scraper import Scraper



# ---------- BbcScrapper class ----------

class BbcScraper(Scraper):

    def __init__(self, GET: bool) -> None:
        
        # Base urls
        stem = "https://www.bbc.com/sitemaps/https-sitemap-com"
        #base_urls = [f"{stem}-archive-{i}.xml" for i in range(2,5)] # + [f"{stem}-news-{i}.xml" for i in range(1, 4)]]
        base_urls = [f"https://www.bbc.com/sitemaps/https-sitemap-com-archive-{i}.xml" for i in range(13, 14)]   
        # Regex pattern for parsing the article urls
        regex_patterns = [r"https:\/\/www.bbc.com\/news\/[\d]{8}",
        r"https:\/\/www.bbc.com\/sport\/[\d]{8}",
        r"https:\/\/www.bbc.com\/business\/[\d]{8}",
        r"https:\/\/www.bbc.com\/sport\/[\a-zA-Z\/\-]{4,35}[\d]{8}",
        r"https:\/\/www.bbc.com\/news\/[\a-zA-Z\/\-]{4,35}[\d]{8}",
        r"https:\/\/www.bbc.com\/sport\/[\a-zA-Z\/\-]{4,35}[\d]{8}",
        r"https:\/\/www.bbc.com\/business\/[\a-zA-Z\/\-]{4,35}[\d]{8}"]
        
        #[r"https:\/\/www.bbc.com\/news\/[\d]{8}", 
        #r"https:\/\/www.bbc.com\/(sport|news|business)\/([\a-zA-Z\/\-]{4,35})?[\d]{8}"]  #[r"https:\/\/www.bbc.com\/[\w\/-]{5,35}(?=[\w\s\-\<\>:\/]{110,150}>en<\/)",

        # Initialize super class
        super().__init__(base_urls, regex_patterns, GET)

        print(len(self.articles))

        author_patterns = [r'(?<=">By )[\w\s]{4,25}(?=<\/div)']
        text_tags = ["p"]

        self.store_data(PATH="data\\news_articles\\the_bbc", 
                        author_patterns=author_patterns, 
                        text_tags=text_tags
                        )
    

BbcScraper(GET=True)