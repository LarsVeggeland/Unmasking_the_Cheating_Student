# ---------- Imported libraries ----------

import regex as re
from Scraper import Scraper



# ---------- BbcScrapper class ----------

class BbcScraper(Scraper):

    def __init__(self, GET: bool) -> None:
        
        # Base urls
        stem = "https://www.bbc.com/sitemaps/https-sitemap-com"
        base_urls = [f"{stem}-news-{i}.xml" for i in range(1, 4)] + [f"{stem}-archive-{i}.xml" for i in range(1,6)]
            
        # Regex pattern for parsing the article urls
        regex_patterns = [r"https:\/\/www.bbc.com\/[\w\/-]{5,35}(?=[\w\s\-\<\>:\/]{110,150}>en<\/)",
                        r"https:\/\/www.bbc.com\/(sport|news)\/([\a-zA-Z\/\-]{4,25})?[\d]{8}"
                        ]

        # Initialize super class
        super().__init__(base_urls, regex_patterns, GET)

        print(len(self.articles))

BbcScraper(GET=True)