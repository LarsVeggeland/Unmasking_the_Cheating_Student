# ---------- Imported libraries ----------

import regex as re
from Scraper import Scraper



# ---------- The Sun Scrapper ----------

class SunScraper(Scraper):

    def __init__(self, GET : bool) -> None:
        # Base urls
        base_urls = ["https://www.thesun.co.uk/news-sitemap.xml"]

        # Pattern for parsing the article urls
        regex_patterns = [r"(?<=<loc>)https:\/\/www.thesun.co.uk\/[a-zA-Z0-9\/-]{5,150}(?=<\/loc>)"]

        # Initialize super class
        super().__init__(base_urls, regex_patterns, GET)
        
        # Remove noise at the end
        self.cutoff_end()

        # Parse and svae the collected articles
        author_patterns = [r'(?<=aria-label="Posts by )' + r"[a-zA-z'\s]{3,30}" + r'(?=">)',
                            r'(?<=<span class="article__author-name t-p-color">)' + r"[a-zA-Z'\s]{3,30}(?=<\/span>)"
                            ]
        text_tags = ["p"]
        self.store_data(PATH="data\\news_articles\\the_sun", 
                        author_patterns=author_patterns, 
                        text_tags=text_tags
                        )
    

    def cutoff_end(self) -> None:
        """
        Removes noise found at the end of every The Sun article
        """
        failures = 0
        for i, article in enumerate(self.articles):
            try:
                pos = re.search(pattern=r"News Group Newspapers Limited in England No",
                            string=article
                            ).start()
                self.articles[i] = article[:pos-1]
            except AttributeError as a:
                print(a)
                failures += 1
                
    
        print("Failures:", failures)

SunScraper(GET=True)
