import re
import emoji
import contractions
from typing import List
import html

class TweetCleaner:
    
    def __init__(self, 
                 lowercase=True,
                 remove_urls=True,
                 remove_mentions=True,
                 remove_hashtags=False,
                 remove_numbers=False,
                 remove_emojis=False,
                 expand_contractions=True):
        
        self.lowercase = lowercase
        self.remove_urls = remove_urls
        self.remove_mentions = remove_mentions
        self.remove_hashtags = remove_hashtags
        self.remove_numbers = remove_numbers
        self.remove_emojis = remove_emojis
        self.expand_contractions = expand_contractions
    
    def clean(self, text: str) -> str:

        if not isinstance(text, str):
            return ""
        
        # Decode HTML entities
        text = html.unescape(text)
        
        # Remove URLs
        if self.remove_urls:
            text = re.sub(r'http\S+|www.\S+', '', text)
        
        # Remove mentions (@username)
        if self.remove_mentions:
            text = re.sub(r'@\w+', '', text)
        
        # Remove hashtags (or just the # symbol)
        if self.remove_hashtags:
            text = re.sub(r'#\w+', '', text)
        else:
            text = re.sub(r'#', '', text)  # Remove # but keep word
        
        # Expand contractions (don't -> do not)
        if self.expand_contractions:
            text = contractions.fix(text)
        
        # Remove emojis
        if self.remove_emojis:
            text = emoji.replace_emoji(text, replace='')
        
        # Remove numbers
        if self.remove_numbers:
            text = re.sub(r'\d+', '', text)
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        # Lowercase
        if self.lowercase:
            text = text.lower()
        
        return text
    
    def clean_batch(self, texts: List[str]) -> List[str]:
        return [self.clean(text) for text in texts]


if __name__ == "__main__":
    cleaner = TweetCleaner()