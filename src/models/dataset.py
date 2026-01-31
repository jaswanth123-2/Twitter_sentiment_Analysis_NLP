import torch
from torch.utils.data import Dataset
import pandas as pd
from transformers import BertTokenizer

class TweetSentimentDataset(Dataset):
    """
    PyTorch Dataset for tweet sentiment classification
    """
    
    def __init__(self, 
                 dataframe, 
                 tokenizer, 
                 max_length=64,
                 text_column='text_clean',
                 label_column='sentiment'):
        """
        Args:
            dataframe: pandas DataFrame with text and labels
            tokenizer: Hugging Face tokenizer
            max_length: Maximum sequence length
            text_column: Name of text column
            label_column: Name of label column
        """
        self.data = dataframe.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.text_column = text_column
        self.label_column = label_column
    
    def __len__(self):
        """Return the number of samples"""
        return len(self.data)
    
    def __getitem__(self, idx):
        """
        Get one sample
        
        Returns:
            dict with:
                - input_ids: Token IDs
                - attention_mask: Mask for padding
                - labels: Sentiment label (0 or 1)
        """
        # Get text and label
        text = str(self.data.loc[idx, self.text_column])
        label = int(self.data.loc[idx, self.label_column])
        
        # Tokenize
        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,      
            max_length=self.max_length,   
            padding='max_length',         
            truncation=True,              
            return_attention_mask=True,   
            return_tensors='pt'           
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }

if __name__ == "__main__":
    # Load data
    train_df = pd.read_csv(r'C:\Users\JASWA\Documents\Projects_AI\Twitter_sentiment_Analysis_NLP\data\processed\train.csv')
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    dataset = TweetSentimentDataset(
        dataframe=train_df.head(100), 
        tokenizer=tokenizer,
        max_length=64
    )
    
    print(f"Dataset size: {len(dataset)}")
    sample = dataset[0]
    
    print(f"\nSample structure:")
    print(f"Input IDs shape: {sample['input_ids'].shape}")
    print(f"Attention mask shape: {sample['attention_mask'].shape}")
    print(f"Label: {sample['labels']}")
    print(f"\nInput IDs (first 20): {sample['input_ids'][:20]}")
    print(f"Attention mask (first 20): {sample['attention_mask'][:20]}")
    # Decode to verify
    decoded = tokenizer.decode(sample['input_ids'])
    print(f"\nDecoded text: {decoded}")
    
    print("\n✓ Dataset class working correctly!")