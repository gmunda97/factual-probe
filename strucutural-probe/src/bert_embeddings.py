"""
Module to generate BERT embeddings for the entities
"""

from abc import ABC, abstractmethod
import torch
from transformers import BertModel, BertTokenizer


class BaseEmbeddings(ABC):
    '''
    Base class for creating embeddings
    '''
    @abstractmethod
    def __init__(self, model_name: str) -> None:
        pass

    @abstractmethod
    def __call__(self, entity_text: str) -> torch.Tensor:
        pass


class BERTEmbeddings(BaseEmbeddings):
    '''
    Class to create BERT embeddings by averaging
    and pooling the last hidden state
    '''
    def __init__(self, model_name: str) -> None:
        self.tokenizer = BertTokenizer.from_pretrained(model_name)
        self.model = BertModel.from_pretrained(model_name)

    def __call__(self, entity_text: str) -> torch.Tensor:
        inputs = self.tokenizer(entity_text, return_tensors='pt', padding=True, truncation=True)
        outputs = self.model(**inputs)
        embeddings = outputs.last_hidden_state.mean(dim=1)

        return embeddings


class BERTEmbeddingsWithCLS(BaseEmbeddings):
    '''
    Class to create BERT embeddings by extracting
    the [CLS] token from the last hidden state
    '''
    def __init__(self, model_name: str) -> None:
        self.tokenizer = BertTokenizer.from_pretrained(model_name)
        self.model = BertModel.from_pretrained(model_name)

    def __call__(self, entity_text: str) -> torch.Tensor:
        inputs = self.tokenizer(entity_text, return_tensors='pt', padding=True, truncation=True)
        outputs = self.model(**inputs)
        # Extract the hidden states from the last layer
        last_hidden_states = outputs.last_hidden_state

        # Extract the hidden state corresponding to [CLS] token (first token)
        cls_token_state = last_hidden_states[:, 0, :]

        return cls_token_state
