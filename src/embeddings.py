'''Module to extract embeddings from PLMs for the entities'''

from abc import ABC, abstractmethod
import torch
from transformers import AutoTokenizer, BertModel, BertTokenizer, GPT2Model, GPT2Tokenizer, \
    BartModel, BartTokenizer, ModernBertModel


class BaseEmbeddings(ABC):
    '''
    Base class for creating embeddings
    '''
    @abstractmethod
    def __init__(self, model_name: str) -> None:
        pass

    @abstractmethod
    def __call__(self, entity_text: str, entity_description: str = None) -> torch.Tensor:
        pass


class BERTEmbeddings(BaseEmbeddings):
    '''
    Class to create BERT embeddings by averaging
    and pooling the last hidden state
    '''
    def __init__(self, model_name: str) -> None:
        self.tokenizer = BertTokenizer.from_pretrained(model_name)
        self.model = BertModel.from_pretrained(model_name)

    def pool(self, last_hidden_state: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        mask = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
        summed = (last_hidden_state * mask).sum(dim=1)
        counts = mask.sum(dim=1).clamp(min=1e-9)
        return summed / counts

    def __call__(self, entity_text: str, entity_description: str = None) -> torch.Tensor:
        if entity_description:
            entity_text = entity_text + ' [SEP] ' + entity_description
        inputs = self.tokenizer(entity_text, return_tensors='pt', padding=True, truncation=True)
        outputs = self.model(**inputs)
        embeddings = self.pool(outputs.last_hidden_state)

        return embeddings


class BERTEmbeddingsWithCLS(BaseEmbeddings):
    '''
    Class to create BERT embeddings by extracting
    the [CLS] token from the last hidden state
    '''
    def __init__(self, model_name: str) -> None:
        self.tokenizer = BertTokenizer.from_pretrained(model_name)
        self.model = BertModel.from_pretrained(model_name)

    def __call__(self, entity_text: str, entity_description: str = None) -> torch.Tensor:
        if entity_description:
            entity_text = entity_text + ' [SEP] ' + entity_description
        inputs = self.tokenizer(entity_text, return_tensors='pt', padding=True, truncation=True)
        outputs = self.model(**inputs)
        # Extract the hidden states from the last layer
        last_hidden_states = outputs.last_hidden_state

        # Extract the hidden state corresponding to [CLS] token (first token)
        cls_token_state = self.pool(last_hidden_states)

        return cls_token_state

    def pool(self, last_hidden_state: torch.Tensor, attention_mask: torch.Tensor = None) -> torch.Tensor:
        return last_hidden_state[:, 0, :]


class GPTEmbeddings(BaseEmbeddings):
    '''
    Class to create GPT-2 embeddings by averaging
    and pooling the last hidden state
    '''
    def __init__(self, model_name: str) -> None:
        self.tokenizer = GPT2Tokenizer.from_pretrained(model_name)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = GPT2Model.from_pretrained(model_name)

    def __call__(self, entity_text: str, entity_description: str = None) -> torch.Tensor:
        if entity_description:
            entity_text = entity_text + ' [SEP] ' + entity_description
        inputs = self.tokenizer(entity_text, return_tensors='pt', padding=True, truncation=True)
        outputs = self.model(**inputs)
        embeddings = outputs.last_hidden_state.mean(dim=1)

        return embeddings


class BARTEmbeddings(BaseEmbeddings):
    '''
    Class to create BART embeddings by averaging
    and pooling the last hidden state
    '''
    def __init__(self, model_name: str) -> None:
        self.tokenizer = BartTokenizer.from_pretrained(model_name)
        self.model = BartModel.from_pretrained(model_name)

    def __call__(self, entity_text: str, entity_description: str = None) -> torch.Tensor:
        if entity_description:
            entity_text = entity_text + ' [SEP] ' + entity_description
        inputs = self.tokenizer(entity_text, return_tensors='pt', padding=True, truncation=True)
        outputs = self.model(**inputs)
        embeddings = outputs.last_hidden_state.mean(dim=1)

        return embeddings


class ModernBERTEmbeddings(BaseEmbeddings):
    '''
    Class to create ModernBERT embeddings by averaging
    and pooling the last hidden state
    '''
    def __init__(self, model_name: str) -> None:
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = ModernBertModel.from_pretrained(model_name)

    def pool(self, last_hidden_state: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        mask = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
        summed = (last_hidden_state * mask).sum(dim=1)
        counts = mask.sum(dim=1).clamp(min=1e-9)
        return summed / counts

    def __call__(self, entity_text: str, entity_description: str = None) -> torch.Tensor:
        if entity_description:
            entity_text = entity_text + ' ' + entity_description
        inputs = self.tokenizer(entity_text, return_tensors='pt', padding=True, truncation=True)
        outputs = self.model(**inputs)
        embeddings = self.pool(outputs.last_hidden_state, inputs['attention_mask'])

        return embeddings


class ModernBERTEmbeddingsWithCLS(BaseEmbeddings):
    '''
    Class to create ModernBERT embeddings by extracting
    the [CLS] token from the last hidden state
    '''
    def __init__(self, model_name: str) -> None:
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = ModernBertModel.from_pretrained(model_name)

    def pool(self, last_hidden_state: torch.Tensor, attention_mask: torch.Tensor = None) -> torch.Tensor:
        return last_hidden_state[:, 0, :]

    def __call__(self, entity_text: str, entity_description: str = None) -> torch.Tensor:
        if entity_description:
            entity_text = entity_text + ' ' + entity_description
        inputs = self.tokenizer(entity_text, return_tensors='pt', padding=True, truncation=True)
        outputs = self.model(**inputs)
        cls_token_state = self.pool(outputs.last_hidden_state)

        return cls_token_state

