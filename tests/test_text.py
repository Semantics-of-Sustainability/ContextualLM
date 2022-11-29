import torch
from transformers import AutoTokenizer, RobertaModel
from transformers.tokenization_utils_base import TokenSpan

from src import Text

MODEL_NAME = "roberta-base"

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model: RobertaModel = RobertaModel.from_pretrained(MODEL_NAME)


class TestText:
    text = Text("This is a text.", model=model, tokenizer=tokenizer)
    text_token_ids = [0, 713, 16, 10, 2788, 4, 2]

    def test_embeddings(self):
        assert self.text._embeddings.shape == torch.Size(
            [len(self.text_token_ids), model.config.hidden_size]
        )

    def test_token_ids(self):
        assert self.text._token_ids == self.text_token_ids

    def test_tokens(self):
        assert self.text._tokens == ["<s>", "This", "Ġis", "Ġa", "Ġtext", ".", "</s>"]

    def test_word_ids(self):
        assert self.text._word_ids == [None] + list(range(5)) + [None]

    def test_find_token(self):
        assert list(self.text._find_token("is")) == [TokenSpan(2, 3)]

    def test_find_token_not(self):
        assert list(self.text._find_token("his")) == []
        assert list(self.text._find_token("tex")) == []
