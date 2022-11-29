import torch
from transformers import AutoTokenizer, RobertaModel
from transformers.tokenization_utils_base import CharSpan, TokenSpan

from src.context import Context

MODEL_NAME = "DTAI-KULeuven/robbertje-1-gb-non-shuffled"
MODEL_MAX_LENGTH = 512

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME, model_max_length=MODEL_MAX_LENGTH, truncation=True
)
model: RobertaModel = RobertaModel.from_pretrained(MODEL_NAME)


class TestContext:
    def test_has_word(self):
        context: Context = Context(
            "Dit is een tekst.",
            model=model,
            tokenizer=tokenizer,
            metadata={},
            token="is",
            char_index=4,
        )

        assert context.has_word()

    def test_has_word_not(self):
        context: Context = Context(
            "Dit is een tekst.",
            model=model,
            tokenizer=tokenizer,
            metadata={},
            token="TEST",
            char_index=4,
        )

        assert not context.has_word()

    def test_has_subword_not(self):
        context: Context = Context(
            "Dit is duurzaamheid in een tekst.",
            model=model,
            tokenizer=tokenizer,
            metadata={},
            token="duurzaam",
            char_index=7,
        )

        assert not context.has_word()

    def test_has_multi_token_word(self):
        context = Context(
            "duurzaam",
            model=model,
            tokenizer=tokenizer,
            metadata={},
            token="duurzaam",
            char_index=0,
        )
        assert context.has_word()

    def test_has_multi_token_word_not(self):
        context = Context(
            "duurzaam",
            model=model,
            tokenizer=tokenizer,
            metadata={},
            token="duur",
            char_index=0,
        )
        assert not context.has_word()

    def test_token_embedding(self):
        context = Context(
            "Dit is duurzaamheid in een tekst.",
            model=model,
            tokenizer=tokenizer,
            metadata={},
            token="duurzaamheid",
            char_index=7,
        )
        assert context.token_embedding().shape == torch.Size([768])

    def test_token_embedding_none(self):
        context = Context(
            "Dit is duurzaamheid in een tekst.",
            model=model,
            tokenizer=tokenizer,
            metadata={},
            token="duur",
            char_index=7,
        )
        assert context.token_embedding() is None

    def test_word_to_chars(self):
        context = Context(
            "Dit is duurzaamheid in een tekst.",
            model=model,
            tokenizer=tokenizer,
            metadata={},
            token="duurzaamheid",
            char_index=7,
        )
        assert context._word_to_chars() == CharSpan(7, 19)

    def test_word_to_tokens(self):
        context = Context(
            "Dit is duurzaamheid in een tekst.",
            model=model,
            tokenizer=tokenizer,
            metadata={},
            token="duurzaamheid",
            char_index=7,
        )
        assert context._word_to_tokens() == TokenSpan(3, 4)
