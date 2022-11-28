from transformers import AutoTokenizer, pipeline
from transformers.pipelines.feature_extraction import FeatureExtractionPipeline
from transformers.tokenization_utils_base import TokenSpan

from src import Context

MODEL_NAME = "DTAI-KULeuven/robbertje-1-gb-non-shuffled"
MODEL_MAX_LENGTH = 512

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME, model_max_length=MODEL_MAX_LENGTH, truncation=True
)
model: FeatureExtractionPipeline = pipeline(
    "feature-extraction",
    model=MODEL_NAME,
    tokenizer=tokenizer,
    max_length=MODEL_MAX_LENGTH,
    truncation=True,
)


class TestContext:
    def test_has_word(self):
        context: Context = Context(
            "Dit is een tekst.", model=model, metadata={}, token="is", char_index=4
        )

        assert context.has_word()

    def test_has_word_not(self):
        context: Context = Context(
            "Dit is een tekst.", model=model, metadata={}, token="TEST", char_index=4
        )

        assert not context.has_word()

    def test_has_subword_not(self):
        context: Context = Context(
            "Dit is duurzaamheid in een tekst.",
            model=model,
            metadata={},
            token="duurzaam",
            char_index=7,
        )

        assert not context.has_word()

    def test_has_multi_token_word(self):
        context = Context(
            "duurzaam", model=model, metadata={}, token="duurzaam", char_index=0
        )
        assert context.has_word()

    def test_has_multi_token_word_not(self):
        context = Context(
            "duurzaam", model=model, metadata={}, token="duur", char_index=0
        )
        assert not context.has_word()
