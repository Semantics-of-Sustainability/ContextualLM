import string
from dataclasses import dataclass
from functools import cached_property
from typing import Any, Dict, Generator, List, Optional, Tuple

import numpy as np
from numpy.typing import ArrayLike
from transformers.pipelines.feature_extraction import FeatureExtractionPipeline
from transformers.tokenization_utils_base import BatchEncoding, CharSpan, TokenSpan

BEGIN_OF_WORD_CHAR = "Ġ"


# TODO: Rename class to 'Text'
class Sentence:
    text: str
    model: FeatureExtractionPipeline

    def __init__(
        self, text: str, model: FeatureExtractionPipeline, metadata: Dict[str, Any]
    ) -> None:
        self._text = text
        self._model = model
        self._metadata = metadata

    def __contains__(self, s: str):
        return s in self.text

    @property
    def text(self) -> str:
        return self._text

    @property
    def metadata(self) -> Dict[str, Any]:
        return self._metadata

    @property
    def model(self) -> FeatureExtractionPipeline:
        return self._model

    @property
    def tokenizer(self):
        return self.model.tokenizer

    @property
    def max_position_embeddings(self) -> int:
        return self.model.model.config.max_position_embeddings

    @cached_property
    def _embeddings(self):
        # assuming batch size is 0
        return self.model(self.text)[0]

    @cached_property
    def _encoding(self) -> BatchEncoding:
        return self.tokenizer(self.text)

    @property
    def _token_ids(self):
        return self._encoding["input_ids"]

    @property
    def _word_ids(self) -> List[Optional[int]]:
        return self._encoding.word_ids()

    @property
    def _tokens(self) -> List[str]:
        return self.model.tokenizer.convert_ids_to_tokens(self._token_ids)

    def embeddings_matrix(self, token: str) -> List[ArrayLike]:
        return [
            self._aggregate_embeddings(token_span)
            for token_span in self._find_token(token)
        ]

    def _find_token(self, token: str) -> Generator[TokenSpan, None, None]:
        """
        Find sequences of (sub-word) tokens that match a (word) token if merged

        Args:
            - token: a token (word) to find, exact match
        Yields: TokenSpan objects representing the tokens matching the search token
        """

        for word_index in range(len(self._word_ids)):
            if (
                word_index is not None
                and self._encoding.word_to_tokens(word_index) is not None
            ):
                char_span: CharSpan = self._encoding.word_to_chars(word_index)
                if self.text[char_span.start : char_span.end] == token:
                    yield self._encoding.word_to_tokens(word_index)

    def _aggregate_embeddings(self, token_span: TokenSpan) -> ArrayLike:
        """Return a one-dimensional array of the (aggregated) token embedding(s).

        Args:
            - token_span: a TokenSpan object pointing to the token(s) to aggregate

        Returns:
            A vector of shape (<embedding dimensionality, 1)
            for the token embedding at the given position, or the mean of multiple token embeddings
        """
        a: ArrayLike
        n_tokens = token_span.end - token_span.start

        if n_tokens == 1:
            a = np.array(self._embeddings[token_span.start])
        else:
            a = np.array(self._embeddings[token_span.start : token_span.end]).mean(
                axis=0
            )

        assert a.shape == (
            self.model.model.config.hidden_size,
        ), f"Invalid shape: {a.shape} for token span {token_span}."

        return a

    def to_contexts(
        self, token: str, context_characters: int
    ) -> Generator["Context", None, None]:
        """Generate a new Context object for each occurence of a term with a smaller window.

        Can be empty of the token does not occur in the text.
        """

        from .context import Context

        return Context.contexts(
            self._text, token, context_characters, self.model, self.metadata
        )
