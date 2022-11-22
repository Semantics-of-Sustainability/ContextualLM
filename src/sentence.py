import string
from dataclasses import dataclass
from functools import cached_property
from typing import List, Tuple

import numpy as np
from numpy.typing import ArrayLike
from transformers.pipelines.feature_extraction import FeatureExtractionPipeline

BEGIN_OF_WORD_CHAR = "Ġ"


@dataclass
class Sentence:
    text: str
    model: FeatureExtractionPipeline
    year: int
    filename: str

    def __contains__(self, s: str):
        return s in self.text

    @property
    def tokenizer(self):
        return self.model.tokenizer

    @property
    def max_position_embeddings(self) -> int:
        return self.model.model.config.max_position_embeddings

    @cached_property
    def _embeddings(self):
        return self.model(self.text)[0]

    @cached_property
    def _token_ids(self):
        return self.tokenizer(self.text)["input_ids"]

    @property
    def _tokens(self) -> List[str]:
        return self.model.tokenizer.convert_ids_to_tokens(self._token_ids)

    def embeddings_matrix(self, token: str) -> List[ArrayLike]:
        return [
            self._aggregate_embeddings(token_i, token_length)
            for token_i, token_length in self._find_token(token)
        ]

    def _find_token(self, token: str, max_n: int = 5) -> Tuple[int, int]:
        """
        Find sequences of (sub-word) tokens that match a (word) token if merged

        Args:
            - token: a token (word) to find, exact match
            - max_n: the maximum number of sub-word tokens to be merged
        Yields: Tuple[int, int]: the token index and the length (number of tokens to be merged)
        """

        last_token_i = min(len(self._tokens), self.max_position_embeddings - 1)

        for window_size in range(1, max_n):
            for window_index in range(1, last_token_i):
                if window_index + window_size < last_token_i:
                    first_token = self._tokens[window_index]
                    merged_token = self.tokenizer.convert_tokens_to_string(
                        self._tokens[window_index : window_index + window_size]
                    )
                    next_token = self._tokens[window_index + window_size]

                    is_word_begin = window_index == 1 or first_token.startswith(
                        BEGIN_OF_WORD_CHAR
                    )
                    is_word_end = (
                        next_token.startswith(BEGIN_OF_WORD_CHAR)
                        or next_token == self.tokenizer.special_tokens_map["eos_token"]
                        or all(c in string.punctuation for c in next_token)
                    )

                    if is_word_begin and merged_token.strip() == token and is_word_end:
                        yield window_index, window_size

    def _aggregate_embeddings(self, token_start_index: int, n_tokens: int) -> ArrayLike:
        """Return a one-dimensional array of the (aggregated) token embedding(s).

        Args:
            - token_start_index: the token position in the sentence, according to the model's tokenizer
            - n_tokens: the number of tokens to aggregate (1 for a single token)

        Returns:
            A vector of shape (<embedding dimensionlaity, 1)
            for the token embedding at the given position, or the mean of multiple token embeddings
        """
        a: np.array
        if n_tokens == 1:
            a = np.array(self._embeddings[token_start_index])
        else:
            a = np.array(
                self._embeddings[token_start_index : token_start_index + n_tokens]
            ).mean(axis=0)
        assert a.shape == (
            self.model.model.config.hidden_size,
        ), f"Invalid shape: {a.shape} for token index {token_start_index}, token length {n_tokens}."
        return a
