import logging
from functools import cache, cached_property
from typing import Any, Dict, Generator, List, Optional

import torch
from transformers import RobertaModel, RobertaTokenizerFast
from transformers.pipelines.feature_extraction import FeatureExtractionPipeline
from transformers.tokenization_utils_base import BatchEncoding, CharSpan, TokenSpan


class Text:
    AVERAGE_LAST_OUTPUT_LAYERS = 4

    def __init__(
        self,
        text: str,
        model: RobertaModel,
        tokenizer: RobertaTokenizerFast,
        metadata: Dict[str, Any] = None,
    ) -> None:
        if model.config.num_hidden_layers < self.AVERAGE_LAST_OUTPUT_LAYERS:
            raise ValueError(
                f"Model only has {model.config.num_hidden_layers} hidden layers, but {self.AVERAGE_LAST_OUTPUT_LAYERS} required."
            )

        if not model.config.output_hidden_states:
            logging.warning(
                "Setting model configuration 'output_hidden_states' to True."
            )
            model.config.output_hidden_states = True

        self._text = text
        self._model = model
        self._tokenizer = tokenizer
        self._metadata = metadata or {}

    def __repr__(self) -> str:
        return str({"metadata": self.metadata, "text": self.text})

    def __contains__(self, s: str):
        return s in self.text

    @property
    def text(self) -> str:
        return self._text

    @property
    def metadata(self) -> Dict[str, Any]:
        return self._metadata

    @property
    def model(self) -> RobertaModel:
        return self._model

    @property
    def tokenizer(self):
        return self._tokenizer

    @property
    def max_position_embeddings(self) -> int:
        return self.model.config.max_position_embeddings

    @cached_property
    def _embeddings(self):
        output = self.model(**self._encoding_tensors)
        return torch.stack(
            output.hidden_states[-self.AVERAGE_LAST_OUTPUT_LAYERS :]
        ).mean(axis=0)[0]

    @cached_property
    def _encoding(self) -> BatchEncoding:
        return self.tokenizer(self.text)

    @property
    def _encoding_tensors(self):
        return self.tokenizer(self.text, return_tensors="pt")

    @property
    def _token_ids(self):
        return self._encoding["input_ids"]

    @property
    def _word_ids(self) -> List[Optional[int]]:
        return self._encoding.word_ids()

    @property
    def _tokens(self) -> List[str]:
        return self.tokenizer.convert_ids_to_tokens(self._token_ids)

    def embeddings_matrix(self, token: str) -> List[torch.Tensor]:
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

    def _aggregate_embeddings(self, token_span: TokenSpan) -> torch.Tensor:
        """Return a one-dimensional array of the (aggregated) token embedding(s).

        Args:
            - token_span: a TokenSpan object pointing to the token(s) to aggregate

        Returns:
            A one-dimensional tensor with the length of the model embedding dimensionality
            for the token embedding at the given position, or the mean of multiple token embeddings
        """
        return self._embeddings[token_span.start : token_span.end].mean(axis=0)
