import logging
import os
from math import ceil
from pathlib import Path
from typing import Generator, List

import pandas as pd
import spacy
import spacy.cli
from tqdm import tqdm

from .settings import ENCODING


class SentenceConverter:
    """Use SpaCy for sentence splitting"""

    def __init__(self, spacy_model) -> None:
        self._spacy = spacy_model

    def _process(self, text: str) -> Generator[str, None, None]:
        # TODO: prepend date?

        chunk_size = self._spacy.max_length
        n_chunks = ceil(len(text) / chunk_size)
        if n_chunks > 1:
            logging.warning(
                f"Chunking long text with {len(text)} characters into {n_chunks} chunks."
            )

        for chunk in range(0, n_chunks):
            start = chunk * chunk_size
            end = min((chunk + 1) * chunk_size, len(text))

            doc = self._spacy(text[start:end])
            for sent in doc.sents:
                yield str(sent) + os.linesep

    def convert_csv(
        self,
        filepath_or_buffer: Path,
        text_columns: List[str],
        *,
        compression="gzip",
        encoding: str = ENCODING,
        parse_dates: List[str] = ["date"],
        sep=";",
        **kwargs,
    ) -> Generator[str, None, None]:

        df: pd.DataFrame = pd.read_csv(
            filepath_or_buffer=filepath_or_buffer,
            sep=sep,
            compression=compression,
            encoding=encoding,
            # error_bad_lines=False,
            parse_dates=parse_dates,
            # **kwargs
        ).dropna(subset=text_columns)

        for row in tqdm(
            df.itertuples(), total=len(df), unit="row", desc=str(filepath_or_buffer)
        ):
            for column in text_columns:
                yield from self._process(row._asdict()[column])
            yield os.linesep

    @classmethod
    def from_spacy(cls, model_name: str):
        try:
            model = spacy.load(model_name)
        except OSError:
            spacy.cli.download(model_name)
            model = spacy.load(model_name)

        for component in model.pipe_names:
            model.remove_pipe(component)
        model.add_pipe("sentencizer")

        return cls(model)
