import argparse
import logging
import sys

from transformers import AutoTokenizer, PreTrainedTokenizerFast

logging.basicConfig(level=logging.INFO)


def pretrained_tokenizer(tokenizer) -> PreTrainedTokenizerFast:
    try:
        tokenizer = AutoTokenizer.from_pretrained(tokenizer)
    except OSError as e:
        raise ValueError(e)
    if not tokenizer.is_fast:
        raise ValueError(f"Not a fast tokenizer: '{tokenizer}'")
    assert isinstance(tokenizer, PreTrainedTokenizerFast)
    return tokenizer


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-i",
        "--input",
        type=argparse.FileType("rt"),
        default=sys.stdin,
        help="The input file with one sentence per line. Defaults to stdin.",
    )
    parser.add_argument(
        "--old-tokenizer",
        type=pretrained_tokenizer,
        required=True,
        help="Existing tokenizer to use. Must be a fast tokenizer",
    )
    parser.add_argument(
        "--vocab-size",
        type=int,
        required=False,
        help="The size of the vocabulary you want for your tokenizer. Defaults to the size as specified by the old tokenizer model.",
    )
    parser.add_argument(
        "--length",
        type=int,
        required=False,
        help="The total number of sequences in the input file. This is used to provide meaningful progress tracking",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        required=True,
        help="The directory for storing the tokenizer.",
    )

    args = parser.parse_args()

    vocab_size = args.vocab_size or args.old_tokenizer.vocab_size
    logging.info(f"Setting vocabulary size to {vocab_size}.")

    corpus = (line for line in args.input if line.strip())

    tokenizer = args.old_tokenizer.train_new_from_iterator(
        corpus, vocab_size=vocab_size, length=args.length
    )
    tokenizer.save_pretrained(args.output)
