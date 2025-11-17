"""
Whisper tokenizer with support for 99 languages.

Provides a wrapper around tiktoken for BPE tokenization with Whisper-specific
special tokens including language tokens, task tokens, and timestamp tokens.

Based on OpenAI's Whisper tokenizer implementation.
"""

import base64
import json
import os
import shutil
import string
from dataclasses import dataclass, field
from functools import cache, cached_property
from pathlib import Path
from typing import Optional

import tiktoken

# Supported languages (99 total)
LANGUAGES = {
    "en": "english",
    "zh": "chinese",
    "de": "german",
    "es": "spanish",
    "ru": "russian",
    "ko": "korean",
    "fr": "french",
    "ja": "japanese",
    "pt": "portuguese",
    "tr": "turkish",
    "pl": "polish",
    "ca": "catalan",
    "nl": "dutch",
    "ar": "arabic",
    "sv": "swedish",
    "it": "italian",
    "id": "indonesian",
    "hi": "hindi",
    "fi": "finnish",
    "vi": "vietnamese",
    "he": "hebrew",
    "uk": "ukrainian",
    "el": "greek",
    "ms": "malay",
    "cs": "czech",
    "ro": "romanian",
    "da": "danish",
    "hu": "hungarian",
    "ta": "tamil",
    "no": "norwegian",
    "th": "thai",
    "ur": "urdu",
    "hr": "croatian",
    "bg": "bulgarian",
    "lt": "lithuanian",
    "la": "latin",
    "mi": "maori",
    "ml": "malayalam",
    "cy": "welsh",
    "sk": "slovak",
    "te": "telugu",
    "fa": "persian",
    "lv": "latvian",
    "bn": "bengali",
    "sr": "serbian",
    "az": "azerbaijani",
    "sl": "slovenian",
    "kn": "kannada",
    "et": "estonian",
    "mk": "macedonian",
    "br": "breton",
    "eu": "basque",
    "is": "icelandic",
    "hy": "armenian",
    "ne": "nepali",
    "mn": "mongolian",
    "bs": "bosnian",
    "kk": "kazakh",
    "sq": "albanian",
    "sw": "swahili",
    "gl": "galician",
    "mr": "marathi",
    "pa": "punjabi",
    "si": "sinhala",
    "km": "khmer",
    "sn": "shona",
    "yo": "yoruba",
    "so": "somali",
    "af": "afrikaans",
    "oc": "occitan",
    "ka": "georgian",
    "be": "belarusian",
    "tg": "tajik",
    "sd": "sindhi",
    "gu": "gujarati",
    "am": "amharic",
    "yi": "yiddish",
    "lo": "lao",
    "uz": "uzbek",
    "fo": "faroese",
    "ht": "haitian creole",
    "ps": "pashto",
    "tk": "turkmen",
    "nn": "nynorsk",
    "mt": "maltese",
    "sa": "sanskrit",
    "lb": "luxembourgish",
    "my": "myanmar",
    "bo": "tibetan",
    "tl": "tagalog",
    "mg": "malagasy",
    "as": "assamese",
    "tt": "tatar",
    "haw": "hawaiian",
    "ln": "lingala",
    "ha": "hausa",
    "ba": "bashkir",
    "jw": "javanese",
    "su": "sundanese",
}

# Language code lookup by name, with aliases
TO_LANGUAGE_CODE = {
    **{language: code for code, language in LANGUAGES.items()},
    "burmese": "my",
    "valencian": "ca",
    "flemish": "nl",
    "haitian": "ht",
    "letzeburgesch": "lb",
    "pushto": "ps",
    "panjabi": "pa",
    "moldavian": "ro",
    "moldovan": "ro",
    "sinhalese": "si",
    "castilian": "es",
    "mandarin": "zh",
}


@dataclass
class WhisperTokenizer:
    """Whisper tokenizer with special tokens for multilingual ASR.

    Wraps tiktoken encoding and provides quick access to Whisper special tokens:
    - Start/end of transcript tokens
    - Language tokens (99 languages)
    - Task tokens (transcribe/translate)
    - Timestamp tokens (0.00 to 30.00 seconds in 0.02s increments)
    - No-speech token

    The tokenizer constructs a start-of-transcript (SOT) sequence based on:
    - Language (if multilingual)
    - Task (transcribe or translate)
    - Timestamp mode (with or without timestamps)

    Attributes:
        encoding: tiktoken BPE encoding
        num_languages: Number of supported languages (default: 99)
        language: Language code (e.g., "en", "zh")
        task: Task type ("transcribe" or "translate")
        sot_sequence: Start-of-transcript token sequence
        special_tokens: Dictionary of special token names to IDs
    """

    encoding: tiktoken.Encoding
    num_languages: int
    language: Optional[str] = None
    task: Optional[str] = None
    sot_sequence: tuple[int, ...] = ()
    special_tokens: dict[str, int] = field(default_factory=dict)

    def __post_init__(self):
        """Initialize special tokens and construct SOT sequence."""
        # Build special tokens dictionary
        for special in self.encoding.special_tokens_set:
            special_token = self.encoding.encode_single_token(special)
            self.special_tokens[special] = special_token

        # Construct start-of-transcript sequence
        sot: int = self.special_tokens["<|startoftranscript|>"]
        translate: int = self.special_tokens["<|translate|>"]
        transcribe: int = self.special_tokens["<|transcribe|>"]

        langs = tuple(LANGUAGES.keys())[: self.num_languages]
        sot_sequence = [sot]

        # Add language token if specified
        if self.language is not None:
            sot_sequence.append(sot + 1 + langs.index(self.language))

        # Add task token if specified
        if self.task is not None:
            task_token: int = transcribe if self.task == "transcribe" else translate
            sot_sequence.append(task_token)

        self.sot_sequence = tuple(sot_sequence)

    @classmethod
    def from_pretrained(cls, path: str | Path) -> "WhisperTokenizer":
        """Load tokenizer from a directory.

        Args:
            path: Path to directory containing tokenizer config and vocab files

        Returns:
            WhisperTokenizer instance

        Raises:
            FileNotFoundError: If config or vocab files not found

        Example:
            >>> tokenizer = WhisperTokenizer.from_pretrained("./saved_tokenizer")
        """
        path = Path(path)
        config_path = path / "tokenizer_config.json"

        if not config_path.exists():
            raise FileNotFoundError(
                f"Tokenizer config not found at {config_path}. "
                "Make sure the directory contains tokenizer_config.json"
            )

        # Load configuration
        with open(config_path) as f:
            config = json.load(f)

        # Load encoding from saved vocab file
        encoding_name = config.get("encoding_name", "multilingual")
        num_languages = config.get("num_languages", 99)
        vocab_path = path / f"{encoding_name}.tiktoken"

        if not vocab_path.exists():
            raise FileNotFoundError(
                f"Tokenizer vocab not found at {vocab_path}. "
                f"Make sure the directory contains {encoding_name}.tiktoken"
            )

        # Load BPE ranks
        ranks = {}
        with open(vocab_path) as fid:
            for line in fid:
                parts = line.split()
                if len(parts) == 2:
                    token, rank = parts
                    ranks[base64.b64decode(token)] = int(rank)

        n_vocab = len(ranks)
        special_tokens = {}

        # Add special tokens
        specials = [
            "<|endoftext|>",
            "<|startoftranscript|>",
            *[f"<|{lang}|>" for lang in list(LANGUAGES.keys())[:num_languages]],
            "<|translate|>",
            "<|transcribe|>",
            "<|startoflm|>",
            "<|startofprev|>",
            "<|nospeech|>",
            "<|notimestamps|>",
            *[f"<|{i * 0.02:.2f}|>" for i in range(1501)],  # Timestamps 0.00-30.00
        ]

        for token in specials:
            special_tokens[token] = n_vocab
            n_vocab += 1

        # Create tiktoken encoding
        encoding = tiktoken.Encoding(
            name=os.path.basename(vocab_path),
            explicit_n_vocab=n_vocab,
            pat_str=r"""'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""",
            mergeable_ranks=ranks,
            special_tokens=special_tokens,
        )

        # Create tokenizer with loaded config
        return cls(
            encoding=encoding,
            num_languages=num_languages,
            language=config.get("language"),
            task=config.get("task"),
        )

    def save_pretrained(self, path: str | Path) -> None:
        """Save tokenizer to a directory.

        Saves tokenizer configuration and vocab files to enable reloading.

        Args:
            path: Path to directory where tokenizer should be saved

        Example:
            >>> tokenizer = get_tokenizer(multilingual=True, language="en")
            >>> tokenizer.save_pretrained("./saved_tokenizer")
        """
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        # Determine encoding name from the encoding
        encoding_name = "multilingual" if self.num_languages > 1 else "gpt2"

        # Save configuration
        config = {
            "encoding_name": encoding_name,
            "num_languages": self.num_languages,
            "language": self.language,
            "task": self.task,
        }

        config_path = path / "tokenizer_config.json"
        with open(config_path, "w") as f:
            json.dump(config, f, indent=2)

        # Copy vocab file from assets
        assets_dir = Path(__file__).parent / "assets"
        source_vocab = assets_dir / f"{encoding_name}.tiktoken"
        target_vocab = path / f"{encoding_name}.tiktoken"

        if source_vocab.exists():
            shutil.copy2(source_vocab, target_vocab)
        else:
            # If source doesn't exist, we need to reconstruct it from the encoding
            # This shouldn't happen in normal usage, but handle it gracefully
            raise FileNotFoundError(
                f"Could not find vocab file at {source_vocab}. "
                "Cannot save tokenizer without vocab file."
            )

    def encode(self, text: str, **kwargs) -> list[int]:
        """Encode text to token IDs.

        Args:
            text: Text to encode
            **kwargs: Additional arguments for tiktoken encoding

        Returns:
            List of token IDs

        Example:
            >>> tokenizer = get_tokenizer(multilingual=True, language="en")
            >>> tokens = tokenizer.encode("Hello, world!")
            >>> len(tokens)
            4
        """
        return self.encoding.encode(text, **kwargs)

    def decode(self, token_ids: list[int], **kwargs) -> str:
        """Decode token IDs to text, excluding timestamp tokens.

        Timestamp tokens are filtered out before decoding.

        Args:
            token_ids: Token IDs to decode
            **kwargs: Additional arguments for tiktoken decoding

        Returns:
            Decoded text

        Example:
            >>> tokenizer = get_tokenizer(multilingual=True, language="en")
            >>> text = tokenizer.decode([15496, 11, 1002, 0])
            >>> text
            'Hello, world!'
        """
        # Filter out timestamp tokens (they are above timestamp_begin)
        token_ids = [t for t in token_ids if t < self.timestamp_begin]
        return self.encoding.decode(token_ids, **kwargs)

    def decode_with_timestamps(self, token_ids: list[int], **kwargs) -> str:
        """Decode token IDs including timestamp annotations.

        Timestamp tokens are decoded as "<|1.08|>" format.

        Args:
            token_ids: Token IDs to decode
            **kwargs: Additional arguments for tiktoken decoding

        Returns:
            Decoded text with timestamp annotations

        Example:
            >>> tokens = [50364, 15496, 50389]  # <|0.00|> Hello <|0.50|>
            >>> tokenizer.decode_with_timestamps(tokens)
            '<|0.00|>Hello<|0.50|>'
        """
        return self.encoding.decode(token_ids, **kwargs)

    # Special token properties

    @cached_property
    def eot(self) -> int:
        """End-of-transcript token ID."""
        return self.encoding.eot_token

    @cached_property
    def sot(self) -> int:
        """Start-of-transcript token ID."""
        return self.special_tokens["<|startoftranscript|>"]

    @cached_property
    def sot_lm(self) -> int:
        """Start-of-language-model token ID."""
        return self.special_tokens["<|startoflm|>"]

    @cached_property
    def sot_prev(self) -> int:
        """Start-of-previous token ID (for conditioning on previous text)."""
        return self.special_tokens["<|startofprev|>"]

    @cached_property
    def transcribe(self) -> int:
        """Transcribe task token ID."""
        return self.special_tokens["<|transcribe|>"]

    @cached_property
    def translate(self) -> int:
        """Translate task token ID."""
        return self.special_tokens["<|translate|>"]

    @cached_property
    def no_speech(self) -> int:
        """No-speech token ID."""
        return self.special_tokens["<|nospeech|>"]

    @cached_property
    def no_timestamps(self) -> int:
        """No-timestamps token ID."""
        return self.special_tokens["<|notimestamps|>"]

    @cached_property
    def timestamp_begin(self) -> int:
        """First timestamp token ID (<|0.00|>)."""
        return self.special_tokens["<|0.00|>"]

    @cached_property
    def language_token(self) -> int:
        """Get language token ID for configured language.

        Returns:
            Token ID for the language

        Raises:
            ValueError: If no language is configured
        """
        if self.language is None:
            raise ValueError("This tokenizer does not have language token configured")
        return self.to_language_token(self.language)

    def to_language_token(self, language: str) -> int:
        """Get language token ID for any supported language.

        Args:
            language: Language code (e.g., "en", "zh")

        Returns:
            Token ID for the language

        Raises:
            KeyError: If language is not supported
        """
        if token := self.special_tokens.get(f"<|{language}|>", None):
            return token
        raise KeyError(f"Language {language} not found in tokenizer.")

    @cached_property
    def all_language_tokens(self) -> tuple[int, ...]:
        """Get all language token IDs.

        Returns:
            Tuple of language token IDs (up to num_languages)
        """
        result = []
        for token, token_id in self.special_tokens.items():
            if token.strip("<|>") in LANGUAGES:
                result.append(token_id)
        return tuple(result)[: self.num_languages]

    @cached_property
    def all_language_codes(self) -> tuple[str, ...]:
        """Get all language codes.

        Returns:
            Tuple of language codes (e.g., ("en", "zh", "de", ...))
        """
        return tuple(self.decode([_l]).strip("<|>") for _l in self.all_language_tokens)

    @cached_property
    def sot_sequence_including_notimestamps(self) -> tuple[int, ...]:
        """Get SOT sequence with no-timestamps token appended.

        Returns:
            SOT sequence with no-timestamps token
        """
        return tuple(list(self.sot_sequence) + [self.no_timestamps])

    @cached_property
    def non_speech_tokens(self) -> tuple[int, ...]:
        """Get tokens to suppress to avoid non-speech annotations.

        These tokens represent speaker tags, musical notes, and other
        non-speech annotations that should be suppressed during generation:
        - ♪♪♪
        - ( SPEAKING FOREIGN LANGUAGE )
        - [DAVID] Hey there,

        Basic punctuation like commas, periods, etc. are kept.

        Returns:
            Tuple of token IDs to suppress
        """
        symbols = list('"#()*+/:;<=>@[\\]^_`{|}~「」『』')
        symbols += "<< >> <<< >>> -- --- -( -[ (' (\" (( )) ((( ))) [[ ]] {{ }} ♪♪ ♪♪♪".split()

        # Musical symbols (U+2640 to U+267F)
        miscellaneous = set("♩♪♫♬♭♮♯")
        assert all(0x2640 <= ord(c) <= 0x267F for c in miscellaneous)

        # Allow hyphens and quotes between words, but not at the beginning
        result = {self.encoding.encode(" -")[0], self.encoding.encode(" '")[0]}
        for symbol in symbols + list(miscellaneous):
            for tokens in [
                self.encoding.encode(symbol),
                self.encoding.encode(" " + symbol),
            ]:
                if len(tokens) == 1 or symbol in miscellaneous:
                    result.add(tokens[0])

        return tuple(sorted(result))

    def split_to_word_tokens(
        self, tokens: list[int]
    ) -> tuple[list[str], list[list[int]]]:
        """Split tokens into words for word-level timestamp alignment.

        For languages that don't use spaces (Chinese, Japanese, Thai, etc.),
        splits at valid unicode boundaries. For other languages, splits on spaces.

        Args:
            tokens: List of token IDs

        Returns:
            Tuple of (words, word_tokens):
            - words: List of decoded word strings
            - word_tokens: List of token ID lists for each word

        Example:
            >>> tokens = tokenizer.encode("Hello world")
            >>> words, word_tokens = tokenizer.split_to_word_tokens(tokens)
            >>> words
            ['Hello', ' world']
            >>> len(word_tokens[0]), len(word_tokens[1])
            (1, 1)
        """
        if self.language in {"zh", "ja", "th", "lo", "my", "yue"}:
            # Languages without spaces - split on unicode boundaries
            return self.split_tokens_on_unicode(tokens)
        return self.split_tokens_on_spaces(tokens)

    def split_tokens_on_unicode(
        self, tokens: list[int]
    ) -> tuple[list[str], list[list[int]]]:
        """Split tokens at valid unicode boundaries.

        Used for languages that don't use spaces (Chinese, Japanese, etc.).

        Args:
            tokens: List of token IDs

        Returns:
            Tuple of (words, word_tokens)
        """
        decoded_full = self.decode_with_timestamps(tokens)
        replacement_char = "\ufffd"

        words = []
        word_tokens = []
        current_tokens = []
        unicode_offset = 0

        for token in tokens:
            current_tokens.append(token)
            decoded = self.decode_with_timestamps(current_tokens)

            # Check if we have a valid unicode character
            if (
                replacement_char not in decoded
                or decoded_full[unicode_offset + decoded.index(replacement_char)]
                == replacement_char
            ):
                words.append(decoded)
                word_tokens.append(current_tokens)
                current_tokens = []
                unicode_offset += len(decoded)

        return words, word_tokens

    def split_tokens_on_spaces(
        self, tokens: list[int]
    ) -> tuple[list[str], list[list[int]]]:
        """Split tokens on spaces and punctuation.

        Used for languages that use spaces (English, Spanish, etc.).

        Args:
            tokens: List of token IDs

        Returns:
            Tuple of (words, word_tokens)
        """
        subwords, subword_tokens_list = self.split_tokens_on_unicode(tokens)
        words = []
        word_tokens = []

        for subword, subword_tokens in zip(subwords, subword_tokens_list):
            special = subword_tokens[0] >= self.eot
            with_space = subword.startswith(" ")
            punctuation = subword.strip() in string.punctuation

            if special or with_space or punctuation or len(words) == 0:
                # Start new word
                words.append(subword)
                word_tokens.append(subword_tokens)
            else:
                # Append to previous word
                words[-1] = words[-1] + subword
                word_tokens[-1].extend(subword_tokens)

        return words, word_tokens


@cache
def get_encoding(name: str = "gpt2", num_languages: int = 99) -> tiktoken.Encoding:
    """Load tiktoken encoding from assets.

    Args:
        name: Encoding name ("gpt2" for English-only, "multilingual" for 99 languages)
        num_languages: Number of languages to support (default: 99)

    Returns:
        tiktoken.Encoding instance

    Raises:
        FileNotFoundError: If vocab file not found in assets
    """
    # Get assets directory
    assets_dir = Path(__file__).parent / "assets"
    vocab_path = assets_dir / f"{name}.tiktoken"

    if not vocab_path.exists():
        raise FileNotFoundError(
            f"Tokenizer vocab not found at {vocab_path}. "
            "Please ensure the assets directory is properly set up."
        )

    # Load BPE ranks
    ranks = {}
    with open(vocab_path) as fid:
        for line in fid:
            parts = line.split()
            if len(parts) == 2:
                token, rank = parts
                ranks[base64.b64decode(token)] = int(rank)

    n_vocab = len(ranks)
    special_tokens = {}

    # Add special tokens
    specials = [
        "<|endoftext|>",
        "<|startoftranscript|>",
        *[f"<|{lang}|>" for lang in list(LANGUAGES.keys())[:num_languages]],
        "<|translate|>",
        "<|transcribe|>",
        "<|startoflm|>",
        "<|startofprev|>",
        "<|nospeech|>",
        "<|notimestamps|>",
        *[f"<|{i * 0.02:.2f}|>" for i in range(1501)],  # Timestamps 0.00-30.00
    ]

    for token in specials:
        special_tokens[token] = n_vocab
        n_vocab += 1

    # Create tiktoken encoding
    return tiktoken.Encoding(
        name=os.path.basename(vocab_path),
        explicit_n_vocab=n_vocab,
        pat_str=r"""'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""",
        mergeable_ranks=ranks,
        special_tokens=special_tokens,
    )


@cache
def get_tokenizer(
    multilingual: bool,
    *,
    num_languages: int = 99,
    language: Optional[str] = None,
    task: Optional[str] = None,
) -> WhisperTokenizer:
    """Get Whisper tokenizer.

    Args:
        multilingual: Whether to use multilingual tokenizer (99 languages)
                     or English-only tokenizer
        num_languages: Number of languages to support (default: 99)
        language: Language code (e.g., "en", "zh"). Required for multilingual.
        task: Task type ("transcribe" or "translate"). Default: "transcribe"

    Returns:
        WhisperTokenizer instance

    Raises:
        ValueError: If unsupported language is specified

    Example:
        >>> # English-only tokenizer
        >>> tokenizer = get_tokenizer(multilingual=False)

        >>> # Multilingual tokenizer for Spanish transcription
        >>> tokenizer = get_tokenizer(
        ...     multilingual=True,
        ...     language="es",
        ...     task="transcribe"
        ... )

        >>> # Multilingual tokenizer for Chinese → English translation
        >>> tokenizer = get_tokenizer(
        ...     multilingual=True,
        ...     language="zh",
        ...     task="translate"
        ... )
    """
    # Validate and normalize language
    if language is not None:
        language = language.lower()
        if language not in LANGUAGES:
            if language in TO_LANGUAGE_CODE:
                language = TO_LANGUAGE_CODE[language]
            else:
                raise ValueError(f"Unsupported language: {language}")

    # Configure tokenizer based on multilingual flag
    if multilingual:
        encoding_name = "multilingual"
        language = language or "en"
        task = task or "transcribe"
    else:
        encoding_name = "gpt2"
        language = None
        task = None

    # Load encoding and create tokenizer
    encoding = get_encoding(name=encoding_name, num_languages=num_languages)

    return WhisperTokenizer(
        encoding=encoding, num_languages=num_languages, language=language, task=task
    )
