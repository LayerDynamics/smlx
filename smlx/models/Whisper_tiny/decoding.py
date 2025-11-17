"""
Whisper decoding logic with KV caching and logit filtering.

Implements greedy decoding, temperature sampling, and various logit filters
for timestamp handling and token suppression.

Based on OpenAI's Whisper decoding implementation.
"""

import zlib
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field, replace
from typing import Optional, Union

import mlx.core as mx
import numpy as np
from mlx.utils import tree_map

from .audio import CHUNK_LENGTH
from .model import Whisper
from .tokenizer import WhisperTokenizer, get_tokenizer


def compression_ratio(text: str) -> float:
    """Calculate compression ratio of text.

    Compression ratio is used as a quality metric. Low compression ratio
    indicates repetitive or low-quality text.

    Args:
        text: Text to measure

    Returns:
        Ratio of uncompressed to compressed size

    Example:
        >>> compression_ratio("Hello world")
        1.2
        >>> compression_ratio("a" * 100)  # Highly compressible/repetitive
        10.5
    """
    text_bytes = text.encode("utf-8")
    return len(text_bytes) / len(zlib.compress(text_bytes))


def detect_language(
    model: Whisper, mel: mx.array, tokenizer: WhisperTokenizer = None
) -> tuple[mx.array | list[mx.array], dict[str, float] | list[dict[str, float]]]:
    """Detect spoken language in audio.

    Uses encoder output and a single forward pass with SOT token to
    detect language probabilities.

    Args:
        model: Whisper model
        mel: Mel spectrogram of shape (n_mels, n_frames) or (batch, n_mels, n_frames)
        tokenizer: Tokenizer (created if not provided)

    Returns:
        Tuple of (language_tokens, language_probs):
        - language_tokens: Most probable language token IDs (single array or list)
        - language_probs: Dict or list of dicts with probability for each language

    Raises:
        ValueError: If model doesn't support language detection

    Example:
        >>> mel = prepare_audio("speech.wav")
        >>> lang_tokens, lang_probs = detect_language(model, mel)
        >>> max(lang_probs, key=lang_probs.get)  # single input returns dict
        'en'
    """
    if tokenizer is None:
        tokenizer = get_tokenizer(
            model.is_multilingual, num_languages=model.num_languages
        )

    if (
        tokenizer.language is None
        or tokenizer.language_token not in tokenizer.sot_sequence
    ):
        raise ValueError(
            "This model doesn't have language tokens so it can't perform language detection"
        )

    single = mel.ndim == 2
    if single:
        mel = mel[None]

    # Encode audio if needed
    if mel.shape[-2:] != (model.config.n_audio_ctx, model.config.n_audio_state):
        mel = model.encoder(mel)

    # Forward pass with SOT token only
    n_audio = mel.shape[0]
    x = mx.array([[tokenizer.sot]] * n_audio)  # [n_audio, 1]
    logits, _, _ = model.decoder(x, mel)
    logits = logits[:, 0]

    # Mask all non-language tokens
    mask = np.full(logits.shape[-1], -np.inf, dtype=np.float32)
    mask[list(tokenizer.all_language_tokens)] = 0.0
    logits += mx.array(mask)

    # Get language predictions
    language_tokens = mx.argmax(logits, axis=-1)
    language_token_probs = mx.softmax(logits, axis=-1)

    language_probs = [
        {
            code: language_token_probs[i, token_id].item()
            for token_id, code in zip(
                tokenizer.all_language_tokens, tokenizer.all_language_codes
            )
        }
        for i in range(n_audio)
    ]

    if single:
        language_tokens = language_tokens[0]
        language_probs = language_probs[0]

    return language_tokens, language_probs


@dataclass(frozen=True)
class DecodingOptions:
    """Options for Whisper decoding.

    Controls language, task, sampling strategy, and various filtering options.

    Attributes:
        task: "transcribe" (X->X) or "translate" (X->English)
        language: Audio language code (e.g., "en"). None = auto-detect.
        temperature: Sampling temperature (0 = greedy)
        sample_len: Maximum tokens to generate (default: n_text_ctx // 2)
        best_of: Number of samples to generate and rank (if temperature > 0)
        beam_size: Beam search width (1 = greedy, >1 = beam search)
        patience: Beam search patience parameter for early stopping
        length_penalty: Length penalty for ranking (None = simple length norm)
        prompt: Previous context to condition on
        prefix: Text to prefix the current transcription
        suppress_tokens: Token IDs to suppress ("-1" = non-speech tokens)
        suppress_blank: Suppress blank outputs at start
        without_timestamps: Disable timestamp generation
        max_initial_timestamp: Maximum timestamp for first token (seconds)
        fp16: Use float16 for most calculations
    """

    task: str = "transcribe"
    language: Optional[str] = None

    # Sampling options
    temperature: float = 0.0
    sample_len: Optional[int] = None
    best_of: Optional[int] = None
    beam_size: Optional[int] = None
    patience: Optional[float] = None
    length_penalty: Optional[float] = None

    # Conditioning options
    prompt: Optional[Union[str, list[int]]] = None
    prefix: Optional[Union[str, list[int]]] = None

    # Suppression options
    suppress_tokens: Optional[Union[str, Iterable[int]]] = "-1"
    suppress_blank: bool = True

    # Timestamp options
    without_timestamps: bool = False
    max_initial_timestamp: Optional[float] = 1.0

    # Implementation options
    fp16: bool = True


@dataclass(frozen=True)
class DecodingResult:
    """Result of Whisper decoding.

    Contains decoded text, tokens, and various quality metrics.

    Attributes:
        audio_features: Encoded audio features from encoder
        language: Detected or specified language code
        language_probs: Probability distribution over languages (if detected)
        tokens: Generated token IDs
        text: Decoded text
        avg_logprob: Average log probability of tokens
        no_speech_prob: Probability of no speech in audio
        temperature: Temperature used for sampling
        compression_ratio: Text compression ratio (quality metric)
    """

    audio_features: mx.array
    language: str
    language_probs: Optional[dict[str, float]] = None
    tokens: list[int] = field(default_factory=list)
    text: str = ""
    avg_logprob: float = np.nan
    no_speech_prob: float = np.nan
    temperature: float = np.nan
    compression_ratio: float = np.nan


class Inference:
    """Forward pass through decoder with KV caching.

    Maintains KV cache across decoding steps for efficient autoregressive generation.

    Attributes:
        model: Whisper model
        initial_token_length: Length of initial token sequence (SOT + prefix)
        kv_cache: Cached key/value tensors for each decoder layer
    """

    def __init__(self, model: Whisper, initial_token_length: int):
        """Initialize inference.

        Args:
            model: Whisper model
            initial_token_length: Length of initial token sequence
        """
        self.model = model
        self.initial_token_length = initial_token_length
        self.kv_cache = None

    def logits(self, tokens: mx.array, audio_features: mx.array) -> mx.array:
        """Compute logits for next token.

        Uses KV cache to avoid recomputing past tokens.

        Args:
            tokens: Current token sequence of shape (batch, seq_len)
            audio_features: Encoded audio of shape (batch, audio_ctx, n_state)

        Returns:
            Logits of shape (batch, seq_len, vocab_size)
        """
        if tokens.shape[-1] > self.initial_token_length:
            # Only need last token except in first forward pass
            tokens = tokens[:, -1:]

        logits, self.kv_cache, _ = self.model.decoder(
            tokens, audio_features, kv_cache=self.kv_cache
        )
        return logits.astype(mx.float32)

    def rearrange_kv_cache(self, source_indices: list[int]):
        """Update KV cache for beam search reordering.

        Args:
            source_indices: Indices to select from cache
        """
        if source_indices != list(range(len(source_indices))):
            self.kv_cache = tree_map(lambda x: x[source_indices], self.kv_cache)

    def reset(self):
        """Reset KV cache for new sequence."""
        self.kv_cache = None


class SequenceRanker:
    """Base class for ranking sampled sequences."""

    def rank(
        self, tokens: list[list[list[int]]], sum_logprobs: list[list[float]]
    ) -> list[int]:
        """Rank sequences and return best indices.

        Args:
            tokens: List of token sequences for each audio
            sum_logprobs: Cumulative log probabilities

        Returns:
            List of indices of best sequence in each group
        """
        raise NotImplementedError


class MaximumLikelihoodRanker(SequenceRanker):
    """Rank sequences by log probability with length penalty.

    Uses either simple length normalization or Google NMT-style length penalty.
    """

    def __init__(self, length_penalty: Optional[float]):
        """Initialize ranker.

        Args:
            length_penalty: Length penalty parameter (None = simple normalization)
        """
        self.length_penalty = length_penalty

    def rank(
        self, tokens: list[list[list[int]]], sum_logprobs: list[list[float]]
    ) -> list[int]:
        """Rank sequences by penalized log probability.

        Args:
            tokens: List of token sequences
            sum_logprobs: Cumulative log probabilities

        Returns:
            List of best sequence indices
        """

        def scores(logprobs, lengths):
            result = []
            for logprob, length in zip(logprobs, lengths):
                if self.length_penalty is None:
                    penalty = length
                else:
                    # Google NMT length penalty
                    penalty = ((5 + length) / 6) ** self.length_penalty
                result.append(logprob / penalty)
            return result

        lengths = [[len(t) for t in s] for s in tokens]
        return [np.argmax(scores(p, lens)) for p, lens in zip(sum_logprobs, lengths)]


class TokenDecoder:
    """Base class for token selection strategies."""

    def reset(self):
        """Initialize stateful variables for new sequence."""
        pass

    def update(
        self, tokens: mx.array, logits: mx.array, sum_logprobs: mx.array
    ) -> tuple[mx.array, bool, mx.array]:
        """Select next token based on logits.

        Args:
            tokens: Current tokens of shape (n_batch, seq_len)
            logits: Logits of shape (n_batch, vocab_size)
            sum_logprobs: Cumulative log probs of shape (n_batch)

        Returns:
            Tuple of (updated_tokens, completed, updated_sum_logprobs)
        """
        raise NotImplementedError

    def finalize(
        self, tokens: mx.array, sum_logprobs: mx.array
    ) -> tuple[Sequence[Sequence[mx.array]], list[list[float]]]:
        """Finalize and return candidate sequences.

        Args:
            tokens: All tokens of shape (n_audio, n_group, seq_len)
            sum_logprobs: Log probs of shape (n_audio, n_group)

        Returns:
            Tuple of (tokens, sum_logprobs)
        """
        raise NotImplementedError


class GreedyDecoder(TokenDecoder):
    """Greedy or temperature-based sampling decoder.

    Selects highest probability token (greedy) or samples from distribution
    (temperature > 0).
    """

    def __init__(self, temperature: float, eot: int):
        """Initialize decoder.

        Args:
            temperature: Sampling temperature (0 = greedy)
            eot: End-of-transcript token ID
        """
        self.temperature = temperature
        self.eot = eot

    def update(
        self, tokens: mx.array, logits: mx.array, sum_logprobs: mx.array
    ) -> tuple[mx.array, bool, mx.array]:
        """Select next token.

        Args:
            tokens: Current tokens
            logits: Next token logits
            sum_logprobs: Cumulative log probabilities

        Returns:
            Tuple of (updated_tokens, all_completed, updated_sum_logprobs)
        """
        if self.temperature == 0:
            next_tokens = mx.argmax(logits, axis=-1)
        else:
            next_tokens = mx.random.categorical(logits=logits / self.temperature)

        # Compute log probabilities
        logits = logits.astype(mx.float32)
        logprobs = logits - mx.logsumexp(logits, axis=-1, keepdims=True)

        # Update cumulative log probs (don't update for sequences that ended)
        current_logprobs = logprobs[mx.arange(logprobs.shape[0]), next_tokens]
        sum_logprobs += current_logprobs * (tokens[:, -1] != self.eot)

        # Keep EOT for sequences that already ended
        eot_mask = tokens[:, -1] == self.eot
        next_tokens = next_tokens * (1 - eot_mask) + self.eot * eot_mask
        tokens = mx.concatenate([tokens, next_tokens[:, None]], axis=-1)

        # Check if all sequences completed
        completed = mx.all(tokens[:, -1] == self.eot)
        return tokens, completed, sum_logprobs

    def finalize(
        self, tokens: mx.array, sum_logprobs: mx.array
    ) -> tuple[mx.array, list[list[float]]]:
        """Finalize sequences.

        Ensures each sequence has at least one EOT token.

        Args:
            tokens: All tokens
            sum_logprobs: Log probabilities

        Returns:
            Tuple of (tokens, sum_logprobs)
        """
        # Ensure EOT at end
        tokens = mx.pad(tokens, [(0, 0), (0, 0), (0, 1)], constant_values=self.eot)
        return tokens, sum_logprobs.tolist()


class BeamSearchDecoder(TokenDecoder):
    """Beam search decoder that maintains multiple hypotheses.

    Explores multiple candidate sequences (beams) in parallel and selects
    the best one based on cumulative log probabilities. Generally produces
    higher quality results than greedy decoding.
    """

    def __init__(
        self,
        beam_size: int,
        eot: int,
        patience: Optional[float] = None,
    ):
        """Initialize beam search decoder.

        Args:
            beam_size: Number of beams to maintain
            eot: End-of-transcript token ID
            patience: Beam search patience factor (early stopping when best
                     incomplete beam score falls below best complete beam score
                     by this factor). If None, no early stopping.
        """
        self.beam_size = beam_size
        self.eot = eot
        self.patience = patience
        self.max_candidates = max(1, int(beam_size * (patience or 1.0)))
        self.finished_sequences: list[tuple[mx.array, float]] = []

    def reset(self):
        """Reset state for new sequence."""
        self.finished_sequences = []

    def update(
        self, tokens: mx.array, logits: mx.array, sum_logprobs: mx.array
    ) -> tuple[mx.array, bool, mx.array]:
        """Update beams with next token.

        Args:
            tokens: Current tokens of shape (n_batch * n_beam, seq_len)
            logits: Next token logits of shape (n_batch * n_beam, vocab_size)
            sum_logprobs: Cumulative log probs of shape (n_batch * n_beam)

        Returns:
            Tuple of (updated_tokens, all_completed, updated_sum_logprobs)
        """
        n_batch_beam = tokens.shape[0]
        n_batch = n_batch_beam // self.beam_size

        # Compute log probabilities
        logits = logits.astype(mx.float32)
        logprobs = logits - mx.logsumexp(logits, axis=-1, keepdims=True)

        # For each batch, we maintain beam_size beams
        # At each step, we consider beam_size * vocab_size candidates
        # and select top beam_size beams

        new_tokens_list = []
        new_sum_logprobs_list = []

        for batch_idx in range(n_batch):
            # Get beams for this batch
            batch_start = batch_idx * self.beam_size
            batch_end = batch_start + self.beam_size

            batch_tokens = tokens[batch_start:batch_end]
            batch_logprobs = logprobs[batch_start:batch_end]
            batch_sum_logprobs = sum_logprobs[batch_start:batch_end]

            # Check which beams already ended
            finished_mask = batch_tokens[:, -1] == self.eot

            # For finished beams, only allow EOT token
            # For active beams, consider all vocabulary
            vocab_size = batch_logprobs.shape[-1]

            # Compute scores for all candidates: (beam_size, vocab_size)
            # scores[i, j] = log_prob of beam i with next token j
            candidate_scores = batch_sum_logprobs[:, None] + batch_logprobs

            # Mask out finished beams (only allow EOT)
            for beam_idx in range(self.beam_size):
                if finished_mask[beam_idx]:
                    candidate_scores[beam_idx, :] = -np.inf
                    candidate_scores[beam_idx, self.eot] = batch_sum_logprobs[beam_idx]

            # Flatten to (beam_size * vocab_size) and select top beam_size
            candidate_scores_flat = candidate_scores.reshape(-1)
            top_indices = mx.argsort(-candidate_scores_flat)[:self.beam_size]

            # Convert flat indices to (beam_idx, token_idx)
            beam_indices = top_indices // vocab_size
            token_indices = top_indices % vocab_size

            # Update tokens and scores
            new_batch_tokens = []
            new_batch_scores = []

            for beam_idx, token_idx in zip(beam_indices, token_indices):
                old_tokens = batch_tokens[beam_idx]
                new_token = token_idx
                new_tokens = mx.concatenate([old_tokens, new_token[None]])
                new_score = candidate_scores[beam_idx, token_idx]

                new_batch_tokens.append(new_tokens)
                new_batch_scores.append(new_score)

            new_tokens_list.extend(new_batch_tokens)
            new_sum_logprobs_list.extend(new_batch_scores)

        # Stack all beams
        # Pad to maximum length
        max_len = max(t.shape[0] for t in new_tokens_list)
        padded_tokens = []
        for t in new_tokens_list:
            if t.shape[0] < max_len:
                padding = mx.full((max_len - t.shape[0],), self.eot, dtype=t.dtype)
                t = mx.concatenate([t, padding])
            padded_tokens.append(t)

        tokens = mx.stack(padded_tokens, axis=0)
        sum_logprobs = mx.array(new_sum_logprobs_list)

        # Check if all beams completed
        completed = mx.all(tokens[:, -1] == self.eot)

        return tokens, completed, sum_logprobs

    def finalize(
        self, tokens: mx.array, sum_logprobs: mx.array
    ) -> tuple[Sequence[Sequence[mx.array]], list[list[float]]]:
        """Select best beam from each batch.

        Args:
            tokens: All tokens of shape (n_batch * n_beam, seq_len)
            sum_logprobs: Log probs of shape (n_batch * n_beam)

        Returns:
            Tuple of (tokens, sum_logprobs) where each is a list of sequences
        """
        n_batch_beam = tokens.shape[0]
        n_batch = n_batch_beam // self.beam_size

        # For each batch, select beam with highest score
        batch_tokens = []
        batch_logprobs = []

        for batch_idx in range(n_batch):
            batch_start = batch_idx * self.beam_size
            batch_end = batch_start + self.beam_size

            beams = tokens[batch_start:batch_end]
            scores = sum_logprobs[batch_start:batch_end]

            # Find lengths (exclude padding)
            lengths = []
            for beam in beams:
                # Find first EOT position
                eot_positions = mx.where(beam == self.eot)[0]
                if len(eot_positions) > 0:
                    length = int(eot_positions[0]) + 1
                else:
                    length = len(beam)
                lengths.append(length)

            # Normalize by length (average log prob)
            normalized_scores = []
            for score, length in zip(scores, lengths):
                normalized_scores.append(float(score) / length)

            # Select best beam
            best_beam_idx = int(mx.argmax(mx.array(normalized_scores)))
            best_beam = beams[best_beam_idx]
            best_score = float(scores[best_beam_idx])

            # Trim to EOT
            best_length = lengths[best_beam_idx]
            best_beam = best_beam[:best_length]

            batch_tokens.append([best_beam])
            batch_logprobs.append([best_score])

        return batch_tokens, batch_logprobs


class LogitFilter:
    """Base class for logit filters."""

    def apply(self, logits: mx.array, tokens: mx.array) -> mx.array:
        """Apply filter to logits.

        Args:
            logits: Logits of shape (n_batch, vocab_size)
            tokens: Current tokens of shape (n_batch, seq_len)

        Returns:
            Filtered logits
        """
        raise NotImplementedError


class SuppressBlank(LogitFilter):
    """Suppress blank tokens at start of generation.

    Prevents generating space or EOT as first token.
    """

    def __init__(self, tokenizer: WhisperTokenizer, sample_begin: int, n_vocab: int):
        """Initialize filter.

        Args:
            tokenizer: Tokenizer
            sample_begin: Index where sampling begins
            n_vocab: Vocabulary size
        """
        self.sample_begin = sample_begin
        mask = np.zeros(n_vocab, np.float32)
        mask[tokenizer.encode(" ") + [tokenizer.eot]] = -np.inf
        self.mask = mx.array(mask)

    def apply(self, logits: mx.array, tokens: mx.array) -> mx.array:
        """Apply suppression at start of generation.

        Args:
            logits: Current logits
            tokens: Current tokens

        Returns:
            Filtered logits
        """
        if tokens.shape[1] == self.sample_begin:
            return logits + self.mask
        return logits


class SuppressTokens(LogitFilter):
    """Suppress specified tokens.

    Used to suppress non-speech tokens, task tokens, etc.
    """

    def __init__(self, suppress_tokens: Sequence[int], n_vocab: int):
        """Initialize filter.

        Args:
            suppress_tokens: Token IDs to suppress
            n_vocab: Vocabulary size
        """
        mask = np.zeros(n_vocab, np.float32)
        mask[list(suppress_tokens)] = -np.inf
        self.mask = mx.array(mask)

    def apply(self, logits: mx.array, tokens: mx.array) -> mx.array:
        """Apply token suppression.

        Args:
            logits: Current logits
            tokens: Current tokens

        Returns:
            Filtered logits
        """
        return logits + self.mask


class ApplyTimestampRules(LogitFilter):
    """Apply timestamp token rules.

    Enforces:
    - Timestamps appear in pairs (text between two timestamps)
    - Timestamps are monotonically increasing
    - First token is a timestamp if max_initial_timestamp is set
    - Timestamp selected if probability mass exceeds text tokens
    """

    def __init__(
        self,
        tokenizer: WhisperTokenizer,
        sample_begin: int,
        max_initial_timestamp_index: Optional[int],
    ):
        """Initialize filter.

        Args:
            tokenizer: Tokenizer
            sample_begin: Index where sampling begins
            max_initial_timestamp_index: Maximum initial timestamp index
        """
        self.tokenizer = tokenizer
        self.sample_begin = sample_begin
        self.max_initial_timestamp_index = max_initial_timestamp_index

    def apply(self, logits: mx.array, tokens: mx.array) -> mx.array:
        """Apply timestamp rules.

        Args:
            logits: Current logits
            tokens: Current tokens

        Returns:
            Filtered logits
        """
        mask = np.zeros(logits.shape, np.float32)

        # Suppress <|notimestamps|> (handled by without_timestamps option)
        if self.tokenizer.no_timestamps is not None:
            mask[:, self.tokenizer.no_timestamps] = -np.inf

        # Enforce timestamp pairing and monotonicity
        for k in range(tokens.shape[0]):
            sampled_tokens = tokens[k, self.sample_begin :]
            seq = sampled_tokens.tolist()
            last_was_timestamp = (
                len(seq) >= 1 and seq[-1] >= self.tokenizer.timestamp_begin
            )
            penultimate_was_timestamp = (
                len(seq) < 2 or seq[-2] >= self.tokenizer.timestamp_begin
            )

            if last_was_timestamp:
                if penultimate_was_timestamp:
                    # Has to be non-timestamp (text token)
                    mask[k, self.tokenizer.timestamp_begin :] = -np.inf
                else:
                    # Cannot be normal text tokens
                    mask[k, : self.tokenizer.eot] = -np.inf

            # Enforce monotonicity and nonzero segment length
            timestamps = [
                i for i, v in enumerate(seq) if v >= self.tokenizer.timestamp_begin
            ]
            if len(timestamps) > 0:
                last_timestamp = seq[timestamps[-1]]
                if not last_was_timestamp or penultimate_was_timestamp:
                    last_timestamp += 1
                mask[k, self.tokenizer.timestamp_begin : last_timestamp] = -np.inf

        # At start of generation
        if tokens.shape[1] == self.sample_begin:
            # Suppress non-timestamp tokens at beginning
            mask[:, : self.tokenizer.timestamp_begin] = -np.inf

            # Apply max_initial_timestamp constraint
            if self.max_initial_timestamp_index is not None:
                last_allowed = (
                    self.tokenizer.timestamp_begin + self.max_initial_timestamp_index
                )
                mask[:, last_allowed + 1 :] = -np.inf

        # If timestamp probability mass > text probability mass, force timestamp
        logprobs = logits - mx.logsumexp(logits, axis=-1, keepdims=True)
        for k in range(tokens.shape[0]):
            timestamp_logprob = logprobs[k, self.tokenizer.timestamp_begin :].logsumexp(
                axis=-1
            )
            max_text_token_logprob = logprobs[k, : self.tokenizer.timestamp_begin].max()
            if timestamp_logprob > max_text_token_logprob:
                mask[k, : self.tokenizer.timestamp_begin] = -np.inf

        return logits + mx.array(mask, logits.dtype)


class DecodingTask:
    """Main decoding orchestrator.

    Coordinates encoder, decoder, tokenizer, and various filters to
    perform complete Whisper decoding.

    Attributes:
        model: Whisper model
        options: Decoding options
        tokenizer: Tokenizer
        inference: Inference engine with KV caching
        sequence_ranker: Sequence ranking strategy
        decoder: Token decoder strategy
        logit_filters: List of logit filters to apply
    """

    def __init__(self, model: Whisper, options: DecodingOptions):
        """Initialize decoding task.

        Args:
            model: Whisper model
            options: Decoding options
        """
        self.model = model
        self.options = self._verify_options(options)

        # Initialize tokenizer
        language = options.language or "en"
        self.tokenizer = get_tokenizer(
            model.is_multilingual,
            num_languages=model.num_languages,
            language=language,
            task=options.task,
        )

        # Set up decoding parameters
        self.n_group = options.beam_size or options.best_of or 1
        self.n_ctx = model.config.n_text_ctx
        self.sample_len = options.sample_len or model.config.n_text_ctx // 2

        # Construct SOT sequence
        self.sot_sequence = self.tokenizer.sot_sequence
        if self.options.without_timestamps:
            self.sot_sequence = self.tokenizer.sot_sequence_including_notimestamps

        self.initial_tokens = self._get_initial_tokens()
        self.sample_begin = len(self.initial_tokens)
        self.sot_index = self.initial_tokens.index(self.tokenizer.sot)

        # Initialize components
        self.inference = Inference(model, len(self.initial_tokens))
        self.sequence_ranker = MaximumLikelihoodRanker(options.length_penalty)

        # Initialize decoder
        if options.beam_size is not None and options.beam_size > 1:
            self.decoder = BeamSearchDecoder(
                options.beam_size, self.tokenizer.eot, options.patience
            )
        else:
            self.decoder = GreedyDecoder(options.temperature, self.tokenizer.eot)

        # Initialize logit filters
        self.logit_filters = []
        if self.options.suppress_blank:
            self.logit_filters.append(
                SuppressBlank(
                    self.tokenizer, self.sample_begin, model.config.n_vocab
                )
            )
        if self.options.suppress_tokens:
            self.logit_filters.append(
                SuppressTokens(
                    self._get_suppress_tokens(), model.config.n_vocab
                )
            )
        if not options.without_timestamps:
            precision = CHUNK_LENGTH / model.config.n_audio_ctx  # 0.02 seconds
            max_initial_timestamp_index = None
            if options.max_initial_timestamp:
                max_initial_timestamp_index = round(
                    self.options.max_initial_timestamp / precision
                )
            self.logit_filters.append(
                ApplyTimestampRules(
                    self.tokenizer, self.sample_begin, max_initial_timestamp_index
                )
            )

    def _verify_options(self, options: DecodingOptions) -> DecodingOptions:
        """Verify decoding options are valid.

        Args:
            options: Decoding options

        Returns:
            Validated options

        Raises:
            ValueError: If options are invalid
        """
        if options.beam_size is not None and options.best_of is not None:
            raise ValueError("beam_size and best_of can't be given together")
        if options.temperature == 0:
            if options.best_of is not None:
                raise ValueError(
                    "best_of with greedy sampling (temperature=0) is not compatible"
                )
        if options.patience is not None and options.beam_size is None:
            raise ValueError("patience requires beam_size to be given")
        if options.length_penalty is not None and not (
            0 <= options.length_penalty <= 1
        ):
            raise ValueError(
                "length_penalty (alpha) should be a value between 0 and 1"
            )
        return options

    def _get_initial_tokens(self) -> tuple[int, ...]:
        """Construct initial token sequence.

        Includes SOT sequence, optional prefix, and optional prompt.

        Returns:
            Tuple of initial token IDs
        """
        tokens = list(self.sot_sequence)

        # Add prefix if specified
        if prefix := self.options.prefix:
            prefix_tokens = (
                self.tokenizer.encode(" " + prefix.strip())
                if isinstance(prefix, str)
                else prefix
            )
            if self.sample_len is not None:
                max_prefix_len = self.n_ctx // 2 - self.sample_len
                prefix_tokens = prefix_tokens[-max_prefix_len:]
            tokens = tokens + prefix_tokens

        # Add prompt if specified
        if prompt := self.options.prompt:
            prompt_tokens = (
                self.tokenizer.encode(" " + prompt.strip())
                if isinstance(prompt, str)
                else prompt
            )
            tokens = (
                [self.tokenizer.sot_prev]
                + prompt_tokens[-(self.n_ctx // 2 - 1) :]
                + tokens
            )

        return tuple(tokens)

    def _get_suppress_tokens(self) -> tuple[int, ...]:
        """Get tokens to suppress.

        Returns:
            Tuple of token IDs to suppress
        """
        suppress_tokens = self.options.suppress_tokens

        if isinstance(suppress_tokens, str):
            suppress_tokens = [int(t) for t in suppress_tokens.split(",")]

        if -1 in suppress_tokens:
            # -1 means suppress all non-speech tokens
            suppress_tokens = [t for t in suppress_tokens if t >= 0]
            suppress_tokens.extend(self.tokenizer.non_speech_tokens)
        elif suppress_tokens is None or len(suppress_tokens) == 0:
            suppress_tokens = []
        else:
            assert isinstance(suppress_tokens, list), "suppress_tokens must be a list"

        # Always suppress task/control tokens
        suppress_tokens.extend(
            [
                self.tokenizer.transcribe,
                self.tokenizer.translate,
                self.tokenizer.sot,
                self.tokenizer.sot_prev,
                self.tokenizer.sot_lm,
            ]
        )
        if self.tokenizer.no_speech is not None:
            suppress_tokens.append(self.tokenizer.no_speech)

        return tuple(sorted(set(suppress_tokens)))

    def _get_audio_features(self, mel: mx.array) -> mx.array:
        """Encode audio to features.

        Args:
            mel: Mel spectrogram or pre-encoded features

        Returns:
            Audio features from encoder
        """
        if self.options.fp16:
            mel = mel.astype(mx.float16)

        if mel.shape[-2:] == (
            self.model.config.n_audio_ctx,
            self.model.config.n_audio_state,
        ):
            # Already encoded
            audio_features = mel
        else:
            audio_features = self.model.encoder(mel)

        expected_dtype = mx.float16 if self.options.fp16 else mx.float32
        if audio_features.dtype != expected_dtype:
            raise TypeError(
                f"audio_features has incorrect dtype: {audio_features.dtype}"
            )

        return audio_features

    def _detect_language(
        self, audio_features: mx.array, tokens: np.ndarray
    ) -> tuple[list[str], Optional[list[dict[str, float]]]]:
        """Detect language for audio.

        Args:
            audio_features: Encoded audio features
            tokens: Initial tokens (will be updated with language tokens)

        Returns:
            Tuple of (languages, language_probs)
        """
        languages = [self.options.language] * audio_features.shape[0]
        lang_probs = None

        if self.options.language is None or self.options.task == "lang_id":
            lang_tokens, lang_probs = detect_language(
                self.model, audio_features, self.tokenizer
            )
            if not isinstance(lang_probs, list):
                lang_probs = [lang_probs]
            languages = [max(probs, key=probs.get) for probs in lang_probs]

            if self.options.language is None:
                # Write language tokens to initial sequence
                tokens[:, self.sot_index + 1] = np.array(
                    lang_tokens if isinstance(lang_tokens, list) else [lang_tokens]
                )

        return languages, lang_probs

    def _main_loop(
        self, audio_features: mx.array, tokens: mx.array
    ) -> tuple[mx.array, mx.array, list[float]]:
        """Main decoding loop.

        Args:
            audio_features: Encoded audio
            tokens: Initial tokens

        Returns:
            Tuple of (tokens, sum_logprobs, no_speech_probs)
        """
        n_batch = tokens.shape[0]
        sum_logprobs = mx.zeros(n_batch)
        no_speech_probs = [np.nan] * n_batch

        try:
            for i in range(self.sample_len):
                logits = self.inference.logits(tokens, audio_features)

                # Save no-speech probability at first step
                if i == 0 and self.tokenizer.no_speech is not None:
                    probs_at_sot = mx.softmax(
                        logits[:, self.sot_index].astype(mx.float32), axis=-1
                    )
                    no_speech_probs = probs_at_sot[:, self.tokenizer.no_speech].tolist()

                # Get logits for last token only
                logits = logits[:, -1]

                # Apply logit filters
                for logit_filter in self.logit_filters:
                    logits = logit_filter.apply(logits, tokens)

                # Select next token
                tokens, completed, sum_logprobs = self.decoder.update(
                    tokens, logits, sum_logprobs
                )

                if completed or tokens.shape[-1] > self.n_ctx:
                    break
        finally:
            self.inference.reset()

        return tokens, sum_logprobs, no_speech_probs

    def run(self, mel: mx.array) -> list[DecodingResult]:
        """Run complete decoding.

        Args:
            mel: Mel spectrogram(s)

        Returns:
            List of DecodingResult
        """
        self.decoder.reset()
        n_audio = mel.shape[0]

        # Encode audio
        audio_features = self._get_audio_features(mel)

        # Initialize tokens
        tokens = np.array(self.initial_tokens)
        tokens = np.broadcast_to(tokens, (n_audio, len(self.initial_tokens))).copy()

        # Detect language if needed
        languages, language_probs = self._detect_language(audio_features, tokens)

        # Handle language ID task
        if self.options.task == "lang_id":
            return [
                DecodingResult(
                    audio_features=features, language=language, language_probs=probs
                )
                for features, language, probs in zip(
                    audio_features, languages, language_probs
                )
            ]

        # Convert to MLX array
        tokens = mx.array(tokens)

        # Expand for best-of-n sampling or beam search
        if self.n_group > 1:
            tokens = tokens[:, None, :]
            tokens = mx.broadcast_to(
                tokens, [n_audio, self.n_group, len(self.initial_tokens)]
            )
            tokens = tokens.reshape(n_audio * self.n_group, len(self.initial_tokens))

        # Main decoding loop
        tokens, sum_logprobs, no_speech_probs = self._main_loop(audio_features, tokens)

        # Reshape results
        audio_features = audio_features[:: self.n_group]
        no_speech_probs = no_speech_probs[:: self.n_group]
        assert audio_features.shape[0] == len(no_speech_probs) == n_audio

        tokens = tokens.reshape(n_audio, self.n_group, -1)
        sum_logprobs = sum_logprobs.reshape(n_audio, self.n_group)

        # Finalize and select best sequences
        tokens, sum_logprobs = self.decoder.finalize(tokens, sum_logprobs)
        tokens = tokens[..., self.sample_begin :].tolist()
        tokens = [
            [t[: t.index(self.tokenizer.eot)] if self.tokenizer.eot in t else t for t in s]
            for s in tokens
        ]

        # Rank and select best
        selected = self.sequence_ranker.rank(tokens, sum_logprobs)
        tokens: list[list[int]] = [t[i] for i, t in zip(selected, tokens)]
        texts: list[str] = [self.tokenizer.decode(t).strip() for t in tokens]

        sum_logprobs: list[float] = [lp[i] for i, lp in zip(selected, sum_logprobs)]
        avg_logprobs: list[float] = [
            lp / (len(t) + 1) for t, lp in zip(tokens, sum_logprobs)
        ]

        # Build results
        fields = (texts, languages, tokens, audio_features, avg_logprobs, no_speech_probs)
        if len(set(map(len, fields))) != 1:
            raise RuntimeError(f"Inconsistent result lengths: {list(map(len, fields))}")

        return [
            DecodingResult(
                audio_features=features,
                language=language,
                tokens=tokens,
                text=text,
                avg_logprob=avg_logprob,
                no_speech_prob=no_speech_prob,
                temperature=self.options.temperature,
                compression_ratio=compression_ratio(text),
            )
            for text, language, tokens, features, avg_logprob, no_speech_prob in zip(
                *fields
            )
        ]


def decode(
    model: Whisper,
    mel: mx.array,
    options: DecodingOptions = DecodingOptions(),
    **kwargs,
) -> Union[DecodingResult, list[DecodingResult]]:
    """Decode 30-second audio segment(s) from mel spectrogram(s).

    Main entry point for Whisper decoding.

    Args:
        model: Whisper model
        mel: Mel spectrogram of shape (80, 3000) or (batch, 80, 3000)
        options: Decoding options
        **kwargs: Override options with keyword arguments

    Returns:
        DecodingResult or list of DecodingResults

    Example:
        >>> from smlx.models.Whisper_tiny import load
        >>> from smlx.models.Whisper_tiny.audio import prepare_audio
        >>> from smlx.models.Whisper_tiny.decoding import decode, DecodingOptions
        >>>
        >>> model, tokenizer = load()
        >>> mel = prepare_audio("speech.wav")
        >>>
        >>> # Greedy decoding
        >>> result = decode(model, mel)
        >>> print(result.text)
        >>>
        >>> # With temperature sampling
        >>> options = DecodingOptions(temperature=0.5)
        >>> result = decode(model, mel, options)
        >>>
        >>> # Translate to English
        >>> options = DecodingOptions(task="translate", language="es")
        >>> result = decode(model, mel, options)
    """
    if single := mel.ndim == 2:
        mel = mel[None]

    if kwargs:
        options = replace(options, **kwargs)

    result = DecodingTask(model, options).run(mel)
    return result[0] if single else result
