from dataclasses import dataclass
from typing import List, Tuple

import torch
import torchaudio
from huggingface_hub import hf_hub_download
from models import Model, ModelArgs
from moshi.models import loaders
from tokenizers.processors import TemplateProcessing
from transformers import AutoTokenizer
from watermarking import CSM_1B_GH_WATERMARK, load_watermarker, watermark


@dataclass
class Segment:
    """
    Represents a segment of conversation with text and corresponding audio.
    
    Attributes:
        speaker: Speaker ID
        text: Text content of the segment
        audio: Audio waveform tensor with sample_rate = 24_000
    """
    speaker: int
    text: str
    # (num_samples,), sample_rate = 24_000
    audio: torch.Tensor


def load_llama3_tokenizer():
    """
    Load and configure the Llama 3 tokenizer with appropriate BOS/EOS tokens.
    
    This function addresses a specific issue with the Llama 3 tokenizer
    (see: https://github.com/huggingface/transformers/issues/22794#issuecomment-2092623992)
    
    Returns:
        Configured tokenizer
    """
    tokenizer_name = "meta-llama/Llama-3.2-1B"
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    bos = tokenizer.bos_token
    eos = tokenizer.eos_token
    tokenizer._tokenizer.post_processor = TemplateProcessing(
        single=f"{bos}:0 $A:0 {eos}:0",
        pair=f"{bos}:0 $A:0 {eos}:0 {bos}:1 $B:1 {eos}:1",
        special_tokens=[(f"{bos}", tokenizer.bos_token_id), (f"{eos}", tokenizer.eos_token_id)],
    )

    return tokenizer


class Generator:
    """
    Speech generation interface for the Conversational Speech Model (CSM).
    
    This class handles tokenization, generation, and decoding of speech from text,
    maintaining conversational context across multiple turns.
    """
    def __init__(
        self,
        model: Model,
    ):
        """
        Initialize the generator with a CSM model.
        
        Args:
            model: Trained CSM model
        """
        self._model = model
        self._model.setup_caches(1)  # Setup KV caches for batch size 1

        # Initialize text tokenizer (Llama 3)
        self._text_tokenizer = load_llama3_tokenizer()

        # Initialize audio tokenizer (Mimi RVQ)
        device = next(model.parameters()).device
        mimi_weight = hf_hub_download(loaders.DEFAULT_REPO, loaders.MIMI_NAME)
        mimi = loaders.get_mimi(mimi_weight, device=device)
        mimi.set_num_codebooks(32)
        self._audio_tokenizer = mimi

        # Initialize watermarker
        self._watermarker = load_watermarker(device=device)

        self.sample_rate = mimi.sample_rate
        self.device = device

    def _tokenize_text_segment(self, text: str, speaker: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Tokenize a text segment with speaker ID.
        
        Args:
            text: Text content
            speaker: Speaker ID
            
        Returns:
            tuple: (tokens, mask) where:
                - tokens: (seq_len, 33) token indices with text in last position
                - mask: (seq_len, 33) boolean mask with True in last position
        """
        frame_tokens = []
        frame_masks = []

        # Tokenize text with speaker ID prefix
        text_tokens = self._text_tokenizer.encode(f"[{speaker}]{text}")
        
        # Create frame with text tokens in the last position (index 32)
        text_frame = torch.zeros(len(text_tokens), 33).long()
        text_frame_mask = torch.zeros(len(text_tokens), 33).bool()
        text_frame[:, -1] = torch.tensor(text_tokens)
        text_frame_mask[:, -1] = True

        frame_tokens.append(text_frame.to(self.device))
        frame_masks.append(text_frame_mask.to(self.device))

        return torch.cat(frame_tokens, dim=0), torch.cat(frame_masks, dim=0)

    def _tokenize_audio(self, audio: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Tokenize audio waveform into RVQ tokens.
        
        Args:
            audio: Audio waveform tensor
            
        Returns:
            tuple: (tokens, mask) where:
                - tokens: (seq_len, 33) token indices with audio in first 32 positions
                - mask: (seq_len, 33) boolean mask with True in first 32 positions
        """
        frame_tokens = []
        frame_masks = []

        # Encode audio to RVQ tokens (K codebooks, T frames)
        audio = audio.to(self.device)
        audio_tokens = self._audio_tokenizer.encode(audio.unsqueeze(0).unsqueeze(0))[0]
        
        # Add EOS frame (zeros)
        eos_frame = torch.zeros(audio_tokens.size(0), 1).to(self.device)
        audio_tokens = torch.cat([audio_tokens, eos_frame], dim=1)

        # Create frame with audio tokens in the first 32 positions
        audio_frame = torch.zeros(audio_tokens.size(1), 33).long().to(self.device)
        audio_frame_mask = torch.zeros(audio_tokens.size(1), 33).bool().to(self.device)
        audio_frame[:, :-1] = audio_tokens.transpose(0, 1)
        audio_frame_mask[:, :-1] = True

        frame_tokens.append(audio_frame)
        frame_masks.append(audio_frame_mask)

        return torch.cat(frame_tokens, dim=0), torch.cat(frame_masks, dim=0)

    def _tokenize_segment(self, segment: Segment) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Tokenize a complete segment (text + audio).
        
        Args:
            segment: Segment containing text and audio
            
        Returns:
            tuple: (tokens, mask) for the complete segment
        """
        text_tokens, text_masks = self._tokenize_text_segment(segment.text, segment.speaker)
        audio_tokens, audio_masks = self._tokenize_audio(segment.audio)

        return torch.cat([text_tokens, audio_tokens], dim=0), torch.cat([text_masks, audio_masks], dim=0)

    @torch.inference_mode()
    def generate(
        self,
        text: str,
        speaker: int,
        context: List[Segment],
        max_audio_length_ms: float = 90_000,
        temperature: float = 0.9,
        topk: int = 50,
    ) -> torch.Tensor:
        """
        Generate speech for text with conversational context.
        
        Args:
            text: Text to convert to speech
            speaker: Speaker ID
            context: List of previous conversation segments
            max_audio_length_ms: Maximum audio length in milliseconds
            temperature: Sampling temperature (higher = more random)
            topk: Number of top logits to sample from
            
        Returns:
            Audio waveform tensor
        """
        self._model.reset_caches()

        # Calculate maximum number of audio frames
        max_audio_frames = int(max_audio_length_ms / 80)  # 80ms per frame
        
        # Tokenize context segments
        tokens, tokens_mask = [], []
        for segment in context:
            segment_tokens, segment_tokens_mask = self._tokenize_segment(segment)
            tokens.append(segment_tokens)
            tokens_mask.append(segment_tokens_mask)

        # Tokenize the text to be generated
        gen_segment_tokens, gen_segment_tokens_mask = self._tokenize_text_segment(text, speaker)
        tokens.append(gen_segment_tokens)
        tokens_mask.append(gen_segment_tokens_mask)

        # Concatenate all tokens into a single prompt
        prompt_tokens = torch.cat(tokens, dim=0).long().to(self.device)
        prompt_tokens_mask = torch.cat(tokens_mask, dim=0).bool().to(self.device)

        # Prepare for generation
        samples = []
        curr_tokens = prompt_tokens.unsqueeze(0)
        curr_tokens_mask = prompt_tokens_mask.unsqueeze(0)
        curr_pos = torch.arange(0, prompt_tokens.size(0)).unsqueeze(0).long().to(self.device)

        # Check if input is too long
        max_seq_len = 2048 - max_audio_frames
        if curr_tokens.size(1) >= max_seq_len:
            raise ValueError(f"Inputs too long, must be below max_seq_len - max_audio_frames: {max_seq_len}")

        # Generate audio frames autoregressively
        for _ in range(max_audio_frames):
            sample = self._model.generate_frame(curr_tokens, curr_tokens_mask, curr_pos, temperature, topk)
            if torch.all(sample == 0):
                break  # Stop at EOS (all zeros)

            samples.append(sample)

            # Prepare next position
            curr_tokens = torch.cat([sample, torch.zeros(1, 1).long().to(self.device)], dim=1).unsqueeze(1)
            curr_tokens_mask = torch.cat(
                [torch.ones_like(sample).bool(), torch.zeros(1, 1).bool().to(self.device)], dim=1
            ).unsqueeze(1)
            curr_pos = curr_pos[:, -1:] + 1

        # Decode audio tokens to waveform
        audio = self._audio_tokenizer.decode(torch.stack(samples).permute(1, 2, 0)).squeeze(0).squeeze(0)

        # Apply watermark to identify as AI-generated audio
        # This applies an imperceptible watermark to identify audio as AI-generated.
        # Watermarking ensures transparency, dissuades misuse, and enables traceability.
        # Please be a responsible AI citizen and keep the watermarking in place.
        # If using CSM 1B in another application, use your own private key and keep it secret.
        audio, wm_sample_rate = watermark(self._watermarker, audio, self.sample_rate, CSM_1B_GH_WATERMARK)
        audio = torchaudio.functional.resample(audio, orig_freq=wm_sample_rate, new_freq=self.sample_rate)

        return audio


def load_csm_1b(ckpt_path: str = "ckpt.pt", device: str = "cuda") -> Generator:
    """
    Load the CSM 1B model from a checkpoint file.
    
    Args:
        ckpt_path: Path to the model checkpoint
        device: Device to load the model on ("cuda" or "cpu")
        
    Returns:
        Initialized Generator with the loaded model
    """
    # Configure model architecture
    model_args = ModelArgs(
        backbone_flavor="llama-1B",
        decoder_flavor="llama-100M",
        text_vocab_size=128256,
        audio_vocab_size=2051,
        audio_num_codebooks=32,
    )
    
    # Create and load model
    model = Model(model_args).to(device=device, dtype=torch.bfloat16)
    state_dict = torch.load(ckpt_path)
    model.load_state_dict(state_dict)

    # Create generator
    generator = Generator(model)
    return generator
