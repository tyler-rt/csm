from dataclasses import dataclass

import torch
import torch.nn as nn
import torchtune
from torchtune.models import llama3_2


def llama3_2_1B() -> torchtune.modules.transformer.TransformerDecoder:
    """
    Creates a 1B parameter Llama 3.2 transformer model.
    
    Returns:
        A TransformerDecoder with 1B parameters configuration.
    """
    return llama3_2.llama3_2(
        vocab_size=128_256,
        num_layers=16,
        num_heads=32,
        num_kv_heads=8,
        embed_dim=2048,
        max_seq_len=2048,
        intermediate_dim=8192,
        attn_dropout=0.0,
        norm_eps=1e-5,
        rope_base=500_000,
        scale_factor=32,
    )


def llama3_2_100M() -> torchtune.modules.transformer.TransformerDecoder:
    """
    Creates a 100M parameter Llama 3.2 transformer model.
    
    Returns:
        A TransformerDecoder with 100M parameters configuration.
    """
    return llama3_2.llama3_2(
        vocab_size=128_256,
        num_layers=4,
        num_heads=8,
        num_kv_heads=2,
        embed_dim=1024,
        max_seq_len=2048,
        intermediate_dim=8192,
        attn_dropout=0.0,
        norm_eps=1e-5,
        rope_base=500_000,
        scale_factor=32,
    )


# Dictionary mapping model flavor names to their constructor functions
FLAVORS = {
    "llama-1B": llama3_2_1B,
    "llama-100M": llama3_2_100M,
}


def _prepare_transformer(model):
    """
    Prepares a transformer model for use in the CSM architecture by replacing
    the token embedding and output layers with Identity layers.
    
    Args:
        model: The transformer model to prepare
        
    Returns:
        tuple: (modified_model, embedding_dimension)
    """
    embed_dim = model.tok_embeddings.embedding_dim
    model.tok_embeddings = nn.Identity()
    model.output = nn.Identity()
    return model, embed_dim


def _create_causal_mask(seq_len: int, device: torch.device):
    """
    Creates a causal attention mask for transformer models.
    
    Args:
        seq_len: Maximum sequence length
        device: Device to create the mask on
        
    Returns:
        A boolean tensor of shape (seq_len, seq_len) with True in positions
        where attention is allowed (lower triangular matrix)
    """
    return torch.tril(torch.ones(seq_len, seq_len, dtype=torch.bool, device=device))


def _index_causal_mask(mask: torch.Tensor, input_pos: torch.Tensor):
    """
    Indexes into a causal mask using position indices.
    
    Args:
        mask: (max_seq_len, max_seq_len) causal mask
        input_pos: (batch_size, seq_len) position indices
        
    Returns:
        (batch_size, seq_len, max_seq_len) indexed mask
    """
    r = mask[input_pos, :]
    return r


def _multinomial_sample_one_no_sync(probs):
    """
    Performs multinomial sampling without a CUDA synchronization.
    
    Args:
        probs: Probability distribution to sample from
        
    Returns:
        A single sampled token index
    """
    q = torch.empty_like(probs).exponential_(1)
    return torch.argmax(probs / q, dim=-1, keepdim=True).to(dtype=torch.int)


def sample_topk(logits: torch.Tensor, topk: int, temperature: float):
    """
    Samples from the top-k logits after temperature scaling.
    
    Args:
        logits: Raw logits from the model
        topk: Number of top logits to consider
        temperature: Temperature for scaling logits (higher = more random)
        
    Returns:
        A sampled token index
    """
    # Apply temperature scaling
    logits = logits / temperature

    # Filter to only the top-k logits
    filter_value: float = -float("Inf")
    indices_to_remove = logits < torch.topk(logits, topk)[0][..., -1, None]
    scores_processed = logits.masked_fill(indices_to_remove, filter_value)
    scores_processed = torch.nn.functional.log_softmax(scores_processed, dim=-1)
    probs = torch.nn.functional.softmax(scores_processed, dim=-1)

    # Sample from the filtered distribution
    sample_token = _multinomial_sample_one_no_sync(probs)
    return sample_token


@dataclass
class ModelArgs:
    """
    Configuration arguments for the CSM model.
    
    Attributes:
        backbone_flavor: Model flavor for the backbone transformer
        decoder_flavor: Model flavor for the decoder transformer
        text_vocab_size: Size of the text vocabulary
        audio_vocab_size: Size of each audio codebook vocabulary
        audio_num_codebooks: Number of audio codebooks in the RVQ tokenizer
    """
    backbone_flavor: str
    decoder_flavor: str
    text_vocab_size: int
    audio_vocab_size: int
    audio_num_codebooks: int


class Model(nn.Module):
    """
    Conversational Speech Model (CSM) implementation.
    
    This model uses a two-stage architecture:
    1. A backbone transformer that processes interleaved text and audio tokens
    2. A decoder transformer that generates multiple codebooks for high-fidelity audio
    """
    def __init__(self, args: ModelArgs):
        """
        Initialize the CSM model.
        
        Args:
            args: Model configuration arguments
        """
        super().__init__()
        self.args = args

        # Initialize backbone and decoder transformers
        self.backbone, backbone_dim = _prepare_transformer(FLAVORS[args.backbone_flavor]())
        self.decoder, decoder_dim = _prepare_transformer(FLAVORS[args.decoder_flavor]())

        # Embedding layers for text and audio tokens
        self.text_embeddings = nn.Embedding(args.text_vocab_size, backbone_dim)
        self.audio_embeddings = nn.Embedding(args.audio_vocab_size * args.audio_num_codebooks, backbone_dim)

        # Projection from backbone to decoder dimension
        self.projection = nn.Linear(backbone_dim, decoder_dim, bias=False)
        
        # Output heads for audio codebooks
        self.codebook0_head = nn.Linear(backbone_dim, args.audio_vocab_size, bias=False)
        self.audio_head = nn.Parameter(torch.empty(args.audio_num_codebooks - 1, decoder_dim, args.audio_vocab_size))

    def setup_caches(self, max_batch_size: int) -> torch.Tensor:
        """
        Setup KV caches for efficient inference and create causal masks.
        
        Args:
            max_batch_size: Maximum batch size for inference
            
        Returns:
            Causal mask tensor
        """
        dtype = next(self.parameters()).dtype
        device = next(self.parameters()).device

        with device:
            # Setup KV caches for both transformers
            self.backbone.setup_caches(max_batch_size, dtype)
            self.decoder.setup_caches(max_batch_size, dtype, decoder_max_seq_len=self.args.audio_num_codebooks)

        # Create and register causal masks as buffers
        self.register_buffer("backbone_causal_mask", _create_causal_mask(self.backbone.max_seq_len, device))
        self.register_buffer("decoder_causal_mask", _create_causal_mask(self.args.audio_num_codebooks, device))

    def generate_frame(
        self,
        tokens: torch.Tensor,
        tokens_mask: torch.Tensor,
        input_pos: torch.Tensor,
        temperature: float,
        topk: int,
    ) -> torch.Tensor:
        """
        Generate a single frame of audio tokens (all codebooks).
        
        Args:
            tokens: (batch_size, seq_len, audio_num_codebooks+1) token indices
            tokens_mask: (batch_size, seq_len, audio_num_codebooks+1) boolean mask
            input_pos: (batch_size, seq_len) positions for each token
            temperature: Sampling temperature
            topk: Number of top logits to sample from
            
        Returns:
            (batch_size, audio_num_codebooks) sampled audio tokens for one frame
        """
        dtype = next(self.parameters()).dtype
        b, s, _ = tokens.size()

        # Verify caches are enabled for efficient inference
        assert self.backbone.caches_are_enabled(), "backbone caches are not enabled"
        
        # Get causal mask for current positions
        curr_backbone_mask = _index_causal_mask(self.backbone_causal_mask, input_pos)
        
        # Embed and process tokens through backbone transformer
        embeds = self._embed_tokens(tokens)
        masked_embeds = embeds * tokens_mask.unsqueeze(-1)
        h = masked_embeds.sum(dim=2)
        h = self.backbone(h, input_pos=input_pos, mask=curr_backbone_mask).to(dtype=dtype)

        # Generate first codebook (c0) using backbone output
        last_h = h[:, -1, :]
        c0_logits = self.codebook0_head(last_h)
        c0_sample = sample_topk(c0_logits, topk, temperature)
        c0_embed = self._embed_audio(0, c0_sample)

        # Prepare for decoder stage
        curr_h = torch.cat([last_h.unsqueeze(1), c0_embed], dim=1)
        curr_sample = c0_sample.clone()
        curr_pos = torch.arange(0, curr_h.size(1), device=curr_h.device).unsqueeze(0).repeat(curr_h.size(0), 1)

        # Reset decoder caches for this frame
        self.decoder.reset_caches()
        
        # Autoregressively generate remaining codebooks
        for i in range(1, self.args.audio_num_codebooks):
            curr_decoder_mask = _index_causal_mask(self.decoder_causal_mask, curr_pos)
            decoder_h = self.decoder(self.projection(curr_h), input_pos=curr_pos, mask=curr_decoder_mask).to(
                dtype=dtype
            )
            ci_logits = torch.mm(decoder_h[:, -1, :], self.audio_head[i - 1])
            ci_sample = sample_topk(ci_logits, topk, temperature)
            ci_embed = self._embed_audio(i, ci_sample)

            curr_h = ci_embed
            curr_sample = torch.cat([curr_sample, ci_sample], dim=1)
            curr_pos = curr_pos[:, -1:] + 1

        return curr_sample

    def reset_caches(self):
        """Reset KV caches for both transformers."""
        self.backbone.reset_caches()
        self.decoder.reset_caches()

    def _embed_audio(self, codebook: int, tokens: torch.Tensor) -> torch.Tensor:
        """
        Embed audio tokens for a specific codebook.
        
        Args:
            codebook: Codebook index
            tokens: Token indices to embed
            
        Returns:
            Embedded audio tokens
        """
        return self.audio_embeddings(tokens + codebook * self.args.audio_vocab_size)

    def _embed_tokens(self, tokens: torch.Tensor) -> torch.Tensor:
        """
        Embed both text and audio tokens from the unified token representation.
        
        The token tensor has shape (batch_size, seq_len, audio_num_codebooks+1) where:
        - The first audio_num_codebooks positions contain audio tokens
        - The last position contains text tokens
        
        Args:
            tokens: Unified token representation
            
        Returns:
            Embedded tokens with shape (batch_size, seq_len, audio_num_codebooks+1, embed_dim)
        """
        # Embed text tokens (from the last position)
        text_embeds = self.text_embeddings(tokens[:, :, -1]).unsqueeze(-2)

        # Embed audio tokens (from the first audio_num_codebooks positions)
        audio_tokens = tokens[:, :, :-1] + (
            self.args.audio_vocab_size * torch.arange(self.args.audio_num_codebooks, device=tokens.device)
        )
        audio_embeds = self.audio_embeddings(audio_tokens.view(-1)).reshape(
            tokens.size(0), tokens.size(1), self.args.audio_num_codebooks, -1
        )

        # Concatenate audio and text embeddings
        return torch.cat([audio_embeds, text_embeds], dim=-2)
