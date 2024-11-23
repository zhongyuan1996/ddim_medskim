import torch
import torch.nn as nn
from torch.nn import TransformerEncoder, TransformerEncoderLayer
import torch.nn.functional as F
from einops import repeat
import math

class DenoiseTransformerLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(
            embed_dim=config.hidden_size,
            num_heads=config.denoise_num_attention_heads,
            dropout=config.hidden_dropout_prob,
            batch_first=True
        )
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=config.hidden_size,
            num_heads=config.denoise_num_attention_heads,
            dropout=config.hidden_dropout_prob,
            batch_first=True
        )
        
        self.ff = nn.Sequential(
            nn.Linear(config.hidden_size, config.denoise_intermediate_size),
            nn.GELU(),
            nn.Linear(config.denoise_intermediate_size, config.hidden_size)
        )
        
        self.norm1 = nn.LayerNorm(config.hidden_size)
        self.norm2 = nn.LayerNorm(config.hidden_size)
        self.norm3 = nn.LayerNorm(config.hidden_size)
        
    def forward(self, x, prompts, prompt_attention_mask=None):
        # print("\nDenoiseTransformerLayer Forward Pass:")
        # print(f"Input x shape: {x.shape}")
        # print(f"Prompts shape: {prompts.shape}")
        # if prompt_attention_mask is not None:
        #     print(f"Original attention mask shape: {prompt_attention_mask.shape}")
            
        #     # Print sample of original mask
        #     print("\nSample of original attention mask (first 5 tokens, first 10 prompt positions):")
        #     print(prompt_attention_mask[0, :5, :10])
        
        # Self attention
        residual = x
        x = self.norm1(x)
        x, _ = self.self_attn(x, x, x)
        x = residual + x
        
        # Cross attention with prompts
        residual = x
        x = self.norm2(x)
        
        # Reshape attention mask if provided
        if prompt_attention_mask is not None:
            mask = prompt_attention_mask[0].float()
            mask = mask.masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
            # print(f"\nReshaped mask shape: {mask.shape}")
            
            # # Print sample of transformed mask
            # print("Sample of transformed mask (first 5 tokens, first 10 prompt positions):")
            # print(mask[:5, :10])
            
            # # Print statistics about the mask
            # print("\nMask statistics:")
            # print(f"Number of -inf values: {(mask == float('-inf')).sum()}")
            # print(f"Number of 0 values: {(mask == 0.0).sum()}")
        else:
            mask = None
            print("No attention mask provided")
            
        x, _ = self.cross_attn(
            query=x,
            key=prompts,
            value=prompts,
            attn_mask=mask
        )
        x = residual + x
        
        # Feed forward
        residual = x
        x = self.norm3(x)
        x = self.ff(x)
        x = residual + x
        
        return x
    
class LatentDiffusionModule(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # Time embedding projection
        self.time_proj = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size),
            nn.GELU(),
            nn.Linear(config.hidden_size, config.hidden_size)
        )
        
        # Denoising transformer layers with cross attention
        self.denoise_layers = nn.ModuleList([
            DenoiseTransformerLayer(config)
            for _ in range(config.denoise_num_hidden_layers)
        ])
    
    def get_noise_schedule(self, t, beta_start=1e-4, beta_end=0.02):
        """Linear noise schedule."""
        beta_t = beta_start + t * (beta_end - beta_start)
        alpha_t = 1 - beta_t
        alpha_bar_t = torch.cumprod(alpha_t, dim=0)
        return alpha_bar_t.sqrt(), (1 - alpha_bar_t).sqrt()
    
    def add_noise(self, latents, t, modality_mask, special_tokens_mask):
        """Add noise to latent representations."""
        device = latents.device
        batch_size = latents.shape[0]
        noised_latents = latents.clone()
        
        for modality, time in t.items():
            current_mask = modality_mask[modality]
            if current_mask.any():
                sqrt_alpha_bar, sqrt_one_minus_alpha_bar = self.get_noise_schedule(time)
                noise = torch.randn_like(latents)
                noised_latents_current = sqrt_alpha_bar.view(-1, 1, 1) * latents + sqrt_one_minus_alpha_bar.view(-1, 1, 1) * noise
                noised_latents = torch.where(current_mask.unsqueeze(-1), noised_latents_current, noised_latents)
        
        return noised_latents
    
    def denoise(self, noised_latents, t, modality_mask, special_tokens_mask, modality_prompts, cross_prompt):
        """Denoise all modalities simultaneously with modality-specific and global prompt conditioning."""
        batch_size = noised_latents.shape[0]
        device = noised_latents.device
        
        # First, print shape information
        # print("\nShape Information:")
        # print(f"noised_latents shape: {noised_latents.shape}")
        # print(f"prompt shapes:")
        # for modality, prompt in modality_prompts.items():
        #     print(f"{modality} prompt shape: {prompt.shape}")
        # print(f"cross prompt shape: {cross_prompt.shape}")
        
        # Print modality mask information
        # print("\nModality Mask Samples (first sequence):")
        # for modality, mask in modality_mask.items():
        #     true_positions = mask[0].nonzero().squeeze(-1)
        #     print(f"{modality} tokens at positions: {true_positions.tolist()}")
        
        # Time conditioning (keeping the same)
        time_cond = torch.zeros_like(noised_latents)
        for modality, time in t.items():
            time_embed = get_timestep_embedding(time, self.config.hidden_size)
            time_embed = self.time_proj(time_embed)
            modality_tokens = modality_mask[modality]
            time_cond = torch.where(
                modality_tokens.unsqueeze(-1),
                time_embed.view(-1, 1, self.config.hidden_size),
                time_cond
            )
        hidden_states = noised_latents + time_cond
        
        # Create attention mask with debugging prints
        total_prompt_length = sum(p.size(1) for p in modality_prompts.values()) + cross_prompt.size(1)
        # print(f"\nTotal prompt length: {total_prompt_length}")
        
        combined_mask = torch.zeros((batch_size, noised_latents.size(1), total_prompt_length), device=device)
        current_prompt_idx = 0
        
        # Handle modality-specific prompts
        all_prompts = []
        for modality, prompt in modality_prompts.items():
            all_prompts.append(prompt)
            prompt_length = prompt.size(1)
            
            prompt_section = torch.zeros((batch_size, noised_latents.size(1), prompt_length), device=device)
            prompt_section[modality_mask[modality]] = 1
            
            # # Print section information
            # print(f"\n{modality} prompt section:")
            # print(f"Index range: {current_prompt_idx} to {current_prompt_idx + prompt_length}")
            # print(f"Number of attending tokens: {modality_mask[modality][0].sum()}")
            
            # Add to combined mask
            combined_mask[:, :, current_prompt_idx:current_prompt_idx + prompt_length] = prompt_section
            current_prompt_idx += prompt_length
        
        # Add cross prompt section
        all_prompts.append(cross_prompt)
        combined_mask[:, :, -cross_prompt.size(1):] = 1
        # print(f"\nCross prompt section: last {cross_prompt.size(1)} positions")
        
        # # Print final mask statistics
        # print("\nFinal mask statistics for first sequence:")
        # print(f"Shape: {combined_mask.shape}")
        # print("Attention pattern:")
        # for i in range(noised_latents.size(1)):
        #     if combined_mask[0, i].any():
        #         attending_to = combined_mask[0, i].nonzero().squeeze(-1)
        #         print(f"Token {i} attends to prompt positions: {attending_to.tolist()}")
        
        # Concatenate prompts
        combined_prompts = torch.cat(all_prompts, dim=1)
        assert combined_prompts.size(1) == total_prompt_length, "Prompt length mismatch!"
        
        # Process through layers
        for layer in self.denoise_layers:
            hidden_states = layer(hidden_states, combined_prompts, combined_mask)
        
        return hidden_states


def get_timestep_embedding(timesteps, embedding_dim, max_period=10000):
    """
    Create sinusoidal timestep embeddings.
    :param timesteps: a 1-D Tensor of N indices, one per batch element.
    :param embedding_dim: the dimension of the output.
    :param max_period: controls the minimum frequency of the embeddings.
    :return: an [N x embedding_dim] Tensor of positional embeddings.
    """
    half = embedding_dim // 2
    freqs = torch.exp(
        -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
    ).to(device=timesteps.device)
    args = timesteps[:, None].float() * freqs[None]
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if embedding_dim % 2:
        embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
    return embedding

class promptTransformerBlock(nn.Module):
    def __init__(self, config, is_cross_modal=False):
        super(promptTransformerBlock, self).__init__()
        self.config = config

        num_heads = config.cross_modal_num_attention_heads if is_cross_modal else config.modality_num_attention_heads
        num_layers = config.cross_modal_num_hidden_layers if is_cross_modal else config.modality_num_hidden_layers
        intermediate_size = config.cross_modal_intermediate_size if is_cross_modal else config.modality_intermediate_size
        
        encoder_layer = TransformerEncoderLayer(
            d_model=config.hidden_size,
            nhead=num_heads,
            dim_feedforward=intermediate_size,
            dropout=config.hidden_dropout_prob,
            batch_first=True,
        )
        self.transformer = TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.modality_prompt = nn.Parameter(torch.randn(1, config.num_modality_tokens, config.hidden_size))
    
    def forward(self, hidden_states, attention_mask=None):
        batch_size, seq_len, _ = hidden_states.size()
        prompts = repeat(self.modality_prompt, '1 n d -> b n d', b=batch_size)
        hidden_states = torch.cat([prompts, hidden_states], dim=1)

        # Usually attention_mask is None
        if attention_mask is not None:
            attention_mask = F.pad(attention_mask, (self.modality_prompt.size(1), 0), value=1)
            attention_mask = attention_mask.float().masked_fill(
                attention_mask == 0, float('-inf')).masked_fill(attention_mask == 1, float(0.0))
            
        # print(attention_mask[0,0:50])

        hidden_states = self.transformer(hidden_states, src_key_padding_mask=attention_mask)

        # Return both the prompt and the hidden states
        return hidden_states[:, self.modality_prompt.size(1):, :], hidden_states[:, :self.modality_prompt.size(1), :]


class regTransformerBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        self.self_attn = nn.MultiheadAttention(
            embed_dim=config.hidden_size,
            num_heads=config.shared_num_attention_heads,
            dropout=config.hidden_dropout_prob,
            batch_first=True
        )
        
        self.ff = nn.Sequential(
            nn.Linear(config.hidden_size, config.shared_intermediate_size),
            nn.GELU(),
            nn.Linear(config.shared_intermediate_size, config.hidden_size)
        )
        
        self.norm1 = nn.LayerNorm(config.hidden_size)
        self.norm2 = nn.LayerNorm(config.hidden_size)
        
    def forward(self, hidden_states, modality_mask=None):
        # Layer norm and residual connection for self-attention
        residual = hidden_states
        hidden_states = self.norm1(hidden_states)

        if modality_mask is not None:
            batch_size, seq_len, _ = modality_mask.size()
            
            # Add diagonal entries to the modality mask
            diagonal_mask = torch.eye(seq_len, device=modality_mask.device).unsqueeze(0).repeat(batch_size, 1, 1)
            modality_mask = modality_mask.bool() | diagonal_mask.bool()
            
            # Expand mask for multi-head attention
            attention_mask = modality_mask.unsqueeze(1).repeat(1, self.config.shared_num_attention_heads, 1, 1)
            attention_mask = attention_mask.view(
                -1, attention_mask.size(-2), attention_mask.size(-1)
            )
            attention_mask = attention_mask.float().masked_fill(
                attention_mask == 0, -1e5
            )
        else:
            attention_mask = None
        
        # Self-attention
        hidden_states, _ = self.self_attn(hidden_states, hidden_states, hidden_states, attn_mask=attention_mask)
        hidden_states = residual + hidden_states
        # print(hidden_states[0,4:10,4:10])
        # Layer norm and residual connection for feed-forward
        residual = hidden_states
        hidden_states = self.norm2(hidden_states)
        hidden_states = self.ff(hidden_states)
        hidden_states = residual + hidden_states
        # print(hidden_states[0,0:10,0:10])
        return hidden_states


class EHRModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # Token embeddings
        self.token_embedding = nn.Embedding(config.total_vocab_size, config.hidden_size)
        self.position_embedding = nn.Embedding(config.max_position_embeddings, config.hidden_size) #not used for now

        # Modality positional embeddings
        self.modality_embedding = nn.Embedding(config.num_modalities + 1, config.hidden_size)

        # Shared parameter layer
        self.shared_transformer = regTransformerBlock(config)

        # Modality-specific prompt transformers
        self.modaltiy_transformers = nn.ModuleDict({
            'diag': promptTransformerBlock(config, is_cross_modal=False),
            'drug': promptTransformerBlock(config, is_cross_modal=False),
            'proc': promptTransformerBlock(config, is_cross_modal=False),
            'lab': promptTransformerBlock(config, is_cross_modal=False),
        })

        # Cross-modality transformer
        self.cross_modal_transformer = promptTransformerBlock(config, is_cross_modal=True)

        self.diffusion = LatentDiffusionModule(config)

        self.prediction_heads = nn.ModuleDict({
            'diag': nn.Linear(config.hidden_size, config.diag_vocab_size),
            'drug': nn.Linear(config.hidden_size, config.drug_vocab_size),
            'proc': nn.Linear(config.hidden_size, config.proc_vocab_size),
            'lab': nn.Linear(config.hidden_size, config.lab_vocab_size),
        })

    def get_modality_segments(self, input_ids, special_tokens):
        """Split sequence into modality segments based on special tokens."""
        batch_size = input_ids.size(0)
        segments = {}
        
        for modality in ['diag', 'drug', 'proc', 'lab']:
            start_token = special_tokens[f'{modality}_start']
            end_token = special_tokens[f'{modality}_end']
            
            # Create mask for this modality
            modality_mask = torch.zeros_like(input_ids, dtype=torch.bool)
            
            # Process each sequence in the batch
            for b in range(batch_size):
                seq = input_ids[b]
                start_positions = (seq == start_token).nonzero(as_tuple=True)[0]
                end_positions = (seq == end_token).nonzero(as_tuple=True)[0]
                
                # Ensure starts and ends are paired correctly
                if len(start_positions) != len(end_positions):
                    raise ValueError(f"Unmatched start and end tokens for modality {modality} in sequence {b}")
                
                # Mark the tokens between each start-end pair
                for start, end in zip(start_positions, end_positions):
                    if end > start:  # Ensure valid range
                        # Include tokens between start and end (excluding the tokens themselves)
                        modality_mask[b, start+1:end] = True
            
            segments[modality] = modality_mask
            
        return segments

    def forward(self, input_ids, seq_mask, modality_mask, modality_indices, timesteps=None):
        batch_size, seq_len = input_ids.size()

        # positional embeddings are not considered right now

        # Token embeddings
        input_embeddings = self.token_embedding(input_ids)

        # Shared transformer
        shared_output = self.shared_transformer(input_embeddings, modality_mask)

        #########################################################################
        # Calculate modality-specific outputs and prompts

        segments = self.get_modality_segments(input_ids, self.config.special_tokens)

        #modality_outputs is not used for now
        modality_outputs, modality_prompts = {}, {}



        for modality, mask in segments.items():
            # Extract modality-specific tokens
            modality_hidden = shared_output.masked_fill(~mask.unsqueeze(-1), 0)

            # Convert seq_mask to boolean if it isn't already
            seq_mask_bool = seq_mask.bool() if not seq_mask.dtype == torch.bool else seq_mask
            mask_bool = mask.bool() if not mask.dtype == torch.bool else mask
            modality_att_mask = seq_mask_bool & mask_bool
            
            # Process through modality-specific transformer
            modality_output, prompt_states = self.modaltiy_transformers[modality](
                modality_hidden,
                attention_mask=modality_att_mask if self.training else None
            )
            
            modality_outputs[modality] = modality_output
            modality_prompts[modality] = prompt_states

        #########################################################################
        # Calculate cross-modality output to allow for cross-modality interactions

        # Add modality positional embeddings
        if modality_indices is not None:
            modality_positional_embeddings = self.modality_embedding(modality_indices)
            shared_output += modality_positional_embeddings

        cross_output, cross_prompt = self.cross_modal_transformer(
            shared_output,
            attention_mask=seq_mask if self.training else None
        )

        #########################################################################
        # Latent diffusion
        if self.training and timesteps is not None:
            # print("\nInput shapes:")
            # print(f"input_ids: {input_ids.shape}")
            # print(f"cross_output: {cross_output.shape}")

            special_tokens_mask = torch.zeros_like(input_ids, dtype=torch.bool)
            for token in self.config.special_tokens.values():
                special_tokens_mask |= (input_ids == token)
            
            noised_latents = self.diffusion.add_noise(
                cross_output,
                timesteps,
                segments,
                special_tokens_mask
            )
            
            denoised_latents = self.diffusion.denoise(
                noised_latents,
                timesteps,
                segments,
                special_tokens_mask,
                modality_prompts,
                cross_prompt
            )
            
            valid_tokens = ~special_tokens_mask
            diffusion_loss = F.mse_loss(
                denoised_latents[valid_tokens],
                cross_output[valid_tokens]
            )
            # print(f"\nDiffusion loss: {diffusion_loss.item():.4f}")
            
            # Get predictions for each modality
            predictions = {}
            # print("\nPredictions per modality:")
            for modality, mask in segments.items():
                valid_tokens = mask & ~special_tokens_mask
                modality_preds = self.prediction_heads[modality](denoised_latents)
                
                # print(f"\n{modality}:")
                # print(f"Prediction logits shape: {modality_preds.shape}")
                # print(f"Valid tokens in first batch: {valid_tokens[0].sum().item()}")
                # print(f"Valid token positions in first batch: {valid_tokens[0].nonzero().squeeze(-1)[:10].tolist()[:10]}...")
                
                # # Sample of prediction values for first batch, first valid position
                # first_valid_pos = valid_tokens[0].nonzero()[0].item() if valid_tokens[0].any() else 0
                # print(f"Sample logits at first valid position: {modality_preds[0, first_valid_pos, :5].tolist()}")
                
                predictions[modality] = {
                    'logits': modality_preds,
                    'mask': valid_tokens
                }

            return predictions, diffusion_loss
            
        else:
            # Create special tokens mask for inference 
            special_tokens_mask = torch.zeros_like(input_ids, dtype=torch.bool)
            for token in self.config.special_tokens.values():
                special_tokens_mask |= (input_ids == token)
                    
            # Inference predictions per modality
            predictions = {}
            for modality, mask in segments.items():
                valid_tokens = mask & ~special_tokens_mask
                modality_preds = self.prediction_heads[modality](cross_output)
                
                predictions[modality] = {
                    'logits': modality_preds,
                    'mask': valid_tokens
                }
            
            return predictions, None

        

        



