import torch
import warnings
import numpy as np
import math
from tqdm import tqdm
from torch.nn import functional as F

class InVisitEvaluator:
    def __init__(self, model, device=None):
        self.model = model
        self.device = device if device is not None else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)

    def calculate_perplexity(self, logits, targets, mask):
        """Basic perplexity calculation for masked positions."""
        # Get positions where mask is True
        batch_idx, seq_idx = torch.where(mask)
        if len(batch_idx) == 0:
            return None

        # Get logits and targets for masked positions
        masked_logits = logits[batch_idx, seq_idx]  # (num_masked, vocab_size) 
        masked_targets = targets[batch_idx, seq_idx]  # (num_masked,)

        # Calculate NLL using cross entropy
        nll = F.cross_entropy(masked_logits, masked_targets, reduction='none')
        ppl = torch.exp(nll)

        # Remove invalid values
        valid_ppl = ppl[~torch.isnan(ppl) & ~torch.isinf(ppl)]
        return valid_ppl if len(valid_ppl) > 0 else None

    def lpl(self, logits, targets, mask):
        """
        Calculate in-visit longitudinal perplexity for single modality.
        This measures how well we predict tokens within one modality.
        """
        valid_ppl = self.calculate_perplexity(logits, targets, mask)
        if valid_ppl is not None and len(valid_ppl) > 0:
            return valid_ppl.mean().item()  # Average across positions
        return float('inf')
    
    def mpl(self, logits, targets, mask):
        """
        Calculate in-visit cross-modality perplexity.
        This measures how well we predict tokens using only context from other modalities.
        Uses logits from the second forward pass where same-modality attention is blocked.
        """
        valid_ppl = self.calculate_perplexity(logits, targets, mask)
        if valid_ppl is not None and len(valid_ppl) > 0:
            return valid_ppl.mean().item()  # Average across positions
        return float('inf')
    
    @torch.no_grad()
    def evaluate(self, dataloader, metric_list=['lpl', 'mpl']):
        """
        Evaluate the model's in-visit prediction performance.
        Added debug prints to understand tensor shapes.
        """
        self.model.eval()
        modalities = ['diag', 'drug', 'proc', 'lab']
        results = {f"{metric}_{modality}": [] for metric in metric_list for modality in modalities}

        for batch in tqdm(dataloader, desc="Evaluating"):
            batch = {k: v.to(self.device) for k, v in batch.items()}
            
            # # Debug prints for understanding tensor shapes
            # print("Shapes:")
            # print(f"input_ids shape: {batch['input_ids'].shape}")
            # print(f"attention_mask shape: {batch['attention_mask'].shape}")
            # print(f"modality_mask shape: {batch['modality_mask'].shape}")
            # print(f"modality_indices shape: {batch['modality_indices'].shape}")
            
            # First forward pass (regular) - use for LPL
            predictions, _ = self.model(
                input_ids=batch['input_ids'],
                seq_mask=batch['attention_mask'],
                modality_mask=batch['modality_mask'],
                modality_indices=batch['modality_indices']
            )

            # If MPL is needed, do second forward pass with cross-modal mask
            if 'mpl' in metric_list:
                # Create cross-modal mask that blocks same-modality attention
                cross_modal_mask = batch['modality_mask'].clone()
                # print(f"cross_modal_mask shape: {cross_modal_mask.shape}")
                
                modality_to_idx = getattr(self.model, 'modality_to_idx', {
                    'diag': 0,
                    'drug': 1,
                    'proc': 2,
                    'lab': 3
                })
                
                for modality in modalities:
                    modality_idx = modality_to_idx[modality]
                    # Print shape before masking
                    modality_mask = (batch['modality_indices'] == modality_idx)
                    # print(f"modality_mask shape for {modality}: {modality_mask.shape}")
                    
                    # Try different masking approach based on actual shapes
                    if len(cross_modal_mask.shape) == 3:  # [batch, seq_len, seq_len]
                        mask_2d = modality_mask.unsqueeze(-1).expand(-1, -1, modality_mask.size(1))
                        cross_modal_mask = cross_modal_mask * (~(mask_2d & mask_2d.transpose(1, 2))).float()
                
                predictions_mpl, _ = self.model(
                    input_ids=batch['input_ids'],
                    seq_mask=batch['attention_mask'],
                    modality_mask=cross_modal_mask,
                    modality_indices=batch['modality_indices']
                )

            # Calculate metrics for each modality
            for modality in modalities:
                pred_dict = predictions[modality]
                logits = pred_dict['logits']
                valid_mask = pred_dict['mask']

                if valid_mask.any():
                    targets = batch['input_ids']
                    vocab_offset = {
                        'diag': len(self.model.config.special_tokens),
                        'drug': len(self.model.config.special_tokens) + self.model.config.diag_vocab_size,
                        'proc': len(self.model.config.special_tokens) + self.model.config.diag_vocab_size + self.model.config.drug_vocab_size,
                        'lab': len(self.model.config.special_tokens) + self.model.config.diag_vocab_size + self.model.config.drug_vocab_size + self.model.config.proc_vocab_size
                    }
                    modality_targets = targets - vocab_offset[modality]

                    if 'lpl' in metric_list:
                        lpl_value = self.lpl(logits, modality_targets, valid_mask)
                        if not math.isinf(lpl_value):
                            results[f'lpl_{modality}'].append(lpl_value)

                    if 'mpl' in metric_list:
                        mpl_pred_dict = predictions_mpl[modality]
                        mpl_logits = mpl_pred_dict['logits']
                        mpl_mask = mpl_pred_dict['mask']
                        
                        mpl_value = self.mpl(mpl_logits, modality_targets, mpl_mask)
                        if not math.isinf(mpl_value):
                            results[f'mpl_{modality}'].append(mpl_value)

            # Break after first batch to see prints
            break

        # Aggregate results 
        final_results = {}
        for metric_name, values in results.items():
            if values:
                final_results[metric_name] = np.mean(values)
            else:
                warnings.warn(f'No valid values found for {metric_name}')
                final_results[metric_name] = float('inf')

        return final_results


    # def mpl(self, logits, targets, mask):
    #     """
    #     Calculate in-visit cross-modality perplexity.
    #     This measures how well we predict tokens with context from other modalities.
    #     """
    #     valid_ppl = self.calculate_perplexity(logits, targets, mask)
    #     if valid_ppl is not None and len(valid_ppl) > 0:
    #         # For MPL we calculate the perplexity when predicting with multi-modal context
    #         return valid_ppl.mean().item()
    #     return float('inf')

    # @torch.no_grad()
    # def evaluate(self, dataloader, metric_list=['lpl', 'mpl']):
    #     """
    #     Evaluate the model's in-visit prediction performance.
    #     Returns metrics per modality, aggregated across all visits in the dataset.
    #     """
    #     self.model.eval()
    #     modalities = ['diag', 'drug', 'proc', 'lab']
    #     results = {f"{metric}_{modality}": [] for metric in metric_list for modality in modalities}

    #     for batch in tqdm(dataloader, desc="Evaluating"):
    #         # Move batch to device
    #         batch = {k: v.to(self.device) for k, v in batch.items()}
            
    #         # Get model predictions
    #         predictions, _ = self.model(
    #             input_ids=batch['input_ids'],
    #             seq_mask=batch['attention_mask'],
    #             modality_mask=batch['modality_mask'],
    #             modality_indices=batch['modality_indices']
    #         )

    #         # Calculate metrics for each modality
    #         for modality in modalities:
    #             pred_dict = predictions[modality]
    #             logits = pred_dict['logits']
    #             valid_mask = pred_dict['mask']

    #             if valid_mask.any():
    #                 # Adjust targets for modality-specific vocabulary
    #                 targets = batch['input_ids']
    #                 vocab_offset = {
    #                     'diag': len(self.model.config.special_tokens),
    #                     'drug': len(self.model.config.special_tokens) + self.model.config.diag_vocab_size,
    #                     'proc': len(self.model.config.special_tokens) + self.model.config.diag_vocab_size + self.model.config.drug_vocab_size,
    #                     'lab': len(self.model.config.special_tokens) + self.model.config.diag_vocab_size + self.model.config.drug_vocab_size + self.model.config.proc_vocab_size
    #                 }
    #                 modality_targets = targets - vocab_offset[modality]

    #                 if 'lpl' in metric_list:
    #                     lpl_value = self.lpl(logits, modality_targets, valid_mask)
    #                     if not math.isinf(lpl_value):
    #                         results[f'lpl_{modality}'].append(lpl_value)

    #                 if 'mpl' in metric_list:
    #                     mpl_value = self.mpl(logits, modality_targets, valid_mask)
    #                     if not math.isinf(mpl_value):
    #                         results[f'mpl_{modality}'].append(mpl_value)

    #     # Aggregate results 
    #     final_results = {}
    #     for metric_name, values in results.items():
    #         if values:
    #             final_results[metric_name] = np.mean(values)
    #         else:
    #             warnings.warn(f'No valid values found for {metric_name}')
    #             final_results[metric_name] = float('inf')

    #     return final_results