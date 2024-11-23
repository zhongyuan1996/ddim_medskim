import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from transformers import get_linear_schedule_with_warmup
import logging
import os
import random
import numpy as np
from tqdm import tqdm
from pathlib import Path
import time
from datetime import datetime
import torch.nn.functional as F

from seqStyleDataset import EHRDataloader, create_dataloader
from config_mimic3 import MIMIC3_ModelConfig, MIMIC3_TrainingConfig
from config_mimic4_icd9 import MIMIC4_icd9_ModelConfig, MIMIC4_icd9_TrainingConfig
from model import EHRModel
from evaluator import InVisitEvaluator
import math
import os
import argparse
# os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

class FocalLoss(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.alpha = config.alpha
        self.gamma = config.gamma
        
    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1-pt)**self.gamma * ce_loss
        return focal_loss.mean()

class Runner:
    def __init__(self, model_config: MIMIC3_ModelConfig or MIMIC4_icd9_ModelConfig, training_config:MIMIC3_TrainingConfig or MIMIC4_icd9_TrainingConfig, resume_from=None):
        self.model_config = model_config
        self.training_config = training_config
        self.current_step = 0
        self.resume_from = resume_from

        
        # Set random seeds
        self.set_seed(training_config.seed)
        
        # Setup logging
        self.setup_logging()
        
        # Initialize model and move to device
        self.device = torch.device(training_config.device)
        self.model = EHRModel(model_config).to(self.device)
        
        # Setup optimizer
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=training_config.learning_rate,
            betas=training_config.adam_betas,
            eps=training_config.adam_epsilon,
            weight_decay=training_config.weight_decay
        )
        
        # Initialize tensorboard
        self.writer = SummaryWriter(
            os.path.join(training_config.tensorboard_dir, training_config.experiment_name)
        )
        
        # Setup mixed precision if enabled
        self.scaler = torch.amp.GradScaler('cuda') if training_config.use_amp else None
        
        # Track best model
        self.best_val_loss = float('inf')
        self.best_median_lpl = float('inf')
        self.best_median_mpl = float('inf')
        self.patience_counter = 0
        self.start_epoch = 0

        if resume_from:
            self.load_checkpoint(resume_from)
            
            # Extract epoch number from checkpoint filename if it's a periodic checkpoint
            if 'checkpoint_epoch_' in resume_from:
                try:
                    self.start_epoch = int(resume_from.split('_')[-1].split('.')[0]) + 1
                except ValueError:
                    logging.warning("Could not determine epoch number from checkpoint filename")
            
            logging.info(f"Resuming training from epoch {self.start_epoch}")
        
    def setup_logging(self):
        """Setup logging configuration"""
        # Ensure the log directory exists
        log_dir = Path(self.training_config.log_dir)
        log_dir.mkdir(parents=True, exist_ok=True)
        
        # Set up logging
        logging.basicConfig(
            format='%(asctime)s - %(levelname)s - %(message)s',
            level=logging.INFO,
            handlers=[
                logging.FileHandler(log_dir / f"{self.training_config.experiment_name}.log"),
                logging.StreamHandler()
            ]
        )
        
    def set_seed(self, seed):
        """Set random seeds for reproducibility"""
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    
    def sample_timesteps(self, batch_size):
        """Sample random timesteps for each modality"""
        return {
            'diag': torch.rand(batch_size, device=self.device),
            'drug': torch.rand(batch_size, device=self.device),
            'proc': torch.rand(batch_size, device=self.device),
            'lab': torch.rand(batch_size, device=self.device)
        }
    
    def create_dataloaders(self):
        """
        Create train and validation dataloaders based on config type
        Handles both MIMIC-III and MIMIC-IV ICD9 configurations
        """
        if self.training_config.debug:
            # Use toy dataset for both train and val in debug mode
            if isinstance(self.training_config, MIMIC3_TrainingConfig):
                data_path = self.training_config.mimiciii_toy_path
                code_to_index_path = self.training_config.mimiciii_mapping_path
            elif isinstance(self.training_config, MIMIC4_icd9_TrainingConfig):
                data_path = self.training_config.mimiciv_toy_path
                code_to_index_path = self.training_config.mimiciv_mapping_path
            else:
                raise ValueError("Unsupported training configuration type")

            train_loader = create_dataloader(
                data_path=data_path,
                code_to_index_path=code_to_index_path,
                batch_size=self.training_config.batch_size,
                max_seq_length=self.model_config.max_position_embeddings,
                num_workers=self.training_config.num_workers,
                mask_prob=self.training_config.mask_prob
            )
            val_loader = train_loader
            test_loader = train_loader
        
        else:
            # Regular training mode
            if isinstance(self.training_config, MIMIC3_TrainingConfig):
                train_loader = create_dataloader(
                    data_path=self.training_config.mimiciii_train_path,
                    code_to_index_path=self.training_config.mimiciii_mapping_path,
                    batch_size=self.training_config.batch_size,
                    max_seq_length=self.model_config.max_position_embeddings,
                    num_workers=self.training_config.num_workers,
                    mask_prob=self.training_config.mask_prob
                )
                val_loader = create_dataloader(
                    data_path=self.training_config.mimiciii_val_path,
                    code_to_index_path=self.training_config.mimiciii_mapping_path,
                    batch_size=self.training_config.batch_size,
                    max_seq_length=self.model_config.max_position_embeddings,
                    num_workers=self.training_config.num_workers,
                    mask_prob=self.training_config.mask_prob
                )
                test_loader = create_dataloader(
                    data_path=self.training_config.mimiciii_test_path,
                    code_to_index_path=self.training_config.mimiciii_mapping_path,
                    batch_size=self.training_config.batch_size,
                    max_seq_length=self.model_config.max_position_embeddings,
                    num_workers=self.training_config.num_workers,
                    mask_prob=self.training_config.mask_prob
                )
            
            elif isinstance(self.training_config, MIMIC4_icd9_TrainingConfig):
                train_loader = create_dataloader(
                    data_path=self.training_config.mimiciv_train_path,
                    code_to_index_path=self.training_config.mimiciv_mapping_path,
                    batch_size=self.training_config.batch_size,
                    max_seq_length=self.model_config.max_position_embeddings,
                    num_workers=self.training_config.num_workers,
                    mask_prob=self.training_config.mask_prob
                )
                val_loader = create_dataloader(
                    data_path=self.training_config.mimiciv_val_path,
                    code_to_index_path=self.training_config.mimiciv_mapping_path,
                    batch_size=self.training_config.batch_size,
                    max_seq_length=self.model_config.max_position_embeddings,
                    num_workers=self.training_config.num_workers,
                    mask_prob=self.training_config.mask_prob
                )
                test_loader = create_dataloader(
                    data_path=self.training_config.mimiciv_test_path,
                    code_to_index_path=self.training_config.mimiciv_mapping_path,
                    batch_size=self.training_config.batch_size,
                    max_seq_length=self.model_config.max_position_embeddings,
                    num_workers=self.training_config.num_workers,
                    mask_prob=self.training_config.mask_prob
                )
            
            else:
                raise ValueError("Unsupported training configuration type")

        return train_loader, val_loader, test_loader

    # def create_dataloaders(self):
    #     """Create train and validation dataloaders"""
    #     if self.training_config.debug:
    #         # Use toy dataset for both train and val in debug mode
    #         train_loader = create_dataloader(
    #             data_path=self.training_config.mimiciii_toy_path,
    #             code_to_index_path=self.training_config.mimiciii_mapping_path,
    #             batch_size=self.training_config.batch_size,
    #             max_seq_length=self.model_config.max_position_embeddings,
    #             num_workers=self.training_config.num_workers,
    #             mask_prob=self.training_config.mask_prob
    #         )
    #         val_loader = train_loader
    #         test_loader = train_loader
    #     else:
    #         train_loader = create_dataloader(
    #             data_path=self.training_config.mimiciii_train_path,
    #             code_to_index_path=self.training_config.mimiciii_mapping_path,
    #             batch_size=self.training_config.batch_size,
    #             max_seq_length=self.model_config.max_position_embeddings,
    #             num_workers=self.training_config.num_workers,
    #             mask_prob=self.training_config.mask_prob
    #         )
    #         val_loader = create_dataloader(
    #             data_path=self.training_config.mimiciii_val_path,
    #             code_to_index_path=self.training_config.mimiciii_mapping_path,
    #             batch_size=self.training_config.batch_size,
    #             max_seq_length=self.model_config.max_position_embeddings,
    #             num_workers=self.training_config.num_workers,
    #             mask_prob=self.training_config.mask_prob
    #         )
    #         test_loader = create_dataloader(
    #             data_path=self.training_config.mimiciii_test_path,
    #             code_to_index_path=self.training_config.mimiciii_mapping_path,
    #             batch_size=self.training_config.batch_size,
    #             max_seq_length=self.model_config.max_position_embeddings,
    #             num_workers=self.training_config.num_workers,
    #             mask_prob=self.training_config.mask_prob
    #         )
    #     return train_loader, val_loader, test_loader
    
    def train_epoch(self, dataloader, epoch):
        """Train for one epoch"""
        self.model.train()
        total_loss = 0
        total_diff_loss = 0
        total_pred_loss = 0
        num_batches = len(dataloader)
        focal_loss = FocalLoss(self.training_config)
        
        with tqdm(dataloader, desc=f'Epoch {epoch}', total=num_batches) as pbar:
            for step, batch in enumerate(pbar):
                # Move batch to device
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                modality_mask = batch['modality_mask'].to(self.device)
                modality_indices = batch['modality_indices'].to(self.device)
                mlm_labels = batch['mlm_labels'].to(self.device)
                
                timesteps = self.sample_timesteps(input_ids.size(0))
                
                with torch.amp.autocast(device_type='cuda', enabled=bool(self.scaler)):

                    predictions, diffusion_loss = self.model(
                        input_ids=input_ids,
                        seq_mask=attention_mask,
                        modality_mask=modality_mask,
                        modality_indices=modality_indices,
                        timesteps=timesteps
                    )
                    
                    # Calculate prediction loss for each modality
                    pred_loss = 0
                    for modality in ['diag', 'drug', 'proc', 'lab']:
                        pred_dict = predictions[modality]
                        logits = pred_dict['logits']
                        valid_mask = pred_dict['mask']
                        
                        if valid_mask.sum() > 0:
                            targets = torch.where(
                                mlm_labels != -100,
                                mlm_labels,
                                input_ids
                            )
                            
                            # Get valid indices
                            batch_idx, seq_idx = torch.where(valid_mask)
                            flat_targets = targets[batch_idx, seq_idx]
                            
                            # Filter out special tokens
                            special_token_values = set(self.model_config.special_tokens.values())
                            valid_target_mask = ~torch.tensor([t.item() in special_token_values for t in flat_targets],
                                                            device=flat_targets.device)
                            
                            if valid_target_mask.any():
                                # Filter valid indices
                                valid_batch_idx = batch_idx[valid_target_mask]
                                valid_seq_idx = seq_idx[valid_target_mask]
                                flat_logits = logits[valid_batch_idx, valid_seq_idx]
                                flat_targets = flat_targets[valid_target_mask]
                                
                                # Print shapes and sample data for debugging
                                # print(f"Modality: {modality}")
                                # print(f"Valid batch indices shape: {valid_batch_idx.shape}, seq indices shape: {valid_seq_idx.shape}")
                                # print(f"Flat logits shape: {flat_logits.shape}")
                                # print(f"Flat targets shape: {flat_targets.shape}")
                                # print(f"Flat targets (sample): {flat_targets[:31].tolist()}")
                                
                                # Remap targets to modality-specific vocabulary
                                vocab_offset = {
                                    'diag': len(self.model_config.special_tokens),
                                    'drug': len(self.model_config.special_tokens) + self.model_config.diag_vocab_size,
                                    'proc': len(self.model_config.special_tokens) + self.model_config.diag_vocab_size + self.model_config.drug_vocab_size,
                                    'lab': len(self.model_config.special_tokens) + self.model_config.diag_vocab_size + self.model_config.drug_vocab_size + self.model_config.proc_vocab_size
                                }
                                
                                # Print the vocabulary offset for the current modality
                                # print(f"Vocabulary offset for {modality}: {vocab_offset[modality]}")
                                
                                # Subtract offset to get modality-specific indices
                                flat_targets = flat_targets - vocab_offset[modality]
                                
                                # Debug target value range
                                # print(f"Flat targets after offset (sample): {flat_targets[:31].tolist()}")
                                # print(f"Flat targets range: {flat_targets.min().item()} to {flat_targets.max().item()}")
                                # print(f"Logits size (-1): {logits.size(-1)}")
                                
                                # Verify target values are within vocabulary range
                                assert (flat_targets >= 0).all() and (flat_targets < logits.size(-1)).all(), \
                                    f"Target values outside valid range for {modality}. " \
                                    f"Flat targets range: {flat_targets.min().item()} to {flat_targets.max().item()}, " \
                                    f"Expected range: [0, {logits.size(-1) - 1}]"
                                
                                # Compute prediction loss
                                pred_loss += focal_loss(flat_logits, flat_targets)
                            else:
                                # Log when no valid target exists
                                print(f"No valid targets for modality {modality}. Skipping loss computation.")

                    
                    weighted_pred_loss = pred_loss * self.training_config.pred_weight
                    weighted_diff_loss = diffusion_loss * self.training_config.diff_weight
                    loss = weighted_pred_loss + weighted_diff_loss
                
                if self.scaler:
                    scaled_loss = self.scaler.scale(loss / self.training_config.gradient_accumulation_steps)
                    scaled_loss.backward()
                else:
                    (loss / self.training_config.gradient_accumulation_steps).backward()
                
                if (step + 1) % self.training_config.gradient_accumulation_steps == 0:
                    if self.scaler:
                        self.scaler.unscale_(self.optimizer)
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.training_config.max_grad_norm)
                        self.scaler.step(self.optimizer)
                        self.scaler.update()
                    else:
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.training_config.max_grad_norm)
                        self.optimizer.step()
                    
                    self.optimizer.zero_grad()
                
                # Store weighted losses
                total_loss += loss.item()
                total_diff_loss += weighted_diff_loss.item()
                total_pred_loss += weighted_pred_loss.item()
                
                pbar.set_postfix({
                    'loss': loss.item(),
                    'diff_loss': weighted_diff_loss.item(),
                    'pred_loss': weighted_pred_loss.item()
                })
                
                if step % self.training_config.log_steps == 0:
                    self.writer.add_scalar('train/loss', loss.item(), epoch * num_batches + step)
                    self.writer.add_scalar('train/diffusion_loss', weighted_diff_loss.item(), epoch * num_batches + step)
                    self.writer.add_scalar('train/prediction_loss', weighted_pred_loss.item(), epoch * num_batches + step)
        self.current_step += 1
        return {
            'loss': total_loss / num_batches,
            'diff_loss': total_diff_loss / num_batches,
            'pred_loss': total_pred_loss / num_batches
        }
    
    @torch.no_grad()
    def evaluate(self, dataloader):
        """Evaluate the model using InVisitEvaluator"""
        evaluator = InVisitEvaluator(self.model, self.device)
        metrics = evaluator.evaluate(dataloader)
        
        # Log each metric to tensorboard
        for metric_name, value in metrics.items():
            if not math.isinf(value):
                self.writer.add_scalar(f'val/{metric_name}', value, self.current_step)
        
        # Split metrics into lpl and mpl for each modality
        lpl_metrics = {
            modality: metrics[f'lpl_{modality}'] 
            for modality in ['diag', 'drug', 'proc', 'lab']
        }
        mpl_metrics = {
            modality: metrics[f'mpl_{modality}']
            for modality in ['diag', 'drug', 'proc', 'lab']
        }
        
        # Calculate median scores for logging
        valid_lpl = [v for v in lpl_metrics.values() if not math.isinf(v)]
        valid_mpl = [v for v in mpl_metrics.values() if not math.isinf(v)]
        
        if valid_lpl:
            median_lpl = np.median(valid_lpl)
            self.writer.add_scalar('val/median_lpl', median_lpl, self.current_step)
        
        if valid_mpl:
            median_mpl = np.median(valid_mpl)
            self.writer.add_scalar('val/median_mpl', median_mpl, self.current_step)
        
        return lpl_metrics, mpl_metrics
    
    def train(self):
        """Main training loop"""
        logging.info(f"Starting training from epoch {self.start_epoch}...")
        start_time = time.time()
        
        # Create dataloaders
        train_loader, val_loader, test_loader = self.create_dataloaders()

        # Setup learning rate scheduler
        num_training_steps = len(train_loader) * self.training_config.num_epochs
        scheduler = get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=self.training_config.warmup_steps,
            num_training_steps=num_training_steps
        )
        
        # Skip scheduler steps if resuming
        if self.start_epoch > 0:
            for _ in range(self.start_epoch * len(train_loader)):
                scheduler.step()
        
        for epoch in range(self.training_config.num_epochs):
            # Train one epoch
            train_metrics = self.train_epoch(train_loader, epoch)
            logging.info(f"Epoch {epoch} training metrics: {train_metrics}")
            
            # Evaluate
            lpl_metrics, mpl_metrics = self.evaluate(val_loader)
            
            # Calculate median scores
            curr_median_lpl = np.median([v for v in lpl_metrics.values() if not math.isinf(v)])
            curr_median_mpl = np.median([v for v in mpl_metrics.values() if not math.isinf(v)])
            
            logging.info(f"Epoch {epoch} validation LPL per modality: {lpl_metrics}")
            logging.info(f"Epoch {epoch} validation MPL per modality: {mpl_metrics}")
            logging.info(f"Epoch {epoch} median LPL: {curr_median_lpl:.4f}")
            logging.info(f"Epoch {epoch} median MPL: {curr_median_mpl:.4f}")
            
            # Early stopping check - save if either metric improves
            improved = False
            if curr_median_lpl < self.best_median_lpl:
                self.best_median_lpl = curr_median_lpl
                improved = True
                
            if curr_median_mpl < self.best_median_mpl:
                self.best_median_mpl = curr_median_mpl
                improved = True
                
            if improved:
                self.patience_counter = 0
                self.save_checkpoint('best_model.pt')
            else:
                self.patience_counter += 1
                
            if self.patience_counter >= self.training_config.patience:
                logging.info("Early stopping triggered")
                break
                
            # Save periodic checkpoint
            if (epoch + 1) % self.training_config.save_steps == 0:
                self.save_checkpoint(f'checkpoint_epoch_{epoch}.pt')
            
            # Step the scheduler
            scheduler.step()
        
        training_time = time.time() - start_time
        logging.info(f"Training finished in {training_time:.2f} seconds")

        # Test set evaluation using best model
        logging.info("Starting test set evaluation...")
        
        # Load best model saved during training
        logging.info("Loading best model for test set evaluation...")
        self.load_checkpoint('best_model.pt')
        
        # Ensure model is in eval mode
        self.model.eval()
        
        # Evaluate on test set with no gradients
        logging.info("Evaluating on test set...")
        with torch.no_grad():
            test_lpl_metrics, test_mpl_metrics = self.evaluate(test_loader)
        
        # Calculate and log test set median scores
        test_median_lpl = np.median([v for v in test_lpl_metrics.values() if not math.isinf(v)])
        test_median_mpl = np.median([v for v in test_mpl_metrics.values() if not math.isinf(v)])
        
        logging.info("Test Set Results:")
        logging.info(f"Test LPL per modality: {test_lpl_metrics}")
        logging.info(f"Test MPL per modality: {test_mpl_metrics}")
        logging.info(f"Test median LPL: {test_median_lpl:.4f}")
        logging.info(f"Test median MPL: {test_median_mpl:.4f}")
        
        # Save test results
        test_results = {
            'lpl_metrics': test_lpl_metrics,
            'mpl_metrics': test_mpl_metrics,
            'median_lpl': test_median_lpl,
            'median_mpl': test_median_mpl,
            'timestamp': datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        }
        test_results_path = Path(self.training_config.checkpoint_dir) / 'test_results.pt'
        torch.save(test_results, test_results_path)
        logging.info(f"Saved test results to {test_results_path}")
        
        # Add test metrics to tensorboard
        self.writer.add_scalar('test/median_lpl', test_median_lpl, 0)
        self.writer.add_scalar('test/median_mpl', test_median_mpl, 0)
        for modality in test_lpl_metrics:
            if not math.isinf(test_lpl_metrics[modality]):
                self.writer.add_scalar(f'test/lpl_{modality}', test_lpl_metrics[modality], 0)
            if not math.isinf(test_mpl_metrics[modality]):
                self.writer.add_scalar(f'test/mpl_{modality}', test_mpl_metrics[modality], 0)

        self.writer.close()
        logging.info("Training and evaluation completed.")
        
    def save_checkpoint(self, filename):
        """Save model checkpoint with comprehensive training state"""
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_val_loss': self.best_val_loss,
            'current_epoch': self.current_step,
            'best_median_lpl': getattr(self, 'best_median_lpl', float('inf')),
            'best_median_mpl': getattr(self, 'best_median_mpl', float('inf')),
            'patience_counter': getattr(self, 'patience_counter', 0),
            'scaler': self.scaler.state_dict() if self.scaler else None,
            'timestamp': datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        }
        
        save_path = Path(self.training_config.checkpoint_dir) / filename
        save_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(checkpoint, save_path)
        logging.info(f"Saved checkpoint to {save_path}")
    
    def load_checkpoint(self, filename):
        """Load model checkpoint and restore training state"""
        load_path = Path(self.training_config.checkpoint_dir) / filename
        checkpoint = torch.load(load_path, map_location=self.device)
        
        # Load model and optimizer states
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        # Restore training state
        self.current_step = checkpoint.get('epoch', 0)
        self.best_val_loss = checkpoint.get('best_val_loss', float('inf'))
        self.best_median_lpl = checkpoint.get('best_median_lpl', float('inf'))
        self.best_median_mpl = checkpoint.get('best_median_mpl', float('inf'))
        self.patience_counter = checkpoint.get('patience_counter', 0)
        
        # Restore scaler if it exists
        if self.scaler and checkpoint.get('scaler'):
            self.scaler.load_state_dict(checkpoint['scaler'])
        
        logging.info(f"Loaded checkpoint from {load_path}")

def get_config(dataset_name: str):
    """
    Get the appropriate model and training configs based on dataset name
    
    Args:
        dataset_name: Name of the dataset ('mimic3' or 'mimic4_icd9')
    
    Returns:
        tuple: (model_config, training_config)
    """
    if dataset_name.lower() == 'mimic3':
        return MIMIC3_ModelConfig(), MIMIC3_TrainingConfig()
    elif dataset_name.lower() == 'mimic4_icd9':
        return MIMIC4_icd9_ModelConfig(), MIMIC4_icd9_TrainingConfig()
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}. Choose from: 'mimic3', 'mimic4_icd9'")

if __name__ == "__main__":
    # Setup argument parser
    parser = argparse.ArgumentParser(description='Train EHR model on specified dataset')
    parser.add_argument('--dataset', type=str, 
                      choices=['mimic3', 'mimic4_icd9'],
                      default='mimic3',
                      help='Dataset to use for training (mimic3 or mimic4_icd9)')
    parser.add_argument('--resume_from', type=str, default=None,
                      help='Path to checkpoint to resume training from')
    
    # Parse arguments
    args = parser.parse_args()
    
    # Get appropriate configs based on dataset name
    model_config, training_config = get_config(args.dataset)
    
    # Print training setup
    print(f"\nTraining Setup:")
    print(f"Dataset: {args.dataset}")
    print(f"Resume from: {args.resume_from if args.resume_from else 'No checkpoint'}")
    print(f"Experiment name: {training_config.experiment_name}")
    print(f"Debug mode: {'Enabled' if training_config.debug else 'Disabled'}")
    
    # Create runner
    runner = Runner(model_config, training_config, resume_from=args.resume_from)
    
    # Start training
    runner.train()
