import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import TransformerEncoderLayer, TransformerEncoder
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.metrics import roc_auc_score
import numpy as np
from tqdm import tqdm
import math
from dataclasses import dataclass
from typing import Optional, List
import warnings

device = "cuda" if torch.cuda.is_available() else "cpu"

@dataclass
class ModelConfig:
    """Configuration for CTR Model"""
    num_items: int
    embed_dim: int = 64
    k: int = 16
    num_transformer_layers: int = 2
    num_heads: int = 4
    num_cross_layers: int = 3
    deep_layers: List[int] = None
    dropout: float = 0.2
    learning_rate: float = 5e-4
    weight_decay: float = 1e-5
    max_seq_length: int = 100
    
    def __post_init__(self):
        if self.deep_layers is None:
            self.deep_layers = [1024, 512, 256]


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding for Transformer"""
    
    def __init__(self, d_model, max_len=5000, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        # Create positional encoding matrix
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        """
        Args:
            x: (B, seq_len, d_model)
        Returns:
            x with positional encoding added
        """
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class EmbeddingLayer(nn.Module):
    """Concatenates learnable embeddings with frozen multimodal embeddings."""
    
    def __init__(self, num_items, embed_dim, frozen_embeddings):
        super().__init__()
        
        # Validate inputs
        assert frozen_embeddings.shape[0] == num_items, \
            f"Frozen embeddings ({frozen_embeddings.shape[0]}) must match num_items ({num_items})"
        
        frozen_dim = frozen_embeddings.shape[1]
        
        # Frozen (not trainable) - normalize for stable gradients
        frozen_normalized = frozen_embeddings / (np.linalg.norm(frozen_embeddings, axis=1, keepdims=True) + 1e-8)
        self.register_buffer('frozen_emb', torch.from_numpy(frozen_normalized).float())
        
        # Learnable
        self.item_emb = nn.Embedding(num_items, embed_dim, padding_idx=0)
        
        # Initialize with Xavier/Glorot
        nn.init.xavier_uniform_(self.item_emb.weight)
        self.item_emb.weight.data[0] = 0  # Padding
        
        # Layer norm for learnable embeddings to match frozen scale
        self.layer_norm = nn.LayerNorm(embed_dim)
        
        # Final embedding dim
        self.final_dim = frozen_dim + embed_dim
        self.frozen_dim = frozen_dim
        self.learnable_dim = embed_dim
        
        print(f"  Embedding: frozen_dim={frozen_dim}, learnable_dim={embed_dim}, final_dim={self.final_dim}")
    
    def forward(self, item_ids):
        """
        Args:
            item_ids: (...,) tensor of item indices
            
        Returns:
            embeddings: (..., final_dim) normalized concatenated embeddings
        """
        frozen = self.frozen_emb[item_ids]  # Already normalized
        learnable = self.item_emb(item_ids)
        learnable = self.layer_norm(learnable)  # Normalize learnable embeddings
        
        # Concatenate
        final_emb = torch.cat([frozen, learnable], dim=-1)
        return final_emb


class SequentialLearning(nn.Module):
    """
    Transformer-based sequential learning with positional encoding.
    """
    
    def __init__(self, item_embed_dim, k=16, num_layers=2, num_heads=4, 
                 dropout=0.2, max_seq_length=100):
        super().__init__()
        self.k = k
        self.item_embed_dim = item_embed_dim
        
        # Input dimension: item + target concatenated
        d_model = item_embed_dim * 2
        
        # Positional encoding
        self.pos_encoder = PositionalEncoding(d_model, max_len=max_seq_length, dropout=dropout)
        
        # Transformer encoder
        encoder_layer = TransformerEncoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dim_feedforward=512,
            batch_first=True,
            dropout=dropout,
            activation='gelu',  # GELU often performs better than ReLU
            norm_first=True  # Pre-LN for better training stability
        )
        self.transformer = TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Layer norm after transformer
        self.output_norm = nn.LayerNorm(d_model)
        
        # Output: k items flattened + max pool + mean pool
        self.output_dim = k * d_model + d_model * 2  # Added mean pooling
        
        print(f"  Sequential: d_model={d_model}, k={k}, output_dim={self.output_dim}")
    
    def forward(self, item_ids, item_embeds, target_emb):
        """
        Args:
            item_ids: (B, seq_len) - for masking
            item_embeds: (B, seq_len, item_embed_dim)
            target_emb: (B, item_embed_dim)
            
        Returns:
            S_o: (B, output_dim)
        """
        batch_size, seq_len = item_ids.shape
        
        # Validate sequence length
        if seq_len > self.k:
            # Concatenate target with each sequence item
            target_expanded = target_emb.unsqueeze(1).expand(-1, seq_len, -1)
            seq_input = torch.cat([item_embeds, target_expanded], dim=-1)
            
            # Add positional encoding
            seq_input = self.pos_encoder(seq_input)
            
            # Create padding mask
            padding_mask = (item_ids == 0)
            
            # Transformer
            S = self.transformer(seq_input, src_key_padding_mask=padding_mask)
            S = self.output_norm(S)
            
            # Latest k items (ensure we have at least k items)
            S_k = S[:, -self.k:, :].reshape(batch_size, -1)
            
            # Max pooling
            S_masked = S.clone()
            S_masked[padding_mask] = float('-inf')
            S_max = S_masked.max(dim=1)[0]
            
            # Mean pooling (more stable than max)
            S_masked_mean = S.clone()
            S_masked_mean[padding_mask] = 0
            seq_lengths = (~padding_mask).sum(dim=1, keepdim=True).float()
            S_mean = S_masked_mean.sum(dim=1) / seq_lengths.clamp(min=1)
            
            # Concatenate all representations
            S_o = torch.cat([S_k, S_max, S_mean], dim=1)
        else:
            # Fallback for short sequences
            warnings.warn(f"Sequence length {seq_len} is <= k={self.k}, using simple aggregation")
            S_o = torch.zeros(batch_size, self.output_dim, device=item_ids.device)
        
        return S_o


class DCNv2(nn.Module):
    """
    Deep & Cross Network v2 with batch normalization.
    """
    
    def __init__(self, input_dim, num_cross_layers=3, deep_layers=None, dropout=0.2):
        super().__init__()
        
        if deep_layers is None:
            deep_layers = [1024, 512, 256]
        
        # Input batch norm
        self.input_bn = nn.BatchNorm1d(input_dim)
        
        # Cross layers with batch norm
        self.cross_layers = nn.ModuleList([
            nn.Linear(input_dim, input_dim, bias=True)
            for _ in range(num_cross_layers)
        ])
        self.cross_bns = nn.ModuleList([
            nn.BatchNorm1d(input_dim)
            for _ in range(num_cross_layers)
        ])
        
        # Deep network with batch norm
        layers = []
        in_dim = input_dim
        for h_dim in deep_layers:
            layers.extend([
                nn.Linear(in_dim, h_dim),
                nn.BatchNorm1d(h_dim),
                nn.GELU(),
                nn.Dropout(dropout)
            ])
            in_dim = h_dim
        self.deep_net = nn.Sequential(*layers)
        
        # Output concatenation
        self.output_dim = input_dim + deep_layers[-1]
        
        print(f"  DCNv2: input_dim={input_dim}, cross_layers={num_cross_layers}, output_dim={self.output_dim}")
    
    def forward(self, x0):
        """
        Args:
            x0: (B, input_dim)
            
        Returns:
            output: (B, output_dim)
        """
        # Input normalization
        x0 = self.input_bn(x0)
        
        # Cross path with batch norm
        x_cross = x0
        for layer, bn in zip(self.cross_layers, self.cross_bns):
            x_cross = x0 * layer(x_cross) + x_cross
            x_cross = bn(x_cross)
        
        # Deep path
        x_deep = self.deep_net(x0)
        
        # Concatenate
        return torch.cat([x_cross, x_deep], dim=1)


class CTRModelImproved(nn.Module):
    """Improved CTR Model with better stability and features."""
    
    def __init__(self, config: ModelConfig, frozen_embeddings: np.ndarray):
        super().__init__()
        
        self.config = config
        
        # Validate frozen embeddings
        if frozen_embeddings.shape[0] != config.num_items:
            raise ValueError(
                f"Frozen embeddings shape {frozen_embeddings.shape[0]} "
                f"doesn't match num_items {config.num_items}"
            )
        
        print(f"\n{'='*70}")
        print(f"Initializing Improved CTR Model")
        print(f"{'='*70}")
        
        # 1. Embedding layer
        self.embedding = EmbeddingLayer(
            config.num_items, config.embed_dim, frozen_embeddings
        )
        item_emb_dim = self.embedding.final_dim
        
        # 2. Sequential learning
        self.seq_learning = SequentialLearning(
            item_emb_dim, 
            k=config.k, 
            num_layers=config.num_transformer_layers,
            num_heads=config.num_heads, 
            dropout=config.dropout,
            max_seq_length=config.max_seq_length
        )
        
        # 3. Side features with better processing
        self.side_proj = nn.Sequential(
            nn.Linear(2, 64),
            nn.BatchNorm1d(64),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(64, 32)
        )
        
        # 4. DCNv2
        dcn_input_dim = item_emb_dim + self.seq_learning.output_dim + 32
        self.dcn = DCNv2(
            dcn_input_dim, 
            num_cross_layers=config.num_cross_layers,
            deep_layers=config.deep_layers, 
            dropout=config.dropout
        )
        
        # 5. Prediction head with batch norm
        self.pred_head = nn.Sequential(
            nn.Linear(self.dcn.output_dim, 128),
            nn.BatchNorm1d(128),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(64, 1)
        )
        
        self.to(device)
        
        # Optimizer with better defaults
        self.optimizer = Adam(
            self.parameters(), 
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
            betas=(0.9, 0.999),
            eps=1e-8
        )
        self.scheduler = ReduceLROnPlateau(
            self.optimizer, 
            mode='max', 
            factor=0.5, 
            patience=3,
            verbose=True
        )
        self.criterion = None
        
        # Info
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.parameters())
        print(f"\n  Trainable params: {trainable:,} / {total:,}")
        print(f"  Learning rate: {config.learning_rate}")
        print(f"  Weight decay: {config.weight_decay}")
        print(f"{'='*70}\n")
    
    def forward(self, batch):
        """
        Args:
            batch: dict with keys:
                'item_seqs': (B, seq_len)
                'item_ids': (B,)
                'likes_levels': (B,)
                'views_levels': (B,)
        """
        # Validate inputs
        item_seqs = batch["item_seqs"]
        item_ids = batch["item_ids"]
        likes = batch["likes_levels"].float()
        views = batch["views_levels"].float()
        
        batch_size = item_ids.size(0)
        
        # Get embeddings
        seq_embs = self.embedding(item_seqs)
        target_emb = self.embedding(item_ids)
        
        # Sequential learning
        S_o = self.seq_learning(item_seqs, seq_embs, target_emb)
        
        # Side features (normalize to [0, 1] range for stability)
        side = torch.stack([
            torch.log1p(likes) / 10.0,  # Log-scale normalization
            torch.log1p(views) / 10.0
        ], dim=1)
        side_emb = self.side_proj(side)
        
        # Concatenate
        x0 = torch.cat([target_emb, S_o, side_emb], dim=1)
        
        # Feature interaction
        x_inter = self.dcn(x0)
        
        # Prediction
        logits = self.pred_head(x_inter).squeeze(-1)
        
        return logits
    
    def compute_auc(self, preds, labels):
        """Compute AUC score"""
        preds_np = torch.sigmoid(preds).detach().cpu().numpy()
        labels_np = labels.cpu().numpy()
        try:
            return roc_auc_score(labels_np, preds_np)
        except ValueError:
            return None
    
    def train_epoch(self, loader):
        """Train for one epoch"""
        self.train()
        total_loss = 0
        all_preds = []
        all_labels = []
        
        pbar = tqdm(loader, desc='Training')
        for batch in pbar:
            batch = {k: v.to(device) for k, v in batch.items()}
            labels = batch["labels"].float()
            
            # Forward
            preds = self.forward(batch)
            loss = self.criterion(preds, labels)
            
            # Backward
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.parameters(), 1.0)
            self.optimizer.step()
            
            total_loss += loss.item()
            all_preds.append(preds.detach().cpu())
            all_labels.append(labels.cpu())
            
            # Update progress bar
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        avg_loss = total_loss / len(loader)
        auc = self.compute_auc(torch.cat(all_preds), torch.cat(all_labels))
        return avg_loss, auc
    
    def validate(self, loader):
        """Validate the model"""
        self.eval()
        total_loss = 0
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for batch in tqdm(loader, desc='Validating'):
                batch = {k: v.to(device) for k, v in batch.items()}
                labels = batch["labels"].float()
                
                preds = self.forward(batch)
                loss = self.criterion(preds, labels)
                
                total_loss += loss.item()
                all_preds.append(preds.cpu())
                all_labels.append(labels.cpu())
        
        avg_loss = total_loss / len(loader)
        auc = self.compute_auc(torch.cat(all_preds), torch.cat(all_labels))
        return avg_loss, auc
    
    def fit(self, train_loader, valid_loader, num_epochs=20, save_path=None):
        """
        Train the model with automatic pos_weight calculation
        """
        # Calculate pos_weight from training data
        print("Calculating class weights from training data...")
        all_labels = []
        for batch in tqdm(train_loader, desc="Scanning labels"):
            all_labels.append(batch["labels"])
        all_labels = torch.cat(all_labels).float()
        
        num_pos = all_labels.sum().item()
        num_neg = len(all_labels) - num_pos
        pos_weight = num_neg / num_pos if num_pos > 0 else 1.0
        
        self.criterion = nn.BCEWithLogitsLoss(
            pos_weight=torch.tensor(pos_weight).to(device)
        )
        
        print(f"\nClass distribution:")
        print(f"  Positive: {num_pos:,} ({100*num_pos/len(all_labels):.2f}%)")
        print(f"  Negative: {num_neg:,} ({100*num_neg/len(all_labels):.2f}%)")
        print(f"  Pos weight: {pos_weight:.4f}\n")
        
        best_auc = 0
        patience = 5
        patience_counter = 0
        
        for epoch in range(num_epochs):
            print(f"\n{'='*70}")
            print(f"EPOCH {epoch+1}/{num_epochs}")
            print(f"{'='*70}")
            
            # Train
            tr_loss, tr_auc = self.train_epoch(train_loader)
            print(f"Train: Loss={tr_loss:.4f}, AUC={tr_auc:.4f}")
            
            # Validate
            val_loss, val_auc = self.validate(valid_loader)
            print(f"Val:   Loss={val_loss:.4f}, AUC={val_auc:.4f}")
            
            gap = tr_auc - val_auc if tr_auc and val_auc else 0
            print(f"Gap:   {gap:.4f}")
            
            # Learning rate scheduling
            if val_auc:
                self.scheduler.step(val_auc)
                print(f"LR:    {self.optimizer.param_groups[0]['lr']:.6f}")
            
            # Save best model
            if val_auc and val_auc > best_auc:
                best_auc = val_auc
                patience_counter = 0
                if save_path:
                    torch.save({
                        'model': self.state_dict(),
                        'optimizer': self.optimizer.state_dict(),
                        'scheduler': self.scheduler.state_dict(),
                        'epoch': epoch,
                        'val_auc': val_auc,
                        'config': self.config
                    }, f"{save_path}_epoch{epoch}_auc{val_auc:.4f}.pth")
                    print(f"✓ Saved checkpoint (AUC: {val_auc:.4f})")
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"\nEarly stopping triggered (patience={patience})")
                    break
        
        print(f"\n{'='*70}")
        print(f"Training completed!")
        print(f"Best Val AUC: {best_auc:.4f}")
        print(f"{'='*70}\n")
        
        return best_auc
    
    @classmethod
    def load_from_checkpoint(cls, checkpoint_path, frozen_embeddings):
        """
        Load model from checkpoint with proper error handling
        """
        checkpoint = torch.load(checkpoint_path, map_location=device)
        
        # Extract config
        if 'config' in checkpoint:
            config = checkpoint['config']
        else:
            # Fallback: reconstruct config from model
            print("Warning: Config not found in checkpoint, using defaults")
            config = ModelConfig(num_items=frozen_embeddings.shape[0])
        
        # Create model
        model = cls(config, frozen_embeddings)
        
        # Load state dict with strict checking
        try:
            model.load_state_dict(checkpoint['model'], strict=True)
            print("✓ Model loaded successfully (strict mode)")
        except RuntimeError as e:
            print(f"Warning: Strict loading failed: {e}")
            print("Attempting to load with strict=False...")
            model.load_state_dict(checkpoint['model'], strict=False)
            print("✓ Model loaded with some mismatches")
        
        # Load optimizer and scheduler if resuming training
        if 'optimizer' in checkpoint:
            model.optimizer.load_state_dict(checkpoint['optimizer'])
        if 'scheduler' in checkpoint:
            model.scheduler.load_state_dict(checkpoint['scheduler'])
        
        print(f"Loaded checkpoint from epoch {checkpoint.get('epoch', 'unknown')}")
        print(f"Checkpoint Val AUC: {checkpoint.get('val_auc', 'unknown')}")
        
        return model, checkpoint.get('epoch', 0)
    



"""
Example usage of the improved CTR model
"""

import torch
from torch.utils.data import DataLoader
import sys
import os

sys.path.append("/kaggle/working/recommender_CTR") 

from task2.dataset.dataset import Task2Dataset, collate_fn
from task2.model_loader import load_item_embeddings_and_tags


def main():
    """
    Example 1: Training a new model from scratch
    """
    print("="*70)
    print("EXAMPLE 1: Training from scratch")
    print("="*70)
    
    # Load your data
    # Adjust paths according to your setup
    train_dataset = Task2Dataset(
        data_path="/kaggle/input/www2025-mmctr-data/MicroLens_1M_MMCTR/MicroLens_1M_x1/train.parquet",
        is_train=True
    )
    valid_dataset = Task2Dataset(
        data_path="/kaggle/input/www2025-mmctr-data/MicroLens_1M_MMCTR/MicroLens_1M_x1/valid.parquet",
        is_train=True
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=128,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=4,
        pin_memory=True
    )
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=256,  # Larger batch for validation
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=4,
        pin_memory=True
    )
    
    # Load frozen embeddings
    embeddings, _, num_items, _ = load_item_embeddings_and_tags(
        item_info_path="/kaggle/working/item_info_with_clip.parquet",
        embedding_source="item_clip_emb_d128"
    )
    
    # Create configuration
    config = ModelConfig(
        num_items=num_items,
        embed_dim=64,
        k=16,
        num_transformer_layers=2,
        num_heads=4,
        num_cross_layers=3,
        deep_layers=[1024, 512, 256],
        dropout=0.2,
        learning_rate=5e-4,
        weight_decay=1e-5,
        max_seq_length=100
    )
    
    # Create model
    model = CTRModelImproved(config, embeddings)
    
    # Train
    best_auc = model.fit(
        train_loader=train_loader,
        valid_loader=valid_loader,
        num_epochs=50,
        save_path="./checkpoints/model"
    )
    
    print(f"\nTraining completed! Best AUC: {best_auc:.4f}")


def resume_training():
    """
    Example 2: Resume training from a checkpoint
    """
    print("="*70)
    print("EXAMPLE 2: Resume training from checkpoint")
    print("="*70)
    
    # Load data (same as above)
    train_loader = ...  # Your data loader
    valid_loader = ...  # Your data loader
    
    # Load frozen embeddings
    embeddings, _, num_items, _ = load_item_embeddings_and_tags(
        item_info_path="/kaggle/working/item_info_with_clip.parquet",
        embedding_source="item_clip_emb_d128"
    )
    
    # Load from checkpoint
    checkpoint_path = None
    model, start_epoch = CTRModelImproved.load_from_checkpoint(
        checkpoint_path=checkpoint_path,
        frozen_embeddings=embeddings
    )
    
    print(f"Resuming from epoch {start_epoch}")
    
    # Continue training
    remaining_epochs = 50 - start_epoch
    if remaining_epochs > 0:
        best_auc = model.fit(
            train_loader=train_loader,
            valid_loader=valid_loader,
            num_epochs=remaining_epochs,
            save_path="./checkpoints/model_resumed"
        )
        print(f"\nTraining completed! Best AUC: {best_auc:.4f}")
    else:
        print("Model already trained for target number of epochs")


def inference_example():
    """
    Example 3: Using the model for inference
    """
    print("="*70)
    print("EXAMPLE 3: Inference")
    print("="*70)
    
    # Load frozen embeddings
    embeddings, _, num_items, _ = load_item_embeddings_and_tags(
        item_info_path="/kaggle/working/item_info_with_clip.parquet",
        embedding_source="item_clip_emb_d128"
    )
    
    # Load model
    checkpoint_path = "./checkpoints/best_model.pth"
    model, _ = CTRModelImproved.load_from_checkpoint(
        checkpoint_path=checkpoint_path,
        frozen_embeddings=embeddings
    )
    
    model.eval()
    
    # Prepare test data
    test_dataset = Task2Dataset(
        data_path="/kaggle/input/www2025-mmctr-data/MicroLens_1M_MMCTR/MicroLens_1M_x1/test.parquet",
        is_train=False
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=512,
        shuffle=False,
        collate_fn=collate_fn
    )
    
    # Generate predictions
    all_predictions = []
    
    with torch.no_grad():
        for batch in test_loader:
            batch = {k: v.to(model.device) for k, v in batch.items()}
            
            # Get predictions
            logits = model(batch)
            probs = torch.sigmoid(logits)
            
            all_predictions.append(probs.cpu().numpy())
    
    # Concatenate all predictions
    predictions = np.concatenate(all_predictions)
    
    print(f"Generated {len(predictions)} predictions")
    print(f"Mean prediction: {predictions.mean():.4f}")
    print(f"Std prediction: {predictions.std():.4f}")
    
    # Save predictions
    np.save("predictions.npy", predictions)
    print("Predictions saved to predictions.npy")


def hyperparameter_tuning_example():
    """
    Example 4: Trying different hyperparameters
    """
    print("="*70)
    print("EXAMPLE 4: Hyperparameter tuning")
    print("="*70)
    
    # Load data
    train_loader = ...  # Your data loader
    valid_loader = ...  # Your data loader
    embeddings = ...  # Your embeddings
    num_items = embeddings.shape[0]
    
    # Try different configurations
    configs = [
        # Config 1: Baseline
        ModelConfig(
            num_items=num_items,
            embed_dim=64,
            k=16,
            num_transformer_layers=2,
            dropout=0.2,
            learning_rate=5e-4
        ),
        # Config 2: Larger embeddings
        ModelConfig(
            num_items=num_items,
            embed_dim=128,
            k=16,
            num_transformer_layers=2,
            dropout=0.2,
            learning_rate=5e-4
        ),
        # Config 3: Deeper transformer
        ModelConfig(
            num_items=num_items,
            embed_dim=64,
            k=16,
            num_transformer_layers=4,
            dropout=0.2,
            learning_rate=3e-4
        ),
        # Config 4: More recent items
        ModelConfig(
            num_items=num_items,
            embed_dim=64,
            k=32,
            num_transformer_layers=2,
            dropout=0.2,
            learning_rate=5e-4
        ),
    ]
    
    results = []
    
    for i, config in enumerate(configs, 1):
        print(f"\n{'='*70}")
        print(f"Training configuration {i}/{len(configs)}")
        print(f"{'='*70}")
        
        model = CTRModelImproved(config, embeddings)
        
        best_auc = model.fit(
            train_loader=train_loader,
            valid_loader=valid_loader,
            num_epochs=20,  # Fewer epochs for tuning
            save_path=f"./checkpoints/config{i}"
        )
        
        results.append({
            'config_id': i,
            'embed_dim': config.embed_dim,
            'k': config.k,
            'num_layers': config.num_transformer_layers,
            'best_auc': best_auc
        })
    
    # Print summary
    print("\n" + "="*70)
    print("HYPERPARAMETER TUNING RESULTS")
    print("="*70)
    for result in results:
        print(f"Config {result['config_id']}: "
              f"embed_dim={result['embed_dim']}, "
              f"k={result['k']}, "
              f"layers={result['num_layers']} -> "
              f"AUC={result['best_auc']:.4f}")
    
    best_result = max(results, key=lambda x: x['best_auc'])
    print(f"\nBest configuration: Config {best_result['config_id']} "
          f"with AUC={best_result['best_auc']:.4f}")


if __name__ == "__main__":
    # Choose which example to run
    import argparse
    
    parser = argparse.ArgumentParser(description="CTR Model Examples")
    parser.add_argument(
        '--mode',
        type=str,
        default='train',
        choices=['train', 'resume', 'inference', 'tune'],
        help='Which example to run'
    )
    args = parser.parse_args()
    
    if args.mode == 'train':
        main()
    elif args.mode == 'resume':
        resume_training()
    elif args.mode == 'inference':
        inference_example()
    elif args.mode == 'tune':
        hyperparameter_tuning_example()