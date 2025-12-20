import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import TransformerEncoderLayer, TransformerEncoder
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.metrics import roc_auc_score
import numpy as np
from tqdm import tqdm

import sys 
sys.path.append("/kaggle/working/recommender_CTR")

device = "cuda" if torch.cuda.is_available() else "cpu"

class EmbeddingLayer(nn.Module):
    """Concatenates learnable embeddings with frozen multimodal."""
    
    def __init__(self, num_items, embed_dim, frozen_embeddings, item_tags, num_tags, tag_embed_dim):
        super().__init__()
        
        # Frozen (not trainable)
        self.register_buffer('frozen_emb', torch.from_numpy(frozen_embeddings).float())
        self.register_buffer('item_tags', torch.from_numpy(item_tags).long())
        
        # Learnable
        self.item_emb = nn.Embedding(num_items, embed_dim, padding_idx=0)
        self.tag_emb = nn.Embedding(num_tags, tag_embed_dim, padding_idx=0)
        
        # Initialize
        nn.init.uniform_(self.item_emb.weight, -0.05, 0.05)
        self.item_emb.weight.data[0] = 0
        
        nn.init.uniform_(self.tag_emb.weight, -0.05, 0.05)
        self.tag_emb.weight.data[0] = 0
        
        self.final_dim = embed_dim + 128 + tag_embed_dim  # frozen + learnable + tag_repr
    
    def forward(self, item_ids):
        """
        Returns concatenation: frozen || learnable_id || tag_repr
        
        Args:
            item_ids: (...,) tensor
            
        Returns:
            embeddings: (..., final_dim)
        """
        frozen = self.frozen_emb[item_ids]  # (..., 128)
        learnable = self.item_emb(item_ids)  # (..., embed_dim)
        
        tags = self.item_tags[item_ids]  # (..., 5)
        tag_repr = self.tag_emb(tags)  # (..., 5, tag_embed_dim)
        tag_repr = tag_repr.mean(dim=-2)  # (..., tag_embed_dim)
        
        final_emb = torch.cat([frozen, learnable, tag_repr], dim=-1)  # (..., final_dim)
        return final_emb

class SequentialLearning(nn.Module):
    """
    Transformer-based sequential learning.
    Input: sequence of item embeddings concatenated with target embedding
    Output: latest k items flattened + max pooled features
    """
    
    def __init__(self, item_embed_dim, k=16, num_layers=2, num_heads=4, dropout=0.2):
        super().__init__()
        self.k = k
        self.item_embed_dim = item_embed_dim
        
        # Transformer encoder
        encoder_layer = TransformerEncoderLayer(
            d_model=item_embed_dim * 2,  # Concatenate item + target
            nhead=num_heads,
            dim_feedforward=512,
            batch_first=True,
            dropout=dropout,
            activation='relu'
        )
        self.transformer = TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Output: k items flattened + max pool
        self.output_dim = k * (item_embed_dim * 2) + (item_embed_dim * 2)
    
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
        
        # Concatenate target with each sequence item
        target_expanded = target_emb.unsqueeze(1).expand(-1, seq_len, -1)  # (B, seq_len, item_embed_dim)
        seq_input = torch.cat([item_embeds, target_expanded], dim=-1)  # (B, seq_len, item_embed_dim*2)
        
        # Create padding mask
        padding_mask = (item_ids == 0)
        
        # Transformer
        S = self.transformer(seq_input, src_key_padding_mask=padding_mask)  # (B, seq_len, item_embed_dim*2)
        
        # Latest k items
        S_k = S[:, -self.k:, :].reshape(batch_size, -1)  # (B, k * item_embed_dim * 2)
        
        # Max pooling
        S_masked = S.clone()
        S_masked[padding_mask] = float('-inf')
        S_max = S_masked.max(dim=1)[0]  # (B, item_embed_dim*2)
        
        # Concatenate
        S_o = torch.cat([S_k, S_max], dim=1)  # (B, output_dim)
        
        return S_o

class DCNv2(nn.Module):
    """
    Deep & Cross Network v2 for feature interaction.
    """
    
    def __init__(self, input_dim, num_cross_layers=3, deep_layers=None, dropout=0.2):
        super().__init__()
        
        if deep_layers is None:
            deep_layers = [1024, 512, 256]
        
        # Cross layers
        self.cross_layers = nn.ModuleList([
            nn.Linear(input_dim, input_dim, bias=True)
            for _ in range(num_cross_layers)
        ])
        
        # Deep network
        layers = []
        in_dim = input_dim
        for h_dim in deep_layers:
            layers.append(nn.Linear(in_dim, h_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            in_dim = h_dim
        self.deep_net = nn.Sequential(*layers)
        
        # Output concatenation
        self.output_dim = input_dim + deep_layers[-1]
    
    def forward(self, x0):
        """
        Args:
            x0: (B, input_dim)
            
        Returns:
            output: (B, output_dim)
        """
        # Cross path
        x_cross = x0
        for layer in self.cross_layers:
            x_cross = x0 * layer(x_cross) + x_cross
        
        # Deep path
        x_deep = self.deep_net(x0)
        
        # Concatenate
        return torch.cat([x_cross, x_deep], dim=1)

class CTRModel(nn.Module):
    
    def __init__(
        self,
        num_items,
        frozen_embeddings,
        item_tags,
        num_tags,
        embed_dim=64,
        tag_embed_dim=16,
        k=16,
        num_transformer_layers=2,
        num_heads=4,
        num_cross_layers=3,
        deep_layers=None,
        dropout=0.2,
        learning_rate=5e-4,
    ):
        super().__init__()
        
        if deep_layers is None:
            deep_layers = [1024, 512, 256]
        
        # 1. Embedding layer
        self.embedding = EmbeddingLayer(
            num_items, embed_dim, frozen_embeddings, item_tags, num_tags, tag_embed_dim
        )
        item_emb_dim = self.embedding.final_dim  # 128 + 64 + 16 = 208
        
        # 2. Sequential learning
        self.seq_learning = SequentialLearning(
            item_emb_dim, k=k, num_layers=num_transformer_layers, 
            num_heads=num_heads, dropout=dropout
        )
        
        # 3. Side features (likes + views)
        self.side_proj = nn.Linear(2, 32)
        
        # 4. DCNv2
        dcn_input_dim = item_emb_dim + self.seq_learning.output_dim + 32
        self.dcn = DCNv2(dcn_input_dim, num_cross_layers=num_cross_layers, 
                        deep_layers=deep_layers, dropout=dropout)
        
        # 5. Prediction head
        self.pred_head = nn.Sequential(
            nn.Linear(self.dcn.output_dim, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1)
        )
        
        self.to(device)
        
        # Optimizer
        self.optimizer = Adam(self.parameters(), lr=learning_rate, weight_decay=1e-5)
        self.scheduler = ReduceLROnPlateau(self.optimizer, mode='max', factor=0.5, patience=3)
        self.criterion = None
        
        # Info
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"\n{'='*70}")
        print(f"Trainable params: {trainable:,}")
        print(f"Item embedding dim: {item_emb_dim}")
        print(f"Seq output dim: {self.seq_learning.output_dim}")
        print(f"Learning rate: {learning_rate}")
        print(f"{'='*70}\n")
    
    def forward(self, batch):
        """
        Args:
            batch: {
                'item_seqs': (B, seq_len),
                'item_ids': (B,),
                'likes_levels': (B,),
                'views_levels': (B,)
            }
        """
        item_seqs = batch["item_seqs"]
        item_ids = batch["item_ids"]
        likes = batch["likes_levels"]
        views = batch["views_levels"]
        
        # Get embeddings
        seq_embs = self.embedding(item_seqs)  # (B, seq_len, emb_dim)
        target_emb = self.embedding(item_ids)  # (B, emb_dim)
        
        # Sequential learning
        S_o = self.seq_learning(item_seqs, seq_embs, target_emb)  # (B, seq_output_dim)
        
        # Side features
        side = torch.stack([likes, views], dim=1)  # (B, 2)
        side_emb = self.side_proj(side)  # (B, 32)
        
        # Concatenate
        x0 = torch.cat([target_emb, S_o, side_emb], dim=1)
        
        # Feature interaction
        x_inter = self.dcn(x0)
        
        # Prediction
        logits = self.pred_head(x_inter).squeeze(-1)
        
        return logits
    
    def compute_auc(self, preds, labels):
        preds_np = torch.sigmoid(preds).detach().cpu().numpy()
        labels_np = labels.cpu().numpy()
        try:
            return roc_auc_score(labels_np, preds_np)
        except:
            return None
    
    def train_epoch(self, loader):
        self.train()
        total_loss = 0
        all_preds = []
        all_labels = []
        
        for batch in tqdm(loader, desc='Training'):
            batch = {k: v.to(device) for k, v in batch.items()}
            labels = batch["labels"]
            
            preds = self.forward(batch)
            loss = self.criterion(preds, labels)
            
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.parameters(), 1.0)
            self.optimizer.step()
            
            total_loss += loss.item()
            all_preds.append(preds.detach().cpu())
            all_labels.append(labels.cpu())
        
        avg_loss = total_loss / len(loader)
        auc = self.compute_auc(torch.cat(all_preds), torch.cat(all_labels))
        return avg_loss, auc
    
    def validate(self, loader):
        self.eval()
        total_loss = 0
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for batch in tqdm(loader, desc='Validating'):
                batch = {k: v.to(device) for k, v in batch.items()}
                labels = batch["labels"]
                
                preds = self.forward(batch)
                loss = self.criterion(preds, labels)
                
                total_loss += loss.item()
                all_preds.append(preds.cpu())
                all_labels.append(labels.cpu())
        
        avg_loss = total_loss / len(loader)
        auc = self.compute_auc(torch.cat(all_preds), torch.cat(all_labels))
        return avg_loss, auc
    
    def fit(self, train_loader, valid_loader, num_epochs=20, start_epoch=0, save_path=None):
        
        # Calculate pos_weight
        pos_weight = 3.0000
        self.criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos_weight).to(device))
        print(f"Pos weight: {pos_weight:.4f}\n")
        
        best_auc = 0
        patience = 5
        patience_counter = 0
        
        for epoch in range(start_epoch, start_epoch + num_epochs):
            print(f"\n{'='*70}")
            print(f"EPOCH {epoch+1}/{start_epoch + num_epochs}")
            print(f"{'='*70}")
            
            tr_loss, tr_auc = self.train_epoch(train_loader)
            print(f"Train: Loss={tr_loss:.4f}, AUC={tr_auc:.4f}")
            
            val_loss, val_auc = self.validate(valid_loader)
            print(f"Val:   Loss={val_loss:.4f}, AUC={val_auc:.4f}")
            
            gap = tr_auc - val_auc
            print(f"Gap:   {gap:.4f}")
            
            self.scheduler.step(val_auc)
            
            if val_auc > best_auc:
                best_auc = val_auc
                patience_counter = 0
                if save_path:
                    torch.save({
                        'model': self.state_dict(),
                        'optimizer': self.optimizer.state_dict(),
                        'scheduler': self.scheduler.state_dict(),
                        'epoch': epoch,
                        'val_auc': val_auc,
                    }, f"/kaggle/working/model_MMCTR_{epoch}.pth")
                    print(f"✓ Saved (AUC: {val_auc:.4f})")
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"\nEarly stopping")
                    break
        
        print(f"\n{'='*70}")
        print(f"Best Val AUC: {best_auc:.4f}")
        print(f"{'='*70}")



# from torch.utils.data import DataLoader
# from task2.dataset.dataset import Task2Dataset, collate_fn
# from task2.model_loader import load_item_embeddings_and_tags

# # Load data
# train_dataset = Task2Dataset(data_path="/kaggle/input/www2025-mmctr-data/MicroLens_1M_MMCTR/MicroLens_1M_x1/train.parquet", is_train=True)
# valid_dataset = Task2Dataset(data_path="/kaggle/input/www2025-mmctr-data/MicroLens_1M_MMCTR/MicroLens_1M_x1/valid.parquet", is_train=True)

# train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True, collate_fn=collate_fn)
# valid_loader = DataLoader(valid_dataset, batch_size=128, shuffle=False, collate_fn=collate_fn)

# # Create model

# # Load data
# embeddings, item_tags, num_items, num_tags = load_item_embeddings_and_tags(
#     item_info_path="/kaggle/working/item_info_with_clip.parquet",
#     embedding_source="item_clip_emb_d128"
# )

# checkpoint = torch.load("/kaggle/working/model_21.pth", weights_only=False, map_location='cuda')


# model = CTRModel(
#     num_items=num_items,
#     frozen_embeddings=embeddings,
#     item_tags=item_tags,
#     num_tags=num_tags,
#     embed_dim=64,  # Exact
#     tag_embed_dim=16,  # Exact
#     k=16,  # Exact
#     num_transformer_layers=2,  # Exact
#     num_heads=4,
#     num_cross_layers=3,  # Exact
#     deep_layers=[1024, 512, 256],  # Exact
#     dropout=0.2,  # Exact
#     learning_rate=5e-4,  # Exact
# )

# # Load saved state
# model.load_state_dict(checkpoint['model'], strict=False)
# model.optimizer.load_state_dict(checkpoint['optimizer'])
# model.scheduler.load_state_dict(checkpoint['scheduler'])

# start_epoch = checkpoint['epoch'] 
# print(f"Loaded checkpoint from epoch {checkpoint['epoch']}")
# print(f"Resuming training from epoch {start_epoch}")

# # Resume training
# num_remaining_epochs = 50 - start_epoch
# print(f"Training for {num_remaining_epochs} more epochs\n")

# model.fit(
#     train_loader, 
#     valid_loader, 
#     num_epochs=num_remaining_epochs,
#     start_epoch=start_epoch, 
#     save_path="model"
# )


