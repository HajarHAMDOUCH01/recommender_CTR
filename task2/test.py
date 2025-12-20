import torch
import polars as pl
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm
import sys

sys.path.append("/kaggle/working/recommender_CTR")

from task2.dataset.dataset import Task2Dataset, collate_fn
from task2.model import CTRModel
from task2.model_loader import load_item_embeddings_and_tags

# Configuration
TEST_DATA_PATH = "/kaggle/input/www2025-mmctr-data/MicroLens_1M_MMCTR/MicroLens_1M_x1/test.parquet"
ITEM_INFO_PATH = "/kaggle/working/item_info_with_clip.parquet"
MODEL_PATH = "/kaggle/working/model_21.pth"
OUTPUT_PATH = "/kaggle/working/predictions.csv"
BATCH_SIZE = 256
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

print("="*80)
print("MODEL INFERENCE SCRIPT")
print("="*80)
print(f"Device: {DEVICE}")
print(f"Test data: {TEST_DATA_PATH}")
print(f"Model checkpoint: {MODEL_PATH}")
print(f"Output: {OUTPUT_PATH}\n")

# Load embeddings and tags
print("Loading item embeddings and tags...")
embeddings, item_tags, num_items, num_tags = load_item_embeddings_and_tags(
    item_info_path=ITEM_INFO_PATH,
    embedding_source="item_clip_emb_d128"
)

# Load test dataset
print("\nLoading test dataset...")
test_dataset = Task2Dataset(
    data_path=TEST_DATA_PATH,
    is_train=False
)

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    collate_fn=collate_fn,
    num_workers=2,
    pin_memory=True
)

# Instantiate model with exact architecture
print("\nInstantiating model...")
model = CTRModel(
    num_items=num_items,
    frozen_embeddings=embeddings,
    item_tags=item_tags,
    num_tags=num_tags,
    embed_dim=64,
    tag_embed_dim=16,
    k=16,
    num_transformer_layers=2,
    num_heads=4,
    num_cross_layers=3,
    deep_layers=[1024, 512, 256],
    dropout=0.2,
    learning_rate=5e-4,
)

# Load checkpoint
print(f"\nLoading checkpoint from {MODEL_PATH}...")
checkpoint = torch.load(MODEL_PATH, map_location=DEVICE, weights_only=False)

model.load_state_dict(checkpoint['model'], strict=False)
print(f"✓ Checkpoint loaded from epoch {checkpoint['epoch']}")
print(f"  Validation AUC: {checkpoint.get('val_auc', 'N/A')}")

# Set model to evaluation mode
model.eval()
model.to(DEVICE)

# Generate predictions
print("\n" + "="*80)
print("GENERATING PREDICTIONS")
print("="*80 + "\n")

all_predictions = []

with torch.no_grad():
    for batch in tqdm(test_loader, desc="Inference"):
        # Move batch to device
        batch_gpu = {k: v.to(DEVICE) for k, v in batch.items() if k not in ['id']}
        
        # Get predictions (logits)
        logits = model.forward(batch_gpu)
        
        # Convert to probabilities
        probs = torch.sigmoid(logits).cpu().numpy()
        
        # Store results
        all_predictions.extend(probs)

# Convert to arrays
all_predictions = np.array(all_predictions)

print(f"\nGenerated {len(all_predictions):,} predictions")
print(f"Prediction range: [{all_predictions.min():.6f}, {all_predictions.max():.6f}]")
print(f"Prediction mean: {all_predictions.mean():.6f}, std: {all_predictions.std():.6f}")

# Create submission dataframe
print("\nCreating submission file...")

# Get IDs from the original test data
# Check if 'id' column exists in test data
if "id" in test_dataset.data.columns:
    all_ids = test_dataset.data["id"].to_numpy()
    print(f"Using 'id' column from test data")
else:
    # If no id column, create sequential IDs starting from 0
    all_ids = np.arange(len(all_predictions))
    print(f"No 'id' column found, creating sequential IDs")

print(f"Number of IDs: {len(all_ids):,}")
print(f"Number of predictions: {len(all_predictions):,}")

# Verify lengths match
assert len(all_ids) == len(all_predictions), f"Length mismatch: {len(all_ids)} IDs vs {len(all_predictions)} predictions"

# Both tasks use the same probability predictions
task1_predictions = all_predictions
task2_predictions = all_predictions

# Create dataframe
submission_df = pl.DataFrame({
    "id": all_ids,
    "task1": task1_predictions,
    "task2": task2_predictions
})

# Sort by id
submission_df = submission_df.sort("id")

# Save to CSV
submission_df.write_csv(OUTPUT_PATH)

print(f"\n✓ Predictions saved to {OUTPUT_PATH}")
print(f"\nSubmission format:")
print(submission_df.head(10))

print("\n" + "="*80)
print("Statistics:")
print("="*80)
print(f"Total samples: {len(submission_df):,}")
print(f"\nPrediction Statistics:")
print(f"  Min: {all_predictions.min():.6f}")
print(f"  Max: {all_predictions.max():.6f}")
print(f"  Mean: {all_predictions.mean():.6f}")
print(f"  Median: {np.median(all_predictions):.6f}")
print(f"  Std: {all_predictions.std():.6f}")
print(f"\nPositive rate (>0.5): {(all_predictions > 0.5).mean()*100:.2f}%")
print(f"Negative rate (<=0.5): {(all_predictions <= 0.5).mean()*100:.2f}%")

print("\n" + "="*80)
print("INFERENCE COMPLETE")
print("="*80)