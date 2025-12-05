import numpy as np
import polars as pl
import torch
from typing import Tuple

def load_item_embeddings_and_tags(
    item_info_path: str,
    embedding_source: str = "item_clip_emb_d128"
) -> Tuple[np.ndarray, np.ndarray, int, int]:
    """
    Load item embeddings and tags from parquet file.
    
    Args:
        item_info_path: Path to item_info parquet file
        embedding_source: Which embedding column to use
                         - "item_emb_d128" (BERT+CLIP PCA)
                         - "item_clip_emb_d128" (CLIP only)
    
    Returns:
        embeddings: (num_items, 128) - frozen multimodal embeddings
        item_tags: (num_items, 5) - item tag IDs
        num_items: Total number of items
        num_tags: Total number of unique tag IDs (max tag ID + 1)
    """
    
    print("="*80)
    print(f"LOADING ITEM DATA: {embedding_source}")
    print("="*80)
    
    # Load parquet file
    item_info = pl.read_parquet(item_info_path)
    print(f"\nLoaded item_info shape: {item_info.shape}")
    print(f"Columns: {item_info.columns}\n")
    
    # Sort by item_id to ensure proper indexing
    item_info = item_info.sort("item_id")
    
    if embedding_source not in item_info.columns:
        available = [col for col in item_info.columns if "emb" in col.lower()]
        raise ValueError(
            f"Column '{embedding_source}' not found. "
            f"Available embedding columns: {available}"
        )
    
    print(f"Loading embeddings from: {embedding_source}")
    embeddings_list = item_info[embedding_source].to_list()
    
    # Convert to numpy array
    embeddings = np.array(embeddings_list, dtype=np.float32)
    print(f"Embeddings shape: {embeddings.shape}")
    print(f"Embeddings dtype: {embeddings.dtype}")
    print(f"Embeddings range: [{embeddings.min():.6f}, {embeddings.max():.6f}]")
    print(f"Embeddings mean: {embeddings.mean():.6f}, std: {embeddings.std():.6f}")
    
    # Verify shape
    if embeddings.shape[1] != 128:
        raise ValueError(f"Expected embedding dim 128, got {embeddings.shape[1]}")
    
    if "item_tags" not in item_info.columns:
        raise ValueError("'item_tags' column not found in item_info")
    
    print(f"\nLoading item tags...")
    item_tags_list = item_info["item_tags"].to_list()
    
    # Convert to numpy array
    item_tags = np.array(item_tags_list, dtype=np.int64)
    print(f"Item tags shape: {item_tags.shape}")
    print(f"Item tags dtype: {item_tags.dtype}")
    
    # Verify shape
    if item_tags.ndim != 2 or item_tags.shape[1] != 5:
        raise ValueError(
            f"Expected item_tags shape (N, 5), got {item_tags.shape}. "
            f"Each item should have 5 tags."
        )
    
    print(f"Item tags range: [{item_tags.min()}, {item_tags.max()}]")
    
    num_items = embeddings.shape[0]
    num_tags = int(item_tags.max()) + 1  # max tag ID + 1 for padding (0)
    
    print(f"\nNumber of items: {num_items:,}")
    print(f"Number of unique tags: {num_tags:,}")
    print(f"Max tag ID: {item_tags.max()}")
    
    first_item_id = item_info["item_id"][0]
    if first_item_id == 0:
        is_emb_zero = np.allclose(embeddings[0], 0.0)
        is_tags_zero = np.all(item_tags[0] == 0)
        
        if is_emb_zero and is_tags_zero:
            print(f"✓ Row 0 (item_id=0) is properly zeroed for padding")
        else:
            print(f"⚠️  Warning: Row 0 should be zero padding!")
            print(f"   Embeddings all zero: {is_emb_zero}")
            print(f"   Tags all zero: {is_tags_zero}")
    else:
        print(f"⚠️  Warning: First item_id is {first_item_id}, not 0")
        print(f"   Padding row may not be at index 0")
    
    print("\n" + "="*80)
    print("✓ DATA LOADED SUCCESSFULLY")
    print("="*80 + "\n")
    
    return embeddings, item_tags, num_items, num_tags

