#!/usr/bin/env python3
"""
ISAAC AI – Embedding Generator using FastEmbed

  📚 Reads chunks from chunks/all_chunks.txt
  🧠 Generates embeddings using FastEmbed (lightweight)
  💾 Saves to knowledge/ with metadata
  🔄 Auto-resume support

"""

import os
import sys
import time
import json
import pickle
import gc
import numpy as np
from pathlib import Path
from datetime import datetime

# ============================================================
# CONFIGURATION
# ============================================================

HOME = Path.home()
ISAAC_DIR = HOME / "ISAAC_AI"
CHUNKS_DIR = ISAAC_DIR / "chunks"
KNOWLEDGE_DIR = ISAAC_DIR / "knowledge"

CHUNKS_FILE = CHUNKS_DIR / "all_chunks.txt"
EMBEDDINGS_FILE = KNOWLEDGE_DIR / "embeddings.npy"
PROGRESS_FILE = KNOWLEDGE_DIR / "embedding_progress.pkl"
SUMMARY_FILE = KNOWLEDGE_DIR / "embedding_summary.json"

# FastEmbed settings
BATCH_SIZE = 64  # FastEmbed handles larger batches
MODEL_NAME = "BAAI/bge-small-en-v1.5"  # 384 dimensions, lightweight

# ============================================================
# FUNCTIONS
# ============================================================

def print_header(text):
    print(f"\n{'='*60}")
    print(f"   {text}")
    print(f"{'='*60}\n")

def ensure_dir(path):
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path

def load_chunks():
    """Load all chunks from all_chunks.txt."""
    print_header("📚 LOADING CHUNKS")
    
    if not CHUNKS_FILE.exists():
        print(f"❌ Chunks file not found: {CHUNKS_FILE}")
        return None
    
    print(f"📁 Reading: {CHUNKS_FILE}")
    
    chunks = []
    current_book = None
    current_chunk = ""
    
    with open(CHUNKS_FILE, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            
            if line.startswith("--- BOOK:"):
                # Save previous chunk if exists
                if current_chunk and current_book is not None:
                    chunks.append(current_chunk)
                    current_chunk = ""
                
                current_book = line.replace("--- BOOK:", "").strip()
                
            elif line == "---":
                # End of chunk - save it
                if current_chunk and current_book is not None:
                    chunks.append(current_chunk)
                    current_chunk = ""
                    
            else:
                # Add to current chunk
                if current_book is not None:
                    if current_chunk:
                        current_chunk += " " + line
                    else:
                        current_chunk = line
    
    # Save last chunk if exists
    if current_chunk and current_book is not None:
        chunks.append(current_chunk)
    
    print(f"✅ Loaded {len(chunks)} chunks")
    return chunks

def load_progress():
    """Load embedding progress."""
    if PROGRESS_FILE.exists():
        try:
            with open(PROGRESS_FILE, 'rb') as f:
                return pickle.load(f)
        except:
            pass
    return {"processed": 0, "last_index": 0}

def save_progress(progress):
    """Save embedding progress."""
    ensure_dir(KNOWLEDGE_DIR)
    with open(PROGRESS_FILE, 'wb') as f:
        pickle.dump(progress, f)

def generate_embeddings(chunks):
    """Generate embeddings using FastEmbed."""
    print_header("🧠 GENERATING EMBEDDINGS")
    
    try:
        from fastembed import TextEmbedding
    except ImportError:
        print("❌ FastEmbed not installed!")
        print("   Run: pip install fastembed")
        return None
    
    print(f"📥 Loading model: {MODEL_NAME}")
    try:
        model = TextEmbedding(MODEL_NAME)
        print("✅ Model loaded (384 dimensions)")
    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        return None
    
    # Load progress
    progress = load_progress()
    start_index = progress.get("last_index", 0)
    
    print(f"📊 Resume from index: {start_index}")
    print(f"📊 Total chunks: {len(chunks)}")
    
    if start_index >= len(chunks):
        print("✅ All chunks already embedded!")
        return np.load(EMBEDDINGS_FILE, allow_pickle=True)
    
    # Process chunks
    all_embeddings = []
    total = len(chunks)
    batch_size = BATCH_SIZE
    
    print(f"\n🔄 Processing {total - start_index} chunks")
    print(f"📦 Batch size: {batch_size}\n")
    
    for i in range(start_index, total, batch_size):
        batch = chunks[i:i+batch_size]
        
        try:
            # Generate embeddings
            embeddings = list(model.embed(batch))
            all_embeddings.extend(embeddings)
            
            # Save progress
            progress = {
                "processed": len(all_embeddings),
                "last_index": i + len(batch)
            }
            save_progress(progress)
            
            # Show progress
            done = min(i + batch_size, total)
            print(f"   ✅ {done}/{total} chunks done ({done/total*100:.1f}%)")
            
            # Periodic cleanup
            if len(all_embeddings) % 1000 == 0:
                gc.collect()
                
        except Exception as e:
            print(f"   ❌ Error at chunk {i}: {e}")
            continue
    
    print(f"\n✅ Embedding complete!")
    print(f"   Total: {len(all_embeddings)} embeddings")
    
    return np.array(all_embeddings)

def save_embeddings(embeddings):
    """Save embeddings."""
    ensure_dir(KNOWLEDGE_DIR)
    
    np.save(EMBEDDINGS_FILE, embeddings)
    print(f"✅ Saved embeddings: {EMBEDDINGS_FILE}")
    print(f"   Shape: {embeddings.shape}")
    print(f"   Size: {EMBEDDINGS_FILE.stat().st_size / (1024*1024):.1f} MB")

def save_summary(chunks, embeddings):
    """Save summary."""
    summary = {
        "total_chunks": len(chunks),
        "embedding_dimension": embeddings.shape[1] if embeddings.ndim > 1 else 0,
        "model_name": MODEL_NAME,
        "batch_size": BATCH_SIZE,
        "timestamp": datetime.now().isoformat(),
        "embeddings_file": str(EMBEDDINGS_FILE)
    }
    with open(SUMMARY_FILE, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2)
    print(f"✅ Saved summary: {SUMMARY_FILE}")

# ============================================================
# MAIN
# ============================================================

def main():
    print("\n" + "="*60)
    print("   🧠 ISAAC AI – FastEmbed Embedding Generator")
    print("   📚 Chunks → Embeddings → Knowledge Base")
    print("="*60)
    print(f"\n📁 Chunks folder: {CHUNKS_DIR}")
    print(f"📁 Knowledge folder: {KNOWLEDGE_DIR}")
    print(f"🧠 Model: {MODEL_NAME}")
    print(f"📦 Batch size: {BATCH_SIZE}")
    print("="*60 + "\n")
    
    # Ensure directories exist
    ensure_dir(KNOWLEDGE_DIR)
    
    # Load chunks
    chunks = load_chunks()
    if not chunks:
        print("❌ No chunks to process!")
        return
    
    # Generate embeddings
    embeddings = generate_embeddings(chunks)
    
    if embeddings is None or len(embeddings) == 0:
        print("❌ No embeddings generated!")
        return
    
    # Save
    save_embeddings(embeddings)
    save_summary(chunks, embeddings)
    
    # Summary
    print("\n" + "="*60)
    print("   ✅ EMBEDDING COMPLETE!")
    print("="*60)
    print(f"\n   📁 Knowledge folder: {KNOWLEDGE_DIR}")
    print(f"   📄 Chunks: {len(chunks)}")
    print(f"   📊 Embedding dimension: {embeddings.shape[1] if embeddings.ndim > 1 else 0}")
    print(f"   📦 File size: {(EMBEDDINGS_FILE.stat().st_size / (1024*1024)):.1f} MB")
    print("\n   🚀 Next step: Search or Chat")
    print("      python isaac.py")
    print("="*60)

if __name__ == "__main__":
    main()
