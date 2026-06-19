#!/usr/bin/env python3
"""
ISAAC AI – Reverse Embedding Generator (Processes from END to START)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  📚 Reads chunks from END to START
  🧠 Generates embeddings using FastEmbed
  💾 Appends to same knowledge/ folder
  🔄 Auto-resume and meets the forward process in the middle
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
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
PROGRESS_FILE = KNOWLEDGE_DIR / "embedding_progress_reverse.pkl"
SUMMARY_FILE = KNOWLEDGE_DIR / "embedding_summary_reverse.json"

BATCH_SIZE = 64
MODEL_NAME = "BAAI/bge-small-en-v1.5"
TOTAL_CHUNKS = 78514

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

def load_chunks_reverse():
    """Load chunks from END to START."""
    print_header("📚 LOADING CHUNKS (REVERSE)")
    
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
                if current_chunk and current_book is not None:
                    chunks.append(current_chunk)
                    current_chunk = ""
                current_book = line.replace("--- BOOK:", "").strip()
            elif line == "---":
                if current_chunk and current_book is not None:
                    chunks.append(current_chunk)
                    current_chunk = ""
            else:
                if current_book is not None:
                    if current_chunk:
                        current_chunk += " " + line
                    else:
                        current_chunk = line
    
    if current_chunk and current_book is not None:
        chunks.append(current_chunk)
    
    # Reverse the chunks!
    chunks.reverse()
    print(f"✅ Loaded {len(chunks)} chunks (reversed)")
    return chunks

def load_progress():
    """Load reverse embedding progress."""
    if PROGRESS_FILE.exists():
        try:
            with open(PROGRESS_FILE, 'rb') as f:
                return pickle.load(f)
        except:
            pass
    return {"processed": 0, "last_index": 0, "forward_index": 0}

def save_progress(progress):
    """Save reverse embedding progress."""
    ensure_dir(KNOWLEDGE_DIR)
    with open(PROGRESS_FILE, 'wb') as f:
        pickle.dump(progress, f)

def load_forward_progress():
    """Check forward progress to know when to stop."""
    forward_file = KNOWLEDGE_DIR / "embedding_progress.pkl"
    if forward_file.exists():
        try:
            with open(forward_file, 'rb') as f:
                return pickle.load(f)
        except:
            pass
    return {"last_index": 0}

def generate_embeddings_reverse(chunks):
    """Generate embeddings from END to START."""
    print_header("🧠 GENERATING EMBEDDINGS (REVERSE)")
    
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
    all_embeddings = progress.get("embeddings", [])
    
    print(f"📊 Resume from index: {start_index}")
    print(f"📊 Total chunks: {len(chunks)}")
    
    if start_index >= len(chunks):
        print("✅ All chunks already embedded!")
        return np.array(all_embeddings) if all_embeddings else None
    
    total = len(chunks)
    batch_size = BATCH_SIZE
    
    print(f"\n🔄 Processing {total - start_index} chunks (backwards)")
    print(f"📦 Batch size: {batch_size}\n")
    
    for i in range(start_index, total, batch_size):
        # Check forward progress to know when to stop
        forward = load_forward_progress()
        forward_index = forward.get("last_index", 0)
        
        # If forward has passed this point, we've met in the middle
        if forward_index >= (total - i):
            print(f"\n🔄 MET IN THE MIDDLE! Forward at {forward_index}, Reverse at {total - i}")
            print(f"✅ Both processes completed!")
            break
        
        batch = chunks[i:i+batch_size]
        
        try:
            embeddings = list(model.embed(batch))
            
            if not all_embeddings:
                all_embeddings = list(embeddings)
            else:
                all_embeddings.extend(embeddings)
            
            progress = {
                "processed": len(all_embeddings),
                "last_index": i + len(batch),
                "forward_index": forward_index
            }
            save_progress(progress)
            
            done = min(i + batch_size, total)
            percent = (done / total) * 100
            reverse_pos = total - done
            print(f"   ✅ {done}/{total} chunks done ({percent:.1f}%) | Reverse position: {reverse_pos}")
            
            if len(all_embeddings) % 1000 == 0:
                gc.collect()
                
        except Exception as e:
            print(f"   ❌ Error at chunk {i}: {e}")
            continue
    
    print(f"\n✅ Reverse embedding complete!")
    print(f"   Total: {len(all_embeddings)} embeddings")
    
    return np.array(all_embeddings) if all_embeddings else None

def save_embeddings(embeddings):
    """Save reverse embeddings."""
    ensure_dir(KNOWLEDGE_DIR)
    
    # Load existing embeddings if any
    if EMBEDDINGS_FILE.exists():
        existing = np.load(EMBEDDINGS_FILE, allow_pickle=True)
        # Combine: existing + new
        combined = np.vstack([existing, embeddings])
        np.save(EMBEDDINGS_FILE, combined)
        print(f"✅ Combined embeddings: {combined.shape}")
    else:
        np.save(EMBEDDINGS_FILE, embeddings)
        print(f"✅ Saved embeddings: {EMBEDDINGS_FILE}")
        print(f"   Shape: {embeddings.shape}")

# ============================================================
# MAIN
# ============================================================

def main():
    print("\n" + "="*60)
    print("   🧠 ISAAC AI – Reverse Embedding Generator")
    print("   📚 Processes from END to START")
    print("   🔄 Meets forward process in the middle")
    print("="*60)
    print(f"\n📁 Chunks folder: {CHUNKS_DIR}")
    print(f"📁 Knowledge folder: {KNOWLEDGE_DIR}")
    print(f"🧠 Model: {MODEL_NAME}")
    print(f"📦 Batch size: {BATCH_SIZE}")
    print("="*60 + "\n")
    
    ensure_dir(KNOWLEDGE_DIR)
    
    # Load chunks in reverse
    chunks = load_chunks_reverse()
    if not chunks:
        print("❌ No chunks to process!")
        return
    
    # Generate embeddings
    embeddings = generate_embeddings_reverse(chunks)
    
    if embeddings is None or len(embeddings) == 0:
        print("❌ No embeddings generated!")
        return
    
    # Save
    save_embeddings(embeddings)
    
    print("\n" + "="*60)
    print("   ✅ REVERSE EMBEDDING COMPLETE!")
    print("="*60)
    print(f"\n   📁 Knowledge folder: {KNOWLEDGE_DIR}")
    print(f"   📊 Final shape: {embeddings.shape}")
    print("\n   🚀 Both forward and reverse processes done!")
    print("="*60)

if __name__ == "__main__":
    main()
