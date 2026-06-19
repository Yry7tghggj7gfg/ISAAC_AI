#!/usr/bin/env python3
"""
ISAAC AI – Complete PDF to Vector Pipeline
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  📁 Scan ALL storage → 📄 Convert to TXT → 🧠 Generate Vectors
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""

import os
import sys
import time
import json
import pickle
import subprocess
import numpy as np
from pathlib import Path
from datetime import datetime
from collections import Counter

# ============================================================
# CONFIGURATION
# ============================================================

# Storage paths to scan for PDFs
STORAGE_PATHS = [
    "/storage/emulated/0",
    "/storage/105B-F7AB",
    "/storage/sdcard0",
    "/storage/sdcard1",
    "/storage/extSdCard",
    "/storage/self/primary",
    "/mnt/sdcard",
    "/sdcard",
]

# Output directories
HOME = Path.home()
ISAAC_DIR = HOME / "ISAAC_AI"
TEXT_DIR = ISAAC_DIR / "texts"
KNOWLEDGE_DIR = ISAAC_DIR / "knowledge_fastembed"
PROCESSED_FILE = KNOWLEDGE_DIR / "processed_books.txt"
PROGRESS_FILE = KNOWLEDGE_DIR / "progress.pkl"
CHUNKS_FILE = KNOWLEDGE_DIR / "chunks.txt"
EMBEDDINGS_FILE = KNOWLEDGE_DIR / "embeddings.npy"
METADATA_FILE = KNOWLEDGE_DIR / "pipeline_metadata.txt"

# Processing parameters
CHUNK_SIZE = 300
CHUNK_OVERLAP = 30
MIN_PDF_SIZE = 5 * 1024 * 1024  # 5 MB
MIN_TXT_SIZE = 10 * 1024  # 10 KB
MAX_CHUNKS_PER_BOOK = 500

# Skip these directories
SKIP_DIRS = [
    "Android", "Download", "Music", "Pictures", "Movies", "DCIM",
    "WhatsApp", "Telegram", "Instagram", "Facebook", "TikTok",
    "cache", "tmp", "temp", ".trash", "Lost+Found", "Google",
    "Maps", "YouTube", "Gmail", "Chrome", "Firefox"
]

# ============================================================
# BANNER
# ============================================================

BANNER = """
╔══════════════════════════════════════════════════════════════╗
║                                                              ║
║   📚 PDF → VECTOR PIPELINE                                  ║
║                                                              ║
║   📁 Scan ALL Storage                                       ║
║   📄 Convert PDF → TXT                                     ║
║   🧠 Generate Embeddings                                    ║
║   💾 Save to ISAAC Knowledge                               ║
║                                                              ║
╚══════════════════════════════════════════════════════════════╝
"""

# ============================================================
# UTILITY FUNCTIONS
# ============================================================

def print_header(text):
    print(f"\n{'='*60}")
    print(f"   {text}")
    print(f"{'='*60}\n")

def sanitize_text(text):
    import re
    text = text.replace('\x00', '')
    text = re.sub(r'\n{3,}', '\n\n', text)
    text = re.sub(r'[^\x20-\x7E\n\t]', ' ', text)
    return text

def chunk_text(text, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    chunks = []
    pos = 0
    text_len = len(text)
    while pos < text_len and len(chunks) < MAX_CHUNKS_PER_BOOK:
        end = min(pos + chunk_size, text_len)
        if end < text_len:
            for break_char in ['\n\n', '\n', '. ', '? ', '! ', ' ']:
                last = text.rfind(break_char, pos, end)
                if last > pos + chunk_size // 2:
                    end = last + len(break_char)
                    break
        chunk = text[pos:end].strip()
        if chunk and len(chunk) > 20:
            chunks.append(chunk)
        pos = end - overlap
        if pos >= text_len:
            break
    return chunks

def get_file_size_mb(path):
    try:
        return path.stat().st_size / (1024 * 1024)
    except:
        return 0

def should_skip_path(path):
    path_str = str(path).lower()
    for skip in SKIP_DIRS:
        if skip.lower() in path_str:
            return True
    return False

# ============================================================
# STEP 1: FIND ALL PDFs
# ============================================================

def find_all_pdfs():
    """Find all PDFs ≥5MB across all storage paths."""
    print_header("📁 STEP 1: FINDING ALL PDFs")
    
    pdf_files = []
    seen_paths = set()
    
    for root in STORAGE_PATHS:
        root_path = Path(root)
        if not root_path.exists():
            continue
        
        print(f"🔍 Scanning: {root}")
        count = 0
        
        try:
            for pdf_path in root_path.rglob("*.pdf"):
                if should_skip_path(pdf_path):
                    continue
                
                # Avoid duplicates (symlinks, etc.)
                abs_path = str(pdf_path.absolute())
                if abs_path in seen_paths:
                    continue
                seen_paths.add(abs_path)
                
                # Check size
                size_mb = get_file_size_mb(pdf_path)
                if size_mb >= MIN_PDF_SIZE / (1024 * 1024):
                    pdf_files.append(pdf_path)
                    count += 1
                    if count % 10 == 0:
                        print(f"   Found {count} PDFs...")
        except Exception as e:
            print(f"   ⚠️ Error scanning {root}: {e}")
        
        print(f"   Found {count} PDFs in {root}")
    
    print(f"\n✅ Total PDFs found: {len(pdf_files)}")
    return pdf_files

# ============================================================
# STEP 2: CONVERT PDFs TO TXT
# ============================================================

def convert_pdfs_to_text(pdf_files):
    """Convert all PDFs to .txt files."""
    print_header("📄 STEP 2: CONVERTING PDFs TO TXT")
    
    os.makedirs(TEXT_DIR, exist_ok=True)
    
    converted = 0
    failed = 0
    skipped = 0
    total = len(pdf_files)
    
    for i, pdf_path in enumerate(pdf_files, 1):
        txt_path = TEXT_DIR / f"{pdf_path.stem}.txt"
        
        # Skip if already exists and has content
        if txt_path.exists() and txt_path.stat().st_size > 1000:
            skipped += 1
            continue
        
        print(f"   [{i}/{total}] 📖 {pdf_path.name[:50]}...", end=" ")
        
        try:
            result = subprocess.run(
                ["pdftotext", "-enc", "UTF-8", str(pdf_path), str(txt_path)],
                capture_output=True, text=True
            )
            
            if result.returncode == 0 and txt_path.exists() and txt_path.stat().st_size > 1000:
                size_kb = txt_path.stat().st_size / 1024
                print(f"✅ Converted ({size_kb:.1f} KB)")
                converted += 1
            else:
                txt_path.unlink(missing_ok=True)
                print(f"❌ Failed (empty or corrupt)")
                failed += 1
        except Exception as e:
            print(f"❌ Error: {e}")
            failed += 1
    
    print(f"\n✅ Conversion complete:")
    print(f"   Converted: {converted}")
    print(f"   Failed: {failed}")
    print(f"   Already existed: {skipped}")
    print(f"   Total: {total}")
    
    return converted

# ============================================================
# STEP 3: LOAD TEXT FILES
# ============================================================

def load_text_files():
    """Load all .txt files from TEXT_DIR."""
    print_header("📂 STEP 3: LOADING TEXT FILES")
    
    if not TEXT_DIR.exists():
        print("❌ Text directory not found!")
        return []
    
    text_files = list(TEXT_DIR.glob("*.txt"))
    print(f"📚 Found {len(text_files)} text files")
    
    # Filter by size
    valid_texts = []
    for txt_path in text_files:
        size_mb = get_file_size_mb(txt_path)
        if size_mb < MIN_TXT_SIZE / 1024:
            continue
        try:
            with open(txt_path, 'r', encoding='utf-8', errors='ignore') as f:
                text = f.read()
            if text and len(text) > 100:
                valid_texts.append((txt_path.name, text))
                print(f"   ✅ {txt_path.name} ({len(text)/1024:.1f} KB)")
        except Exception as e:
            print(f"   ❌ {txt_path.name}: {e}")
    
    print(f"✅ Loaded {len(valid_texts)} valid text files")
    return valid_texts

# ============================================================
# STEP 4: GENERATE EMBEDDINGS
# ============================================================

def generate_embeddings(texts):
    """Generate embeddings for all texts."""
    print_header("🧠 STEP 4: GENERATING EMBEDDINGS")
    
    try:
        from fastembed import TextEmbedding
    except ImportError:
        print("❌ FastEmbed not installed. Run: pip install fastembed")
        return None, None
    
    print("📥 Loading embedding model...")
    try:
        model = TextEmbedding("BAAI/bge-small-en-v1.5")
        print("✅ Model loaded (384 dimensions)")
    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        return None, None
    
    all_chunks = []
    all_embeddings = []
    processed_books = []
    total_chunks_created = 0
    
    for i, (name, text) in enumerate(texts, 1):
        print(f"\n   [{i}/{len(texts)}] 📖 Processing: {name[:40]}...")
        
        # Sanitize text
        text = sanitize_text(text)
        if len(text) < 100:
            print(f"      ⚠️ Skipping (too short)")
            continue
        
        # Chunk
        chunks = chunk_text(text)
        if not chunks:
            print(f"      ⚠️ No chunks created")
            continue
        
        print(f"      📄 {len(chunks)} chunks")
        total_chunks_created += len(chunks)
        
        # Generate embeddings for each chunk
        book_chunks = []
        book_embeddings = []
        
        for chunk_idx, chunk in enumerate(chunks):
            try:
                emb = list(model.embed([chunk]))[0]
                book_chunks.append(chunk)
                book_embeddings.append(np.array(emb))
            except Exception as e:
                print(f"      ❌ Chunk {chunk_idx+1} failed: {e}")
                continue
        
        if book_chunks:
            all_chunks.extend(book_chunks)
            all_embeddings.extend(book_embeddings)
            processed_books.append(name)
            print(f"      ✅ Saved {len(book_chunks)} chunks")
    
    print(f"\n✅ Embedding generation complete:")
    print(f"   Total chunks: {len(all_chunks)}")
    print(f"   Books processed: {len(processed_books)}")
    print(f"   Embedding dimension: {all_embeddings[0].shape[0] if all_embeddings else 0}")
    
    return all_chunks, np.array(all_embeddings) if all_embeddings else None

# ============================================================
# STEP 5: SAVE TO KNOWLEDGE
# ============================================================

def save_to_knowledge(chunks, embeddings):
    """Save chunks and embeddings to knowledge folder."""
    print_header("💾 STEP 5: SAVING TO KNOWLEDGE")
    
    os.makedirs(KNOWLEDGE_DIR, exist_ok=True)
    
    # Save chunks
    with open(CHUNKS_FILE, 'w', encoding='utf-8') as f:
        for chunk in chunks:
            f.write(chunk + "\n---\n")
    print(f"✅ Saved chunks: {CHUNKS_FILE} ({len(chunks)} chunks)")
    
    # Save embeddings
    if embeddings is not None:
        np.save(EMBEDDINGS_FILE, embeddings)
        print(f"✅ Saved embeddings: {EMBEDDINGS_FILE} ({embeddings.shape})")
    
    # Save metadata
    with open(METADATA_FILE, 'w', encoding='utf-8') as f:
        f.write(f"Pipeline Run: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Total chunks: {len(chunks)}\n")
        f.write(f"Embedding dimension: {embeddings.shape[1] if embeddings is not None else 0}\n")
        f.write(f"Chunk size: {CHUNK_SIZE}\n")
        f.write(f"Chunk overlap: {CHUNK_OVERLAP}\n")
    print(f"✅ Saved metadata: {METADATA_FILE}")
    
    # Save progress (for resuming)
    progress = {
        "chunks": chunks,
        "embeddings": embeddings.tolist() if embeddings is not None else [],
        "book_counter": len(chunks),
        "current_book": None,
        "chunk_index": 0,
        "timestamp": datetime.now().isoformat()
    }
    with open(PROGRESS_FILE, 'wb') as f:
        pickle.dump(progress, f)
    print(f"✅ Saved progress: {PROGRESS_FILE}")

# ============================================================
# STEP 6: VERIFY & REPORT
# ============================================================

def verify_knowledge():
    """Verify the knowledge base was created correctly."""
    print_header("🔍 STEP 6: VERIFYING KNOWLEDGE")
    
    issues = []
    
    # Check chunks file
    if CHUNKS_FILE.exists():
        with open(CHUNKS_FILE, 'r', encoding='utf-8') as f:
            content = f.read()
            chunks = [c.strip() for c in content.split('---') if c.strip()]
        print(f"✅ Chunks file: {len(chunks)} chunks")
        if len(chunks) == 0:
            issues.append("No chunks found in chunks.txt")
    else:
        issues.append("chunks.txt not found")
    
    # Check embeddings
    if EMBEDDINGS_FILE.exists():
        try:
            embeddings = np.load(EMBEDDINGS_FILE, allow_pickle=True)
            print(f"✅ Embeddings: {embeddings.shape}")
            if len(embeddings) == 0:
                issues.append("Embeddings file is empty")
        except Exception as e:
            issues.append(f"Embeddings file corrupted: {e}")
    else:
        issues.append("embeddings.npy not found")
    
    # Check processed books
    if PROCESSED_FILE.exists():
        with open(PROCESSED_FILE, 'r', encoding='utf-8') as f:
            books = [line.strip() for line in f if line.strip()]
        print(f"✅ Processed books: {len(books)}")
    else:
        print("⚠️ No processed_books.txt (first run)")
    
    # Report issues
    if issues:
        print(f"\n⚠️ Issues found:")
        for issue in issues:
            print(f"   ❌ {issue}")
    else:
        print("\n✅ All checks passed! Knowledge base is ready.")

def report_summary(chunks, embeddings):
    """Print final summary."""
    print("\n" + "="*60)
    print("   ✅ PIPELINE COMPLETE!")
    print("="*60)
    print(f"\n   📁 Knowledge folder: {KNOWLEDGE_DIR}")
    print(f"   📄 Total chunks: {len(chunks)}")
    if embeddings is not None:
        print(f"   📊 Embedding dimension: {embeddings.shape[1]}")
    print(f"   💾 Folder size: {get_folder_size(KNOWLEDGE_DIR):.1f} MB")
    print("\n   🚀 Next steps:")
    print("   │")
    print("   ├── 💬 Chat:   python isaac.py")
    print("   ├── 🔍 Search: python search.py 'your question'")
    print("   └── 📚 Add more: python pdf_to_vectors.py")
    print("="*60)

def get_folder_size(path):
    """Get folder size in MB."""
    if not path.exists():
        return 0
    total = 0
    for f in path.glob('*'):
        if f.is_file():
            total += f.stat().st_size
    return total / (1024 * 1024)

# ============================================================
# MAIN PIPELINE
# ============================================================

def main():
    print(BANNER)
    print(f"📂 ISAAC Directory: {ISAAC_DIR}")
    print(f"📁 Knowledge Directory: {KNOWLEDGE_DIR}")
    print(f"📝 Text Directory: {TEXT_DIR}")
    print(f"📊 Min PDF size: {MIN_PDF_SIZE/(1024*1024):.0f} MB")
    print(f"📄 Chunk size: {CHUNK_SIZE}")
    print("="*60 + "\n")
    
    # Step 1: Find all PDFs
    pdf_files = find_all_pdfs()
    if not pdf_files:
        print("❌ No PDFs found!")
        return
    
    # Step 2: Convert to text
    converted = convert_pdfs_to_text(pdf_files)
    if converted == 0 and not any(TEXT_DIR.glob("*.txt")):
        print("❌ No PDFs could be converted!")
        return
    
    # Step 3: Load text files
    texts = load_text_files()
    if not texts:
        print("❌ No text files loaded!")
        return
    
    # Step 4: Generate embeddings
    chunks, embeddings = generate_embeddings(texts)
    if chunks is None:
        print("❌ Embedding generation failed!")
        return
    
    # Step 5: Save to knowledge
    save_to_knowledge(chunks, embeddings)
    
    # Step 6: Verify
    verify_knowledge()
    
    # Summary
    report_summary(chunks, embeddings)

if __name__ == "__main__":
    main()
