#!/usr/bin/env python3
"""
Extract text from large PDFs (≥5 MB) to dedicated folder.
Output: ~/ISAAC_AI/texts/BookName.txt
Usage: python extract_pdfs.py
"""

import os
import subprocess
import sys
import re
from pathlib import Path

# ---------- CONFIG ----------
HOME = Path.home()
TEXT_DIR = HOME / "ISAAC_AI" / "texts"
TEXT_DIR.mkdir(parents=True, exist_ok=True)

SCAN_DIRS = [
    "/storage/emulated/0",
    "/storage/105B-F7AB"
]

MIN_PDF_SIZE = 5 * 1024 * 1024  # 5 MB

SKIP_DIRS = [
    "Android", "Download", "Music", "Pictures", "Movies", "DCIM",
    "WhatsApp", "Telegram", "Instagram", "Facebook", "TikTok",
    "cache", "tmp", "temp", ".trash", "Lost+Found"
]
# ----------------------------

def should_skip(path):
    path_str = str(path)
    for skip in SKIP_DIRS:
        if skip in path_str:
            return True
    return False

def clean_filename(name):
    """Remove common ebook suffixes and clean up"""
    # Remove Anna's Archive suffix
    name = re.sub(r'--\s*Anna’s Archive\s*$', '', name)
    name = re.sub(r'--\s*Anna\'s Archive\s*$', '', name)
    # Remove trailing numbers and hashes
    name = re.sub(r'--\s*[a-f0-9]{32}\s*$', '', name)
    name = re.sub(r'--\s*\d+\s*$', '', name)
    # Remove extra spaces, dashes, and underscores
    name = re.sub(r'[_\s]+', ' ', name)
    name = re.sub(r'\s+', ' ', name)
    name = name.strip()
    # Limit length
    if len(name) > 60:
        name = name[:60]
    return name

def extract_pdf(pdf_path):
    """Extract text from PDF to ~/ISAAC_AI/texts/"""
    # Clean filename
    base_name = clean_filename(pdf_path.stem)
    txt_path = TEXT_DIR / f"{base_name}.txt"
    
    # Avoid overwriting if exists
    if txt_path.exists() and txt_path.stat().st_size > 1024:
        return True, "already exists"
    
    # Extract using pdftotext
    cmd = ['pdftotext', '-enc', 'UTF-8', str(pdf_path), str(txt_path)]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode == 0:
            if txt_path.exists() and txt_path.stat().st_size > 1024:
                # Sanitize the text file
                sanitize_text_file(txt_path)
                return True, "success"
            else:
                txt_path.unlink(missing_ok=True)
                return False, "empty or no text"
        else:
            return False, f"pdftotext error"
    except Exception as e:
        return False, str(e)[:50]

def sanitize_text_file(txt_path):
    """Remove control characters and fix encoding"""
    try:
        with open(txt_path, 'r', encoding='utf-8', errors='ignore') as f:
            text = f.read()
        # Remove control characters except newline/tab
        text = ''.join(c for c in text if c >= ' ' or c in '\n\t')
        # Remove null bytes
        text = text.replace('\x00', '')
        with open(txt_path, 'w', encoding='utf-8') as f:
            f.write(text)
    except:
        pass  # If it fails, keep the file as-is

def main():
    print("\n" + "="*60)
    print("   PDF Extractor → ~/ISAAC_AI/texts/")
    print("   Only PDFs ≥ 5 MB")
    print("="*60 + "\n")
    
    print(f"📂 Output folder: {TEXT_DIR}\n")
    
    total_pdfs = 0
    extracted = 0
    failed = 0
    skipped = 0
    
    for root in SCAN_DIRS:
        root_path = Path(root)
        if not root_path.exists():
            print(f"⚠️ Skipping {root} (does not exist)")
            continue
        
        print(f"📁 Scanning: {root}")
        pdfs = list(root_path.rglob('*.pdf'))
        print(f"   Found {len(pdfs)} PDFs total")
        
        for pdf_path in pdfs:
            if should_skip(pdf_path):
                skipped += 1
                continue
            
            # Check size
            try:
                size = pdf_path.stat().st_size
                if size < MIN_PDF_SIZE:
                    skipped += 1
                    continue
            except:
                skipped += 1
                continue
            
            total_pdfs += 1
            status, msg = extract_pdf(pdf_path)
            if status:
                extracted += 1
                print(f"   ✅ {pdf_path.name[:40]}... -> {msg}")
            else:
                failed += 1
                print(f"   ❌ {pdf_path.name[:40]}... -> {msg}")
    
    print("\n" + "="*60)
    print("   Summary")
    print("="*60)
    print(f"   Large PDFs found: {total_pdfs}")
    print(f"   Successfully extracted: {extracted}")
    print(f"   Failed: {failed}")
    print(f"   Skipped (small or system dirs): {skipped}")
    print(f"\n📂 All text files saved to: {TEXT_DIR}")
    print("\n💡 Run './learner' now to process these .txt files.")

if __name__ == "__main__":
    main()
