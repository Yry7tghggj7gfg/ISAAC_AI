#!/usr/bin/env python3
"""
ISAAC AI – Smart Assistant (Think First, Search Later)

  🧠 Stage 1: Can Qwen answer from its own knowledge?
  📚 Stage 2: If not, search your books (RAG)
  💬 Stage 3: Combine and respond naturally
  🔄 Stage 4: Auto-start Qwen if not running

"""

import os
import sys
import time
import json
import pickle
import urllib.request
import subprocess
import numpy as np
from pathlib import Path
from datetime import datetime

# ============================================================
# CONFIGURATION
# ============================================================

HOME = Path.home()
ISAAC_DIR = HOME / "ISAAC_AI"
CHUNKS_FILE = ISAAC_DIR / "chunks" / "all_chunks.txt"

# Try multiple possible locations for embeddings
EMBEDDINGS_PATHS = [
    ISAAC_DIR / "knowledge" / "embeddings.npy",
    HOME / "knowledge" / "embeddings.npy",
    ISAAC_DIR / "knowledge_multi" / "embeddings.npy",
]

QWEN_PORT = 8081

# Qwen paths
QWEN_MODEL = None
for path in [
    HOME / "qwen2.5-1.5b-instruct-q4_k_m.gguf",
    ISAAC_DIR / "qwen2.5-1.5b-instruct-q4_k_m.gguf",
]:
    if path.exists():
        QWEN_MODEL = path
        break

LLAMA_SERVER = None
for path in [
    HOME / "llama.cpp/build/bin/llama-server",
    HOME / "llama.cpp/llama-server",
    ISAAC_DIR / "llama-server",
]:
    if path.exists():
        LLAMA_SERVER = path
        break

# ============================================================
# AUTO-START QWEN
# ============================================================

def is_qwen_running():
    """Check if Qwen server is running."""
    try:
        urllib.request.urlopen(f"http://localhost:{QWEN_PORT}/health", timeout=2)
        return True
    except:
        return False

def start_qwen():
    """Auto-start Qwen if not running."""
    if is_qwen_running():
        print("✅ Qwen2.5 already running")
        return True
    
    print("⏳ Auto-starting Qwen2.5...")
    
    if not LLAMA_SERVER or not QWEN_MODEL:
        print("❌ Qwen not found! Please check paths.")
        return False
    
    try:
        subprocess.Popen([
            str(LLAMA_SERVER),
            "-m", str(QWEN_MODEL),
            "-c", "2048",
            "-t", "4",
            "-ngl", "0",
            "--port", str(QWEN_PORT)
        ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        
        for i in range(15):
            time.sleep(1)
            if is_qwen_running():
                print("✅ Qwen2.5 started!")
                return True
            print(f"   Waiting... {i+1}/15")
        
        print("❌ Failed to start Qwen")
        return False
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

# ============================================================
# STAGE 1: THINK (Pure Qwen) – LESS STRICT
# ============================================================

def think_with_qwen(question):
    """Stage 1: Ask Qwen to answer from its own knowledge."""
    print("🧠 Stage 1: Thinking...", end=" ", flush=True)
    
    prompt = f"""<|im_start|>system
You are ISAAC, a helpful and friendly AI assistant. Answer the user's question naturally.
You can answer from your general knowledge. If you're not sure, say so honestly.
Be conversational and helpful.
<|im_end|>
<|im_start|>user
{question}
<|im_end|>
<|im_start|>assistant
"""
    
    try:
        data = json.dumps({
            "prompt": prompt,
            "n_predict": 200,
            "temperature": 0.7,
            "stop": ["<|im_end|>"]
        }).encode()
        
        req = urllib.request.Request(
            f"http://localhost:{QWEN_PORT}/completion",
            data=data,
            headers={"Content-Type": "application/json"}
        )
        resp = urllib.request.urlopen(req, timeout=30)
        result = json.loads(resp.read().decode())
        answer = result.get("content", "").strip()
        
        if not answer or len(answer) < 5:
            print("❌ No response.")
            return None
        
        # Only reject if it clearly says it doesn't know
        doesnt_know = [
            "don't have enough information",
            "don't know",
            "not sure",
            "I can't answer",
            "I'm not confident"
        ]
        
        for phrase in doesnt_know:
            if phrase.lower() in answer.lower()[:50]:
                print("❌ Not confident.")
                return None
        
        print("✅ Confident!")
        return answer
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return None

# ============================================================
# STAGE 2: SEARCH (RAG from books)
# ============================================================

def load_knowledge():
    """Load chunks and embeddings from knowledge base."""
    print("📚 Loading knowledge base...", end=" ", flush=True)
    
    if not CHUNKS_FILE.exists():
        print("❌ No chunks found!")
        return None, None
    
    # Find embeddings
    embeddings_path = None
    for path in EMBEDDINGS_PATHS:
        if path.exists():
            embeddings_path = path
            break
    
    if embeddings_path is None:
        print("❌ No embeddings found!")
        print("   Checked:", [str(p) for p in EMBEDDINGS_PATHS])
        return None, None
    
    # Load chunks
    with open(CHUNKS_FILE, 'r', encoding='utf-8') as f:
        content = f.read()
        chunks = [c.strip() for c in content.split('---') if c.strip()]
    
    # Load embeddings
    embeddings = np.load(embeddings_path, allow_pickle=True)
    
    print(f"✅ {len(chunks)} chunks loaded")
    print(f"   Embeddings: {embeddings_path}")
    return chunks, embeddings

def search_books(question, chunks, embeddings):
    """Stage 2: Search the knowledge base."""
    print("📚 Stage 2: Searching books...", end=" ", flush=True)
    
    try:
        from fastembed import TextEmbedding
    except ImportError:
        print("❌ FastEmbed not installed!")
        return None
    
    try:
        model = TextEmbedding("BAAI/bge-small-en-v1.5")
        query_emb = np.array(list(model.embed([question]))[0])
        
        similarities = np.dot(embeddings, query_emb)
        top_idx = np.argsort(similarities)[-5:][::-1]
        
        context = []
        for idx in top_idx:
            if similarities[idx] > 0.25:
                context.append(chunks[idx])
        
        if not context:
            print("❌ No relevant context.")
            return None
        
        print(f"✅ Found {len(context)} relevant chunks.")
        return "\n\n".join(context)
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return None

def generate_answer(question, context):
    """Stage 3: Generate final answer from context."""
    print("📝 Stage 3: Generating answer...", end=" ", flush=True)
    
    prompt = f"""<|im_start|>system
You are ISAAC, a helpful assistant. Answer the question using the provided context.
If the context doesn't fully answer, use your general knowledge to complement it.
<|im_end|>
<|im_start|>user
Context:
{context}

Question: {question}
<|im_end|>
<|im_start|>assistant
"""
    
    try:
        data = json.dumps({
            "prompt": prompt,
            "n_predict": 400,
            "temperature": 0.3,
            "stop": ["<|im_end|>"]
        }).encode()
        
        req = urllib.request.Request(
            f"http://localhost:{QWEN_PORT}/completion",
            data=data,
            headers={"Content-Type": "application/json"}
        )
        resp = urllib.request.urlopen(req, timeout=60)
        result = json.loads(resp.read().decode())
        answer = result.get("content", "").strip()
        print("✅ Done!")
        return answer
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return None

# ============================================================
# MAIN SMART CHAT
# ============================================================

def smart_chat():
    print("\n" + "="*60)
    print("   🧠 ISAAC AI – Smart Assistant")
    print("   💭 Think first, search second")
    print("   📚 Knowledge base ready")
    print("   💡 Type 'exit' to quit")
    print("="*60 + "\n")
    
    # Auto-start Qwen
    if not is_qwen_running():
        if not start_qwen():
            print("❌ Could not start Qwen.\n")
            return
    
    # Load knowledge base
    chunks, embeddings = load_knowledge()
    if chunks is None:
        print("⚠️ Knowledge base not ready. Running in pure Qwen mode.\n")
    
    while True:
        try:
            question = input("You: ").strip()
            if not question:
                continue
            if question.lower() in ['exit', 'quit', 'bye']:
                print("\n👋 Goodbye!")
                break
            
            print("\n" + "-"*40)
            
            # STAGE 1: THINK
            answer = think_with_qwen(question)
            
            if answer:
                print(f"\n🤖 ISAAC:\n{answer}\n")
                print("-"*40 + "\n")
                continue
            
            # STAGE 2: SEARCH
            if chunks is not None and embeddings is not None:
                context = search_books(question, chunks, embeddings)
                
                if context:
                    answer = generate_answer(question, context)
                    if answer:
                        print(f"\n📚 ISAAC (From Books):\n{answer}\n")
                        print("-"*40 + "\n")
                        continue
            
            print("\n🤖 ISAAC: I don't have information about that.\n")
            print("-"*40 + "\n")
            
        except KeyboardInterrupt:
            print("\n\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"❌ Error: {e}")

if __name__ == "__main__":
    smart_chat()
