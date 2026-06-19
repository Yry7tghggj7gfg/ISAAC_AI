#!/usr/bin/env python3
"""
ISAAC AI Chat – One‑command chat with Qwen (or any llama.cpp model).
Automatically starts the server, waits for it to be ready, then gives you a prompt.
Type 'exit' to quit, 'rag' to switch to book‑based answers (if isaac_manager.py exists).
"""

import subprocess
import time
import sys
import os
import json
import urllib.request
import urllib.error
import shutil
from pathlib import Path

# -------------------- CONFIGURATION --------------------
MODEL_FILE = "qwen2.5-1.5b-instruct-q4_k_m.gguf"
LLAMA_SERVER = "/data/data/com.termux/files/home/llama.cpp/build/bin/llama-server"
PORT = 8081
CONTEXT = 2048
THREADS = 4
GPU_LAYERS = 0
# ------------------------------------------------------

def find_model():
    """Look for the model in current dir, then in home."""
    if Path(MODEL_FILE).exists():
        return MODEL_FILE
    home_model = Path.home() / MODEL_FILE
    if home_model.exists():
        return str(home_model)
    return None

def server_running():
    """Check if llama-server is responding on the given port."""
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{PORT}/health", timeout=2) as resp:
            return resp.getcode() == 200
    except:
        return False

def start_server(model_path):
    """Launch llama-server in the background."""
    cmd = [
        LLAMA_SERVER,
        "-m", model_path,
        "-c", str(CONTEXT),
        "-t", str(THREADS),
        "-ngl", str(GPU_LAYERS),
        "--port", str(PORT)
    ]
    # Redirect output to /dev/null to keep the terminal clean
    with open(os.devnull, 'w') as devnull:
        proc = subprocess.Popen(cmd, stdout=devnull, stderr=devnull)
    return proc

def ask(question):
    """Send a chat completion request to the server and return the answer."""
    data = {
        "messages": [{"role": "user", "content": question}],
        "stream": False
    }
    req = urllib.request.Request(
        f"http://127.0.0.1:{PORT}/chat/completions",
        data=json.dumps(data).encode('utf-8'),
        headers={"Content-Type": "application/json"}
    )
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            result = json.loads(resp.read().decode())
            return result.get("choices", [{}])[0].get("message", {}).get("content", "")
    except Exception as e:
        return f"Error: {e}"

def run_chat():
    print("\n🤖 ISAAC AI Chat (Qwen 1.5B) – type 'exit' to quit, 'rag' for book answers.\n")
    while True:
        try:
            q = input("You: ").strip()
            if q.lower() in ("exit", "quit"):
                break
            if q.lower() == "rag":
                run_rag()
                continue
            if not q:
                continue
            print("AI: ", end="", flush=True)
            answer = ask(q)
            print(answer)
        except KeyboardInterrupt:
            print("\nGoodbye!")
            break

def run_rag():
    """Try to run isaac_manager.py for RAG answers."""
    rag_script = Path("isaac_manager.py")
    if not rag_script.exists():
        print("RAG script not found in current directory.")
        return
    print("📚 Switching to RAG mode (answers from your books)...")
    # We'll just execute it and let it take over the terminal.
    # The user can return to this chat by exiting that script.
    try:
        subprocess.run([sys.executable, str(rag_script)], check=False)
    except Exception as e:
        print(f"Could not run isaac_manager.py: {e}")

def main():
    # 1. Locate model
    model_path = find_model()
    if not model_path:
        print(f"❌ Model file '{MODEL_FILE}' not found in current directory or home.")
        print("   Please download it first or adjust MODEL_FILE in the script.")
        sys.exit(1)

    # 2. Check if server is running
    if not server_running():
        print("⏳ Starting llama-server (this may take ~10 seconds)...")
        start_server(model_path)
        # Wait until the server is ready
        for _ in range(30):  # up to 30 seconds
            time.sleep(1)
            if server_running():
                break
        else:
            print("❌ Server didn't start in time. Check logs manually.")
            sys.exit(1)
        print("✅ Server ready!")

    # 3. Run chat
    run_chat()

if __name__ == "__main__":
    main()
