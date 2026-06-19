#!/usr/bin/env python3
"""
ISAAC AI Control Center – menu‑driven interface for your offline AI.
Options:
  1. Start server (background, logs suppressed)
  2. Stop server
  3. Chat (clean interactive chat)
  4. RAG (answers from your books, runs isaac_manager.py)
  5. Status (check if server is running)
  6. Exit
"""

import subprocess
import time
import sys
import os
import json
import urllib.request
import urllib.error
from pathlib import Path

# ---------- CONFIG ----------
MODEL_FILE = "qwen2.5-1.5b-instruct-q4_k_m.gguf"
LLAMA_SERVER = "/data/data/com.termux/files/home/llama.cpp/build/bin/llama-server"
PORT = 8081
CONTEXT = 2048
THREADS = 4
GPU_LAYERS = 0
# ----------------------------

def find_model():
    if Path(MODEL_FILE).exists():
        return MODEL_FILE
    home_model = Path.home() / MODEL_FILE
    if home_model.exists():
        return str(home_model)
    return None

def server_running():
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{PORT}/health", timeout=2) as resp:
            return resp.getcode() == 200
    except:
        return False

def start_server():
    model = find_model()
    if not model:
        print("❌ Model file not found. Please download it first.")
        return False
    if server_running():
        print("✅ Server is already running.")
        return True
    print("⏳ Starting server (this may take ~10 seconds)...")
    cmd = [
        LLAMA_SERVER,
        "-m", model,
        "-c", str(CONTEXT),
        "-t", str(THREADS),
        "-ngl", str(GPU_LAYERS),
        "--port", str(PORT)
    ]
    with open(os.devnull, 'w') as devnull:
        subprocess.Popen(cmd, stdout=devnull, stderr=devnull)
    # Wait for readiness
    for _ in range(30):
        time.sleep(1)
        if server_running():
            print("✅ Server ready.")
            return True
    print("❌ Server failed to start.")
    return False

def stop_server():
    if not server_running():
        print("ℹ️ Server is not running.")
        return
    subprocess.run(["pkill", "-f", "llama-server"], check=False)
    time.sleep(1)
    if server_running():
        print("❌ Failed to stop server.")
    else:
        print("✅ Server stopped.")

def chat():
    if not server_running():
        print("❌ Server is not running. Start it first (option 1).")
        return
    print("\n🤖 Chat with Qwen – type 'exit' to return to menu.\n")
    while True:
        try:
            q = input("You: ").strip()
            if q.lower() in ("exit", "quit"):
                break
            if not q:
                continue
            print("AI: ", end="", flush=True)
            data = {
                "messages": [{"role": "user", "content": q}],
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
                    answer = result.get("choices", [{}])[0].get("message", {}).get("content", "")
                    print(answer)
            except Exception as e:
                print(f"Error: {e}")
        except KeyboardInterrupt:
            print("\nReturning to menu...")
            break

def run_rag():
    if not server_running():
        print("❌ Server is not running. Start it first (option 1).")
        return
    rag_script = Path("isaac_manager.py")
    if not rag_script.exists():
        print("❌ isaac_manager.py not found in current directory.")
        return
    print("📚 Running RAG (answers from your books)...")
    try:
        subprocess.run([sys.executable, str(rag_script)], check=False)
    except Exception as e:
        print(f"Error: {e}")

def status():
    if server_running():
        print(f"✅ Server is running on port {PORT}.")
        model = find_model()
        if model:
            print(f"   Model: {model}")
    else:
        print("❌ Server is not running.")

def main_menu():
    while True:
        print("\n" + "="*40)
        print("   ISAAC AI Control Center")
        print("="*40)
        print("  1. Start server")
        print("  2. Stop server")
        print("  3. Chat (clean interactive)")
        print("  4. RAG (book‑based answers)")
        print("  5. Status")
        print("  6. Exit")
        choice = input("Select option [1-6]: ").strip()

        if choice == "1":
            start_server()
        elif choice == "2":
            stop_server()
        elif choice == "3":
            chat()
        elif choice == "4":
            run_rag()
        elif choice == "5":
            status()
        elif choice == "6":
            print("Goodbye!")
            break
        else:
            print("Invalid option. Try again.")

if __name__ == "__main__":
    main_menu()
