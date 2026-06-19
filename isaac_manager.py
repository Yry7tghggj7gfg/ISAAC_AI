#!/usr/bin/env python3
"""
ISAAC AI Manager – One-Command Startup
Starts: llama-server, background learner, then chat.
"""

import os
import subprocess
import time
import sys
import signal
import socket
import requests
import glob
from pathlib import Path

HOME = os.path.expanduser("~")
ISAAC_DIR = os.path.join(HOME, "ISAAC_AI")
LOGS_DIR = os.path.join(ISAAC_DIR, "logs")
KNOWLEDGE_DIR = os.path.join(ISAAC_DIR, "knowledge")
SERVER_BIN = os.path.join(HOME, "llama.cpp/build/bin/llama-server")
MODEL_PATH = os.path.join(HOME, "llama.cpp/models/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf")
LEARNER_BIN = os.path.join(ISAAC_DIR, "learner")
ASK_BIN = os.path.join(ISAAC_DIR, "ask")
SERVER_HOST = "127.0.0.1"
SERVER_PORT = 8080
HEALTH_URL = f"http://{SERVER_HOST}:{SERVER_PORT}/health"

# Create directories
os.makedirs(LOGS_DIR, exist_ok=True)
os.makedirs(KNOWLEDGE_DIR, exist_ok=True)

def is_server_running():
    try:
        r = requests.get(HEALTH_URL, timeout=2)
        return r.status_code == 200
    except:
        return False

def start_server():
    if is_server_running():
        print("✅ llama-server already running")
        return

    print("🚀 Starting llama-server (this may take 10-15 seconds)...")
    log_path = os.path.join(LOGS_DIR, "llama_server.log")
    logfile = open(log_path, "w")
    proc = subprocess.Popen(
        [
            SERVER_BIN,
            "-m", MODEL_PATH,
            "--host", SERVER_HOST,
            "--port", str(SERVER_PORT),
            "--no-mmap",
            "--no-warmup",
            "-ngl", "0"
        ],
        stdout=logfile,
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )
    for _ in range(30):
        time.sleep(1)
        if is_server_running():
            print("✅ llama-server is ready")
            return
    print("❌ llama-server failed to start. Check", log_path)
    sys.exit(1)

def is_learner_running():
    try:
        output = subprocess.check_output(["pgrep", "-f", "learner"], text=True)
        return output.strip() != ""
    except:
        return False

def start_learner():
    if is_learner_running():
        print("✅ learner already running")
        return

    print("📚 Starting background learner...")
    log_path = os.path.join(LOGS_DIR, "learner.log")
    logfile = open(log_path, "w")
    proc = subprocess.Popen(
        [LEARNER_BIN],
        stdout=logfile,
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )
    time.sleep(2)
    if is_learner_running():
        print("✅ learner started successfully")
    else:
        print("⚠️  learner may not have started. Check", log_path)

def wait_for_knowledge():
    """Wait until at least one knowledge file exists."""
    print("⏳ Waiting for learner to create knowledge base...")
    for _ in range(120):  # wait up to 2 minutes
        files = glob.glob(os.path.join(KNOWLEDGE_DIR, "*.txt"))
        if files:
            print(f"✅ Knowledge base ready ({len(files)} documents)")
            return True
        time.sleep(2)
    print("❌ Knowledge base not ready after 2 minutes.")
    print("   Check if learner is running and has access to storage.")
    return False

def run_ask():
    if not os.path.exists(ASK_BIN):
        print(f"❌ {ASK_BIN} not found. Compile it first:")
        print("   cd ~/ISAAC_AI && g++ -o ask ask.cpp -std=c++17 -pthread")
        return

    print("\n💬 Starting chat interface... (type 'exit' to quit)\n")
    os.execv(ASK_BIN, [ASK_BIN])

def signal_handler(sig, frame):
    print("\n👋 Shutting down...")
    sys.exit(0)

if __name__ == "__main__":
    signal.signal(signal.SIGINT, signal_handler)

    print("\n╔══════════════════════════════════════════════════════╗")
    print("║           ISAAC AI – One-Command Startup              ║")
    print("║    Starts server, learner, and chat in one go         ║")
    print("╚══════════════════════════════════════════════════════╝\n")

    start_server()
    start_learner()

    # Wait for knowledge to exist
    if not wait_for_knowledge():
        print("🚫 Aborting. Please check learner logs.")
        sys.exit(1)

    run_ask()
