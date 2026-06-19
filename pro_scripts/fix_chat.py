import sys
from pathlib import Path

# Read the file
isaac_path = Path.home() / "ISAAC_AI" / "isaac.py"
with open(isaac_path, 'r') as f:
    content = f.read()

# Find the chat function and replace the prompt
old_prompt = '''prompt = f"""<|im_start|>system
You are ISAAC, a helpful AI assistant. Answer based on the provided context.
If the context doesn\'t contain the answer, say "I don\'t have information about that."
<|im_end|>
<|im_start|>user
Context:
{context}

Question: {query}
<|im_end|>
<|im_start|>assistant
"""'''

new_prompt = '''prompt = f"""<|im_start|>system\\nYou are ISAAC, a helpful AI assistant. Answer based on the provided context.\\nIf the context doesn\'t contain the answer, say "I don\'t have information about that."\\n<|im_end|>\\n<|im_start|>user\\nContext:\\n{context}\\n\\nQuestion: {query}\\n<|im_end|>\\n<|im_start|>assistant\\n"""'''

content = content.replace(old_prompt, new_prompt)

with open(isaac_path, 'w') as f:
    f.write(content)

print("✅ Fixed chat prompt format")
