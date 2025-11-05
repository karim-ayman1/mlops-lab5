import os
import requests
import gradio as gr
from fastapi import FastAPI
from ollama import Client
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Ollama local endpoint
LOCAL_OLLAMA_ENDPOINT = "http://localhost:11434/api/generate"

# Initialize cloud client (optional)
try:
    api_key = os.environ.get('OLLAMA_API_KEY')
    if not api_key:
        print("⚠️  Warning: OLLAMA_API_KEY not found. Cloud models will not work.")
        ollama_cloud_client = None
    else:
        ollama_cloud_client = Client(
            host="https://ollama.com",
            headers={'Authorization': f'Bearer {api_key}'}
        )
except Exception as e:
    print(f"Error initializing Ollama Cloud client: {e}")
    ollama_cloud_client = None


# -------------------------------
# Main text generation function
# -------------------------------
def generate_text(prompt: str, model_name: str):
    """Generate text from either a local Ollama model or a cloud one."""
    if not prompt:
        return "Please enter a prompt."

    try:
        # If model name contains 'cloud', use cloud API
        if 'cloud' in model_name:
            if not ollama_cloud_client:
                return "Error: OLLAMA_API_KEY not set in .env file."
            messages = [{'role': 'user', 'content': prompt}]
            response = ollama_cloud_client.chat(
                model_name.replace('-cloud', ''), 
                messages=messages
            )
            return response['message']['content']

        # Otherwise, use local Ollama API
        data = {
            "model": model_name,
            "prompt": prompt,
            "stream": False
        }

        response = requests.post(LOCAL_OLLAMA_ENDPOINT, json=data)
        response.raise_for_status()

        # Ollama’s local API may return different keys depending on version
        res_json = response.json()
        if "response" in res_json:
            return res_json["response"]
        elif "text" in res_json:
            return res_json["text"]
        else:
            return str(res_json)

    except Exception as e:
        return f"An error occurred: {str(e)}"


# -------------------------------
# Gradio UI definition
# -------------------------------
model_dropdown = gr.Dropdown(
    choices=[
        "gemma:2b",           # Local small model (recommended)
        "stable-code:3b",     # Local coding model alternative
        "phi3:3.8b",          # Balanced local model
        "gpt-oss:120b-cloud"  # Example cloud model (requires key)
    ],
    value="gemma:2b",
    label="Select Model"
)

gui = gr.Interface(
    fn=generate_text,
    inputs=[
        gr.Textbox(
            lines=3,
            label="Your Prompt",
            placeholder="Enter a question or request..."
        ),
        model_dropdown
    ],
    outputs=gr.Textbox(label="Generated Text", lines=12),
    title="Local AI Assistant — Ollama + Gradio",
    description=(
        "💡 Select a model and enter your prompt. "
        "'gemma:2b' runs locally for fast lightweight responses. "
        "Cloud models require an API key."
    )
)

# -------------------------------
# FastAPI + Gradio integration
# -------------------------------
app = FastAPI(title="Ollama LLM API")
app = gr.mount_gradio_app(app, gui, path="/")

# Run directly with:  uvicorn app.main:app --reload
# or just python app/main.py to test locally

if __name__ == "__main__":
    gui.launch(server_name="0.0.0.0", server_port=7860)
