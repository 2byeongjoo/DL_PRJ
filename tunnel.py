import gradio as gr
import requests

def proxy(payload):
    # 로컬 Ollama로 명령을 전달하고 결과를 받아옴
    response = requests.post("http://localhost:11434/api/generate", json=payload)
    return response.json()

# share=True가 핵심입니다. 이게 ngrok 역할을 대신 합니다.
demo = gr.Interface(fn=proxy, inputs="json", outputs="json")
demo.launch(share=True)