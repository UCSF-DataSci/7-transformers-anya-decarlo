from gradio_client import Client
import argparse
import os

SPACE_NAME = "anyadecarlo/datasci223"

class LLMClient:
    def __init__(self, space_name=SPACE_NAME):
        self.client = Client(space_name)

    def get_response(self, prompt: str) -> str:
        try:
            return self.client.predict(prompt, api_name="/predict")
        except Exception as e:
            return f"[Error: {str(e)}]"

class LLMChatTool:
    def __init__(self, llm_client: LLMClient):
        self.llm_client = llm_client

    def chat(self):
        print("Welcome to the LLM Chat Tool! Type 'exit' to quit.")
        while True:
            user_input = input("\nYou: ")
            if user_input.lower() == 'exit':
                print("Goodbye!")
                break
            response = self.llm_client.get_response(user_input)
            print(f"LLM: {response}")

def main():
    parser = argparse.ArgumentParser(description="Chat with the LLM via Hugging Face Space")
    args = parser.parse_args()
    client = LLMClient()
    chat_tool = LLMChatTool(client)
    chat_tool.chat()

if __name__ == "__main__":
    main() 