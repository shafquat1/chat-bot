"""
AI Chatbot using Groq API
This chatbot uses Groq's fast inference API with Llama models
"""

import os
from groq import Groq

class AIChat:
    def __init__(self, api_key=None, model="llama-3.3-70b-versatile"):
        """
        Initialize the chatbot with Groq API
        
        Args:
            api_key: Your Groq API key (or set GROQ_API_KEY environment variable)
            model: Model to use (default: llama-3.3-70b-versatile)
        """
        self.api_key = api_key or os.environ.get("GROQ_API_KEY")
        if not self.api_key:
            raise ValueError("Please provide a Groq API key or set GROQ_API_KEY environment variable")
        
        self.client = Groq(api_key=self.api_key)
        self.model = model
        self.conversation_history = []
        
    def chat(self, user_message, system_prompt=None):
        """
        Send a message and get a response
        
        Args:
            user_message: The user's message
            system_prompt: Optional system prompt to guide the AI's behavior
            
        Returns:
            The AI's response
        """
        # Build messages array
        messages = []
        
        # Add system prompt if provided
        if system_prompt and not self.conversation_history:
            messages.append({
                "role": "system",
                "content": system_prompt
            })
        
        # Add conversation history
        messages.extend(self.conversation_history)
        
        # Add new user message
        messages.append({
            "role": "user",
            "content": user_message
        })
        
        # Get response from Groq
        chat_completion = self.client.chat.completions.create(
            messages=messages,
            model=self.model,
            temperature=0.7,
            max_tokens=1024,
            top_p=1,
            stream=False
        )
        
        # Extract the response
        assistant_message = chat_completion.choices[0].message.content
        
        # Update conversation history
        self.conversation_history.append({
            "role": "user",
            "content": user_message
        })
        self.conversation_history.append({
            "role": "assistant",
            "content": assistant_message
        })
        
        return assistant_message
    
    def stream_chat(self, user_message, system_prompt=None):
        """
        Send a message and stream the response
        
        Args:
            user_message: The user's message
            system_prompt: Optional system prompt
            
        Yields:
            Chunks of the AI's response
        """
        # Build messages array
        messages = []
        
        if system_prompt and not self.conversation_history:
            messages.append({
                "role": "system",
                "content": system_prompt
            })
        
        messages.extend(self.conversation_history)
        messages.append({
            "role": "user",
            "content": user_message
        })
        
        # Stream response from Groq
        stream = self.client.chat.completions.create(
            messages=messages,
            model=self.model,
            temperature=0.7,
            max_tokens=1024,
            top_p=1,
            stream=True
        )
        
        # Collect full response while streaming
        full_response = ""
        for chunk in stream:
            if chunk.choices[0].delta.content is not None:
                content = chunk.choices[0].delta.content
                full_response += content
                yield content
        
        # Update conversation history
        self.conversation_history.append({
            "role": "user",
            "content": user_message
        })
        self.conversation_history.append({
            "role": "assistant",
            "content": full_response
        })
    
    def clear_history(self):
        """Clear the conversation history"""
        self.conversation_history = []
    
    def get_history(self):
        """Get the conversation history"""
        return self.conversation_history


def main():
    """
    Main function to run the chatbot in terminal
    """
    print("=" * 60)
    print("AI Chatbot powered by Groq")
    print("=" * 60)
    print("Commands:")
    print("  - Type 'quit' or 'exit' to end the conversation")
    print("  - Type 'clear' to clear conversation history")
    print("  - Type 'history' to view conversation history")
    print("=" * 60)
    print()
    
    # Get API key
    api_key = input("Enter your Groq API key (or press Enter to use GROQ_API_KEY env var): ").strip()
    if not api_key:
        api_key = None
    
    # Initialize chatbot
    try:
        chatbot = AIChat(api_key=api_key)
        print("\n✓ Chatbot initialized successfully!")
        print(f"Using model: {chatbot.model}\n")
    except ValueError as e:
        print(f"\n✗ Error: {e}")
        return
    
    # Optional system prompt
    use_system_prompt = input("Would you like to set a system prompt? (y/n): ").strip().lower()
    system_prompt = None
    if use_system_prompt == 'y':
        system_prompt = input("Enter system prompt: ").strip()
    
    print("\n" + "=" * 60)
    print("Chat started! (streaming mode)")
    print("=" * 60 + "\n")
    
    # Chat loop
    while True:
        # Get user input
        user_input = input("You: ").strip()
        
        if not user_input:
            continue
        
        # Check for commands
        if user_input.lower() in ['quit', 'exit']:
            print("\nGoodbye! 👋")
            break
        
        if user_input.lower() == 'clear':
            chatbot.clear_history()
            print("\n✓ Conversation history cleared!\n")
            continue
        
        if user_input.lower() == 'history':
            print("\n" + "=" * 60)
            print("Conversation History:")
            print("=" * 60)
            for msg in chatbot.get_history():
                role = msg['role'].capitalize()
                content = msg['content']
                print(f"\n{role}: {content}")
            print("\n" + "=" * 60 + "\n")
            continue
        
        # Get and stream response
        print("\nAssistant: ", end="", flush=True)
        try:
            for chunk in chatbot.stream_chat(user_input, system_prompt if not chatbot.conversation_history else None):
                print(chunk, end="", flush=True)
            print("\n")
        except Exception as e:
            print(f"\n✗ Error: {e}\n")


if __name__ == "__main__":
    main()