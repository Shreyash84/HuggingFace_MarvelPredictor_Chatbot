import requests
import os
from dotenv import load_dotenv

load_dotenv()

API_URL = "https://router.huggingface.co/v1/chat/completions"
API_KEY = os.getenv('API_KEY')
headers = {
    "Authorization": f"Bearer {API_KEY}",
}

messages = [
        {"role": "system", "content" : "You are a predictor of Marvel stories."},
    ]
    
def query(messages, temperature=0, max_tokens=100):

    payload = {
        "model" : "meta-llama/Meta-Llama-3-8B-Instruct:novita",
        "messages" : messages,
        "temperature" : temperature,
        "max_tokens" : max_tokens
    }

    try:
        response = requests.post(API_URL, headers=headers, json=payload)
        data = response.json()
    except Exception as e:
        print(f"❌ Request failed: {e}")
        return "Sorry, I couldn’t connect to the model."

    # Debug: Show the raw API output if something goes wrong
    if "choices" not in data:
        print("⚠️ API Error Response:", data)
        return "Hmm, I didn’t get a proper response from the AI."

    assistant_message = data["choices"][0]["message"]["content"]
    
    return assistant_message

# Implementing Chat Loop:

print("What you want to predict in Marvel universe: (type 'exit' to stop)\n")
while True:
    prompt = input("Your Turn: ")
    
    if prompt.lower() == "exit":
        print("Nice Conversation!! Bye-bye")
        break
    
    #Add user prompts to the messages list(conversation history)
    
    messages.append({"role": "user", "content": prompt})
    
    result = query(messages)   #Model reply

    print((f"\nAssistant: {result}\n"))
    messages.append({"role": "assistant", "content": result})
    
#Timestamp, error handling, Status codes