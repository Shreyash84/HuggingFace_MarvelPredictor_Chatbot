import requests
import streamlit as st
import os
from dotenv import load_dotenv
import time, json

load_dotenv()

API_URL = "https://router.huggingface.co/v1/chat/completions"
API_KEY = os.getenv('API_KEY')
headers = {
    "Authorization": f"Bearer {API_KEY}",
}

# Mode selector
modes = {
    "summarizer": "You are a professional text summarizer. Provide a complete, concise, and meaningful answer within {max_tokens} tokens.",
    "translator": "You are a language translator. Provide a complete translation within {max_tokens} tokens, keeping the context intact.",
    "story generator": "You are a creative story writer. Write a self-contained story that fits within {max_tokens} tokens without getting cut off.",
    "marvel predictor": "You are a predictor of Marvel stories. Predict interesting storylines based on input. Provide a full and coherent prediction within {max_tokens} tokens."
}


def choose_mode():
    print("\nChoose a mode: ")
    for mode in modes:
        print(f"- {mode}")
    mode = input("\nEnter mode: ").strip().lower()
    return mode if mode in modes else "marvel predictor"


def query(messages, temperature=0.7, max_tokens=100):
    # Inject explicit token-aware system prompt
    system_instruction = {
        "role": "system",
        "content": messages[0]["content"].format(max_tokens=max_tokens)
    }

    updated_messages = [system_instruction] + messages[1:]

    payload = {
        "model": "meta-llama/Meta-Llama-3-8B-Instruct:novita",
        "messages": updated_messages,
        "temperature": temperature,
        "max_tokens": max_tokens
    }

    # 🕒 Start timing
    start_time = time.time()

    try:
        response = requests.post(API_URL, headers=headers, json=payload)
        elapsed_time = time.time() - start_time  # ⏱ Total duration
        data = response.json()

        if response.status_code != 200:
            return {
                "error": f"Error {response.status_code}: {response.text}",
                "time": elapsed_time,
                "usage": {}
            }

    except Exception as e:
        elapsed_time = time.time() - start_time
        return {
            "error": f"Request Failed: {e}",
            "time": elapsed_time,
            "usage": {}
        }

    # 🧩 Extract response & usage safely
    assistant_message = data.get("choices", [{}])[0].get("message", {}).get("content", "")
    usage = data.get("usage", {})

    # Handle incomplete response
    if len(assistant_message.split()) > 10 and not assistant_message.endswith(('.', '!', '?')):
        assistant_message += " [...] (response trimmed due to token limit)"

    return {
        "response": assistant_message,
        "time": round(elapsed_time, 2),
        "usage": usage
    }


def save_response_local(prompt, assistant_response, filename="chat_log.json"):
    log_entry = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "user": prompt,
        "assistant": assistant_response.get("response", ""),
        "response_time_sec": assistant_response.get("time", 0),
        "token_usage": assistant_response.get("usage", {})
    }

    if os.path.exists(filename):
        with open(filename, "r", encoding="utf-8") as f:
            logs = json.load(f)
    else:
        logs = []

    logs.append(log_entry)

    with open(filename, "w", encoding="utf-8") as f:
        json.dump(logs, f, indent=4, ensure_ascii=False)


def print_in_chunks(text, chunk_size=500):
    for i in range(0, len(text), chunk_size):
        print(text[i:i + chunk_size])
        time.sleep(0.1)


# ------------------- MAIN (Terminal response)-------------------
# mode = choose_mode()
# print(f"\n🧠 Current Mode: {mode}\n")

# messages = [{"role": "system", "content": modes[mode]}]

# print("Type 'exit' to stop\n")

# while True:
#     prompt = input("Your turn: ")

#     if prompt.lower() == "exit":
#         print("Nice Conversation!! Bye-bye 👋")
#         break

#     messages.append({"role": "user", "content": prompt})

#     assistant_data = query(messages, max_tokens=100)

#     if "error" in assistant_data:
#         print(f"\n⚠️ Error: {assistant_data['error']}")
#         continue

#     print("\nAssistant:")
#     print_in_chunks(assistant_data["response"])
#     print(f"\n⏱ Response Time: {assistant_data['time']} seconds")
#     print(f"📊 Token Usage: {assistant_data.get('usage', {})}\n")

#     messages.append({"role": "assistant", "content": assistant_data["response"]})
#     save_response_local(prompt, assistant_data)

# ------------------- STREAMLIT UI -------------------
st.set_page_config(page_title="Mini AI App", page_icon="🤖", layout="centered")
st.title("🤖 Mini AI Application")

st.markdown("This mini AI app lets you **summarize, translate, generate stories, or predict Marvel plots** using Hugging Face models. Chat conversationally—the AI remembers previous messages!")

# Initialize session state for chat
if "messages" not in st.session_state:
    st.session_state.messages = []
if "current_mode" not in st.session_state:
    st.session_state.current_mode = "marvel predictor"  # Default like terminal fallback
    st.session_state.messages = [{"role": "system", "content": modes[st.session_state.current_mode]}]

# Sidebar
st.sidebar.header("⚙️ Settings")
selected_mode = st.sidebar.selectbox("Choose a mode:", list(modes.keys()), index=list(modes.keys()).index(st.session_state.current_mode))
max_tokens = st.sidebar.slider("Max Tokens", 50, 500, 100, step=50)
temperature = st.sidebar.slider("Creativity (Temperature)", 0.1, 1.5, 0.7, step=0.1)

# Handle mode change: reset messages if mode changed
if selected_mode != st.session_state.current_mode:
    st.session_state.current_mode = selected_mode
    st.session_state.messages = [{"role": "system", "content": modes[st.session_state.current_mode]}]
    st.rerun()

# Display chat messages from history (skip system)
for message in st.session_state.messages[1:]:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat input
if prompt := st.chat_input("Type your message here (or 'exit' to clear chat):"):
    # Append user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    # Display user message
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # Exit command: clear messages (mimic terminal exit by resetting to system only)
    if prompt.lower() == "exit":
        st.session_state.messages = [{"role": "system", "content": modes[st.session_state.current_mode]}]
        st.success("Chat cleared! 👋 Start a new conversation.")
        st.rerun()
    
    # Generate response
    with st.chat_message("assistant"):
        with st.spinner("Generating response... ⏳"):
            result = query(st.session_state.messages, temperature=temperature, max_tokens=max_tokens)
        
        if "error" in result:
            st.error(f"❌ Generation failed: {result['error']}")
        else:
            response = result["response"]
            st.markdown(response)
            
            # Append assistant message to history
            st.session_state.messages.append({"role": "assistant", "content": response})
            
            # Show metadata
            with st.container():
                st.markdown("---")
                st.markdown(f"**⏱ Response Time:** {result['time']} seconds")
                st.markdown(f"**📊 Token Usage:** {result.get('usage', {})}")
                st.markdown("---")
            
            # Save to local log
            save_response_local(prompt, result)

# Previous logs expander (persistent across sessions)
with st.expander("📜 View Previous Sessions Logs"):
    if os.path.exists("chat_log.json"):
        with open("chat_log.json", "r", encoding="utf-8") as f:
            logs = json.load(f)
            if logs:
                for log in reversed(logs[-5:]):  # Show last 5 newest first
                    st.markdown(f"**🕒 {log['timestamp']}**")
                    st.markdown(f"**You:** {log['user']}")
                    st.markdown(f"**AI:** {log['assistant']}")
                    token_str = str(log['token_usage']) if isinstance(log['token_usage'], dict) else log['token_usage']
                    st.markdown(f"**Time:** {log['response_time_sec']} sec | **Tokens:** {token_str}")
                    st.markdown("---")
            else:
                st.info("No logs yet!")
    else:
        st.info("No logs yet!")