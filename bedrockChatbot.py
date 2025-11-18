# advanceChatBot_bedrock.py
"""
Fully working Streamlit chatbot using AWS Bedrock (boto3 runtime).
Default model: amazon.nova-micro-v1:0 (cheapest + fastest).
"""

import os
import json
import time
import boto3
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

# ---------------- CONFIG ----------------
DEFAULT_MODEL_ID = os.getenv("BEDROCK_MODEL_ID", "amazon.nova-micro-v1:0")
AWS_REGION = os.getenv("AWS_REGION", "us-east-1")

# Read credentials from .env (NO CLI REQUIRED)
AWS_ACCESS_KEY_ID = os.getenv("AWS_ACCESS_KEY_ID")
AWS_SECRET_ACCESS_KEY = os.getenv("AWS_SECRET_ACCESS_KEY")

MAX_HISTORY_MESSAGES = int(os.getenv("MAX_HISTORY_MESSAGES", "8"))
LOG_FILENAME = os.getenv("CHAT_LOG_FILE", "chat_log.json")

# Bedrock client with explicit credentials so Streamlit works anywhere
bedrock = boto3.client(
    "bedrock-runtime",
    region_name=AWS_REGION,
    aws_access_key_id=AWS_ACCESS_KEY_ID,
    aws_secret_access_key=AWS_SECRET_ACCESS_KEY
)

# ---------------- MODES ----------------
modes = {
    "summarizer": "You are a professional text summarizer. Provide a complete, concise answer within {max_tokens} tokens.",
    "translator": "You are a language translator. Provide a complete translation within {max_tokens} tokens.",
    "story generator": "You are a creative story writer. Write a complete story within {max_tokens} tokens.",
    "marvel predictor": "Predict an interesting Marvel storyline. Full and coherent, within {max_tokens} tokens."
}


# ---------------- BEDROCK HELPERS ----------------
def build_prompt_from_messages(messages, max_tokens):
    history = messages[-MAX_HISTORY_MESSAGES:] if len(messages) > MAX_HISTORY_MESSAGES else messages[:]

    system = ""
    if history and history[0]["role"] == "system":
        try:
            system = history[0]["content"].format(max_tokens=max_tokens)
        except:
            system = history[0]["content"]
        history = history[1:]

    prompt_parts = []
    if system:
        prompt_parts.append(f"System: {system}")

    for msg in history:
        role = msg["role"]
        content = msg["content"]
        if role == "user":
            prompt_parts.append(f"User: {content}")
        else:
            prompt_parts.append(f"Assistant: {content}")

    prompt_parts.append("Assistant:")
    return "\n".join(prompt_parts)


def parse_bedrock_response(raw_body):
    """Return outputText or fallback JSON."""
    try:
        body_text = raw_body.read().decode("utf-8")
    except:
        body_text = str(raw_body)

    try:
        parsed = json.loads(body_text)
    except:
        return body_text.strip(), {}

    # Nova-style response
    if isinstance(parsed, dict) and "outputText" in parsed:
        return parsed["outputText"].strip(), parsed

    return body_text.strip(), parsed


def query_bedrock(messages, temperature=0.7, max_tokens=200, model_id=DEFAULT_MODEL_ID):
    """
    Proper Bedrock Chat API call for Amazon Nova Micro models.
    System role is converted to a user instruction.
    """
    chat_messages = []

    for msg in messages:
        role = msg["role"]
        content = msg["content"]

        # Convert system -> user because Nova does NOT support "system" role
        if role == "system":
            chat_messages.append({
                "role": "user",
                "content": [{"text": f"System instruction: {content}"}]
            })
        elif role == "user":
            chat_messages.append({
                "role": "user",
                "content": [{"text": content}]
            })
        else:  # assistant
            chat_messages.append({
                "role": "assistant",
                "content": [{"text": content}]
            })

    payload = {
        "messages": chat_messages,
        "inferenceConfig": {
            "maxTokens": int(max_tokens),
            "temperature": float(temperature)
        }
    }

    start = time.time()

    try:
        response = bedrock.invoke_model(
            modelId=model_id,
            contentType="application/json",
            accept="application/json",
            body=json.dumps(payload)
        )

        elapsed = time.time() - start
        response_json = json.loads(response["body"].read())

        # Extract nova output
        reply = response_json["output"]["message"]["content"][0]["text"]

        return {
            "response": reply,
            "time": round(elapsed, 2),
            "usage": response_json.get("usage", {})
        }

    except Exception as e:
        return {"error": str(e), "time": 0, "usage": {}}


# ---------------- UTILITIES ----------------
def save_response_local(prompt, assistant_response, filename=LOG_FILENAME):
    entry = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "user": prompt,
        "assistant": assistant_response.get("response", ""),
        "response_time_sec": assistant_response.get("time", 0),
        "token_usage": assistant_response.get("usage", {})
    }

    logs = []
    if os.path.exists(filename):
        try:
            logs = json.load(open(filename, "r", encoding="utf-8"))
        except:
            logs = []

    logs.append(entry)
    json.dump(logs, open(filename, "w", encoding="utf-8"), indent=4, ensure_ascii=False)


# -------------------------------------------------------------
# ---------------- STREAMLIT UI (CHATBOT) ---------------------
# -------------------------------------------------------------
st.set_page_config(page_title="Bedrock Chatbot", page_icon="🤖", layout="centered")
st.title("🤖 Mini AI Chatbot — Powered by AWS Bedrock")

st.markdown("Chat naturally — the model responds using Amazon Nova Micro via AWS Bedrock.")

# Session init
if "messages" not in st.session_state:
    st.session_state.messages = []

if "current_mode" not in st.session_state:
    st.session_state.current_mode = "marvel predictor"
    st.session_state.messages = [
        {"role": "system", "content": modes["marvel predictor"]}
    ]

# Sidebar
st.sidebar.header("⚙️ Settings")
selected_mode = st.sidebar.selectbox("Mode:", list(modes.keys()))
max_tokens = st.sidebar.slider("Max Tokens", 50, 1024, 200, step=50)
temperature = st.sidebar.slider("Temperature", 0.0, 1.5, 0.7, step=0.1)
model_id = st.sidebar.text_input("Bedrock Model ID", DEFAULT_MODEL_ID)

# If mode changed
if selected_mode != st.session_state.current_mode:
    st.session_state.current_mode = selected_mode
    st.session_state.messages = [
        {"role": "system", "content": modes[selected_mode]}
    ]
    st.rerun()

# Render chat history (skip system)
for msg in st.session_state.messages[1:]:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Chat input
if user_input := st.chat_input("Type your message..."):
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            result = query_bedrock(
                st.session_state.messages,
                temperature=temperature,
                max_tokens=max_tokens,
                model_id=model_id
            )

        if "error" in result:
            st.error("❌ " + result["error"])
        else:
            reply = result["response"]
            st.markdown(reply)
            st.session_state.messages.append({"role": "assistant", "content": reply})

            st.caption(f"⏱ {result['time']} sec | Model: {model_id}")

            save_response_local(user_input, result)

# Logs expander
with st.expander("📜 Previous Logs"):
    if os.path.exists(LOG_FILENAME):
        logs = json.load(open(LOG_FILENAME, "r", encoding="utf-8"))
        for log in reversed(logs[-5:]):
            st.write(f"**{log['timestamp']}**")
            st.write(f"🧑 You: {log['user']}")
            st.write(f"🤖 AI: {log['assistant']}")
            st.write("---")
    else:
        st.info("No logs yet.")

#python -m streamlit run bedrockChatBot.py