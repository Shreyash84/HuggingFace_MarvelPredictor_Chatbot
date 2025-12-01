"""
Refactored and documented Streamlit chatbot using AWS Bedrock (boto3 runtime).
Changes made:
- Inlined detailed line-by-line comments for main helpers and functions.
- Unified response parsing via `parse_bedrock_response` and used it in `query_bedrock`.
- Removed unused prompt-builder variant; kept `build_prompt_from_messages` as an optional helper and documented its usage.
- Added robust error logging and an exponential backoff retry mechanism for `invoke_model`.
- Improved file I/O safety (use `with open(...)`) and defensive JSON handling.
- Kept Streamlit UI logic intact; added safer defaults and clearer logging for errors.

How to run:
python -m streamlit run advanceChatBot_bedrock_refactor.py

Note: keep your .env with AWS credentials and region as before.
"""

import os
import json
import time
import logging
import random
import boto3
import streamlit as st
from dotenv import load_dotenv
from typing import List, Dict, Tuple, Any, Optional

# Load environment variables from .env
load_dotenv()

# ---------------- CONFIG ----------------
DEFAULT_MODEL_ID = os.getenv("BEDROCK_MODEL_ID", "amazon.nova-micro-v1:0")
AWS_REGION = os.getenv("AWS_REGION", "us-east-1")

# Read credentials from .env (NO CLI REQUIRED) - keep this pattern if you want Streamlit to run anywhere
AWS_ACCESS_KEY_ID = os.getenv("AWS_ACCESS_KEY_ID")
AWS_SECRET_ACCESS_KEY = os.getenv("AWS_SECRET_ACCESS_KEY")

MAX_HISTORY_MESSAGES = int(os.getenv("MAX_HISTORY_MESSAGES", "8"))
LOG_FILENAME = os.getenv("CHAT_LOG_FILE", "chat_log.json")
OPERATION_LOG = os.getenv("OPERATION_LOG", "operation.log")

# Configure a basic logger to file and console
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(OPERATION_LOG),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("bedrock_chatbot")

# Bedrock client with explicit credentials so Streamlit works anywhere
# Using boto3 client for bedrock-runtime
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

# ---------------- HELPERS (documented line-by-line) ----------------

def build_prompt_from_messages(messages: List[Dict[str, str]], max_tokens: int) -> str:
    """
    Convert a list of chat messages into a single plain-text prompt.

    Line-by-line explanation:
    - `messages` is expected to be a list of dicts with keys: 'role' and 'content'.
    - `max_tokens` is provided so system messages may include a placeholder like '{max_tokens}'.

    Returns a newline-joined string with labels (System:, User:, Assistant:) and a trailing
    'Assistant:' marker that indicates the model should now produce the assistant reply.
    """

    # Limit the conversation history to the last MAX_HISTORY_MESSAGES entries.
    # This protects against very large prompts that may hit model limits.
    history = messages[-MAX_HISTORY_MESSAGES:] if len(messages) > MAX_HISTORY_MESSAGES else messages[:]

    # Start with empty system instruction string (we may extract one if present)
    system = ""

    # If the first message in the history is a system instruction, pull it out and
    # allow it to be formatted with max_tokens (if the string contains '{max_tokens}').
    if history and history[0]["role"] == "system":
        try:
            # Attempt to format with max_tokens if the system string uses the placeholder
            system = history[0]["content"].format(max_tokens=max_tokens)
        except Exception:
            # If formatting fails for any reason, fallback to the raw content
            system = history[0]["content"]
        # Remove the system message from history to avoid duplicating it below
        history = history[1:]

    # Prepare list to collect the lines of the prompt
    prompt_parts: List[str] = []

    # If we extracted a system instruction, add it first with a clear label
    if system:
        prompt_parts.append(f"System: {system}")

    # Convert each remaining message to a labeled line. Use 'User' for user messages
    # and 'Assistant' for assistant messages. This helps models that prefer a single
    # text prompt instead of structured chat message arrays.
    for msg in history:
        role = msg.get("role")
        content = msg.get("content", "")
        if role == "user":
            prompt_parts.append(f"User: {content}")
        else:
            # Any non-user message is treated as assistant for the prompt formatting
            prompt_parts.append(f"Assistant: {content}")

    # Finally, append an 'Assistant:' marker so the model knows to produce the assistant reply next.
    prompt_parts.append("Assistant:")

    # Join everything into one string and return it
    return "\n".join(prompt_parts)


def parse_bedrock_response(raw_body: Any) -> Tuple[str, Dict[str, Any]]:
    """
    Robustly parse a Bedrock response body and extract a human-readable text reply.

    Returns a tuple `(text, parsed_json)` where `text` is the best-effort assistant reply
    (trimmed string) and `parsed_json` is the full parsed JSON object if JSON parse succeeded,
    otherwise an empty dict.

    Line-by-line:
    - Try to treat raw_body as a stream (file-like) and read/decode it to text.
    - If that fails, fallback to casting to `str()`.
    - Try to load JSON; if JSON parsing fails, return the raw text.
    - If parsed JSON contains known shapes (like nova's `outputText` or the newer
      `output.message.content[0].text`), prefer those for `text`.
    """

    # Try to read like a streamed body (boto3 responses often provide a file-like body)
    try:
        body_text = raw_body.read().decode("utf-8")
    except Exception:
        # If it's already a string or doesn't support read(), cast to string
        body_text = str(raw_body)

    # Attempt JSON parse; if it fails, return the trimmed raw text and an empty dict
    try:
        parsed = json.loads(body_text)
    except json.JSONDecodeError:
        return body_text.strip(), {}

    # If the parsed JSON contains an older Nova-style `outputText`, return it
    if isinstance(parsed, dict) and "outputText" in parsed:
        return parsed["outputText"].strip(), parsed

    # Newer Bedrock/Nova responses often use: output -> message -> content -> [0] -> text
    try:
        # Defensive navigation through the expected structure
        possible = parsed.get("output", {}).get("message", {}).get("content", [])
        if isinstance(possible, list) and len(possible) > 0:
            first = possible[0]
            if isinstance(first, dict) and "text" in first:
                return str(first["text"]).strip(), parsed
    except Exception:
        # If anything goes wrong while inspecting structure, ignore and return full parsed JSON below
        pass

    # Fallback: return the original body text and the parsed JSON
    return body_text.strip(), parsed


def query_bedrock(
    messages: List[Dict[str, str]],
    temperature: float = 0.7,
    max_tokens: int = 200,
    model_id: str = DEFAULT_MODEL_ID,
    max_retries: int = 3,
    base_backoff: float = 0.5
) -> Dict[str, Any]:
    """
    Call AWS Bedrock's `invoke_model` with a structured chat payload.

    Features and protections added:
    - Converts `system` role into a `user` message with a "System instruction:" prefix
      because some Nova models do not support an explicit system role.
    - Uses a retry loop with exponential backoff + jitter for robustness against transient errors.
    - Uses `parse_bedrock_response` to extract the assistant reply regardless of response shape.
    - Returns a dict that always contains either `response` on success or `error` on failure.

    Returns shape example on success:
    {"response": <str>, "time": <float seconds>, "usage": <dict>}

    On error:
    {"error": <str>, "time": <float>, "usage": {}}
    """

    # Build the chat messages payload in the Bedrock Nova style
    chat_messages: List[Dict[str, Any]] = []
    for msg in messages:
        role = msg.get("role")
        content = msg.get("content", "")

        # Convert any system role into a user message that the model will still observe
        if role == "system":
            chat_messages.append({
                "role": "user",
                "content": [{"text": f"System instruction: {content}"}]
            })
        elif role == "user":
            chat_messages.append({"role": "user", "content": [{"text": content}]})
        else:  # assistant
            chat_messages.append({"role": "assistant", "content": [{"text": content}]})

    # Build the full payload with inferenceConfig
    payload = {
        "messages": chat_messages,
        "inferenceConfig": {"maxTokens": int(max_tokens), "temperature": float(temperature)}
    }

    # We'll attempt up to `max_retries` attempts with exponential backoff
    attempt = 0
    start_time = time.time()
    last_exception: Optional[Exception] = None

    while attempt <= max_retries:
        try:
            attempt += 1
            # Use boto3 invoke_model to call Bedrock
            response = bedrock.invoke_model(
                modelId=model_id,
                contentType="application/json",
                accept="application/json",
                body=json.dumps(payload)
            )

            # Parse the response body using our helper for robustness
            raw_body = response.get("body")
            text, parsed_json = parse_bedrock_response(raw_body)

            elapsed = time.time() - start_time

            # If the parsed JSON has usage info, include it, otherwise empty dict
            usage = parsed_json.get("usage", {}) if isinstance(parsed_json, dict) else {}

            # Log successful invocation
            logger.info(f"invoke_model OK (model={model_id}) in {elapsed:.2f}s")

            return {"response": text, "time": round(elapsed, 2), "usage": usage}

        except Exception as e:
            # Record last exception and decide whether to retry
            last_exception = e
            # If we've exhausted retries, break and return the error
            if attempt > max_retries:
                logger.exception("invoke_model failed after retries")
                break

            # Compute exponential backoff with jitter
            backoff = base_backoff * (2 ** (attempt - 1))
            # Add some jitter so concurrent clients don't synchronize retries
            jitter = random.uniform(0, backoff * 0.1)
            sleep_time = backoff + jitter

            logger.warning(f"invoke_model attempt {attempt} failed: {e}. Retrying in {sleep_time:.2f}s...")
            time.sleep(sleep_time)

    # If we reach here, all retries failed
    return {"error": str(last_exception), "time": round(time.time() - start_time, 2), "usage": {}}


# ---------------- UTILITIES ----------------

def save_response_local(prompt: str, assistant_response: Dict[str, Any], filename: str = LOG_FILENAME) -> None:
    """
    Persist a single chat exchange to `filename` as JSON. This function is defensive and
    uses `with open(...)` to avoid file handle leaks; it also tolerates malformed existing files.

    Fields saved:
    - timestamp: current local time
    - user: original prompt string
    - assistant: assistant text (if available)
    - response_time_sec: seconds used according to the response dict
    - token_usage: usage object if available
    """

    entry = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "user": prompt,
        "assistant": assistant_response.get("response", ""),
        "response_time_sec": assistant_response.get("time", 0),
        "token_usage": assistant_response.get("usage", {})
    }

    logs: List[Dict[str, Any]] = []

    # Read existing logs safely
    if os.path.exists(filename):
        try:
            with open(filename, "r", encoding="utf-8") as f:
                logs = json.load(f)
                if not isinstance(logs, list):
                    # If file contents are not a list, reset to an empty list
                    logs = []
        except Exception:
            # If parsing fails, do not crash; continue with empty logs
            logger.exception("Failed to read existing log file; starting fresh")
            logs = []

    # Append and write back safely
    logs.append(entry)
    try:
        with open(filename, "w", encoding="utf-8") as f:
            json.dump(logs, f, indent=4, ensure_ascii=False)
    except Exception:
        logger.exception("Failed to write log file")


# -------------------------------------------------------------
# ---------------- STREAMLIT UI (CHATBOT) ---------------------
# -------------------------------------------------------------
st.set_page_config(page_title="Bedrock Chatbot", page_icon="🤖", layout="centered")
st.title("🤖 Mini AI Chatbot — Powered by AWS Bedrock (Refactor)")

st.markdown("Chat naturally — the model responds using Amazon Nova Micro via AWS Bedrock.")

# Session init
if "messages" not in st.session_state:
    st.session_state.messages = []

if "current_mode" not in st.session_state:
    st.session_state.current_mode = "marvel predictor"
    st.session_state.messages = [{"role": "system", "content": modes["marvel predictor"].format(max_tokens=200)}]

# Sidebar
st.sidebar.header("⚙️ Settings")
selected_mode = st.sidebar.selectbox("Mode:", list(modes.keys()))
max_tokens = st.sidebar.slider("Max Tokens", 50, 1024, 200, step=50)
temperature = st.sidebar.slider("Temperature", 0.0, 1.5, 0.7, step=0.1)
model_id = st.sidebar.text_input("Bedrock Model ID", DEFAULT_MODEL_ID)

# If mode changed
if selected_mode != st.session_state.current_mode:
    st.session_state.current_mode = selected_mode
    st.session_state.messages = [{"role": "system", "content": modes[selected_mode].format(max_tokens=max_tokens)}]

    # Try to trigger a rerun. Some Streamlit versions expose `experimental_rerun`,
    # others expose `rerun()` and some may have neither. Try them in order and
    # fall back to stopping the run so the updated session state persists.
    try:
        try:
            st.experimental_rerun()
        except Exception:
            try:
                st.rerun()
            except Exception:
                st.stop()
    except Exception:
        # As a final fallback, ensure the script stops gracefully.
        st.stop()
    except AttributeError:
    # Fallback for Streamlit versions that do not expose experimental_rerun
    # Stop this run — session state was already updated, and Streamlit will re-run on the next user interaction.
        st.stop()

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
            logger.error(f"Query error: {result['error']}")
        else:
            reply = result["response"]
            st.markdown(reply)
            st.session_state.messages.append({"role": "assistant", "content": reply})

            st.caption(f"⏱ {result['time']} sec | Model: {model_id}")

            # Save to local logs for later inspection
            save_response_local(user_input, result)

# Logs expander
with st.expander("📜 Previous Logs"):
    if os.path.exists(LOG_FILENAME):
        try:
            with open(LOG_FILENAME, "r", encoding="utf-8") as f:
                logs = json.load(f)
        except Exception:
            logs = []
        for log in reversed(logs[-5:]):
            st.write(f"**{log.get('timestamp')}**")
            st.write(f"🧑 You: {log.get('user')}")
            st.write(f"🤖 AI: {log.get('assistant')}")
            st.write("---")
    else:
        st.info("No logs yet.")

# End of file
