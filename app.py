import json
import os
import re
from typing import Any, List, Tuple

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, Response
from pydantic import BaseModel, Field

from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI


app = FastAPI(title="After Effects AI Starter", version="1.1.0")

LATEST_SCRIPT = {
    "content": "",
    "filename": "after_effects_script.jsx",
}


def get_api_key() -> str:
    # Keep your current style as requested.
    api_key = os.getenv("GOOGLE_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError(
            "Missing GEMINI_API_KEY. Set it in your environment before running the app."
        )
    return api_key


def get_llm() -> ChatGoogleGenerativeAI:
    return ChatGoogleGenerativeAI(
        
        model="gemini-3.5-flash",
        google_api_key=os.getenv("GOOGLE_API_KEY", "").strip(),
        temperature=0.2,
    )


class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1)
    conversation: List[dict] = Field(default_factory=list)


class ChatResponse(BaseModel):
    reply: str
    after_effects_jsx: str
    script_filename: str = "after_effects_script.jsx"
    script_status: str = ""
    download_url: str = "/api/download/latest.jsx"


def detect_task_mode(message: str) -> str:
    m = message.lower()

    start_over_tokens = (
        "start over",
        "clear scene",
        "wipe scene",
        "new scene",
        "reset scene",
        "delete everything",
        "remove all",
        "fresh scene",
        "from scratch",
        "new project",
    )
    incremental_tokens = (
        "add one more",
        "another",
        "also add",
        "add another",
        "one more",
        "extend",
        "append",
        "beside",
        "next to",
        "duplicate",
        "plus one",
        "add a second",
    )
    modify_tokens = (
        "make it more realistic",
        "more realistic",
        "improve",
        "refine",
        "edit",
        "modify",
        "change",
        "fix",
        "tweak",
        "adjust",
        "upgrade",
    )

    if any(token in m for token in start_over_tokens):
        return "start_over"
    if any(token in m for token in incremental_tokens):
        return "incremental_add"
    if any(token in m for token in modify_tokens):
        return "modify_existing"
    return "new_object"


PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
You are an expert Adobe After Effects scripting assistant.

Your job:
- Convert the user's request into valid Adobe After Effects ExtendScript (JSX).
- Write code that After Effects can run later.
- Preserve the existing composition or project intent unless the user explicitly asks for a fresh start.
- Edit existing layers, names, or properties when the user asks for changes.
- Do not rebuild everything from scratch if only one element needs editing.

Task modes:
- start_over: create a fresh project/comp setup only when explicitly requested.
- incremental_add: add one more layer/object/effect while preserving the existing AE setup.
- modify_existing: improve or change something already described.
- new_object: create a new animation/object in the current comp without wiping anything.

Important behavior rules:
- If task_mode is incremental_add, do not clear the project.
- If task_mode is incremental_add, create a matching element beside the existing setup.
- If task_mode is modify_existing, keep the current names and refine the current animation instead of replacing it.
- If the user asks for "add one more text layer", keep the current layers and add only one new matching layer.
- If the user asks to "make it smoother", keep the same layers and improve keyframes/easing instead of rewriting the whole script.
- Never use destructive operations unless task_mode is start_over.

AE scripting rules:
- Output Adobe ExtendScript (.jsx) code only in the code field.
- Prefer app.beginUndoGroup("...") and app.endUndoGroup() around changes.
- Use app.project carefully.
- If a comp is needed, create it or find it by name.
- If a layer is needed, find it by name before creating a duplicate.
- Keep the script organized with helper functions.
- Use clear, sensible defaults if anything is ambiguous.
- Prefer valid, practical AE code over fancy code.
- Do not include markdown fences in the code field.

Examples of good AE output:
- Create a comp with a centered text layer and animate scale from 0 to 100.
- Create a shape layer rectangle and animate opacity and position.
- Modify an existing text layer's source text and keep the rest of the comp unchanged.
- Add easing using KeyframeEase and set temporal easing on keyframes.
- Use app.beginUndoGroup() and app.endUndoGroup().

Examples of bad output:
- Markdown fences
- Explanations outside JSON
- Python code
- Blender bpy code
- Recreating the entire project for a small change

Output format:
Return only valid JSON with exactly these keys:
{{
  "reply": "short plain-English explanation",
  "after_effects_jsx": "javascript/ExtendScript code only"
}}

Rules:
- No extra keys.
- No markdown fences.
- The JSX must be valid ExtendScript for After Effects.
- If the user request is ambiguous, choose the most sensible assumption and mention it briefly in reply.
            """.strip(),
        ),
        (
            "human",
            "Task mode: {task_mode}\n\nConversation so far:\n{conversation}\n\nUser request:\n{message}",
        ),
    ]
)


REPAIR_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
You rewrite Adobe After Effects ExtendScript (JSX) scripts.

Rules:
- Preserve the current project intent.
- Do not clear or reset the project unless the user explicitly requested a fresh start.
- Do not use destructive operations like removing all items unless task_mode is start_over.
- Focus on incremental edits.
- Keep existing layers, comps, and animation structure in place.
- Make the smallest change needed to satisfy the request.
- Return JSON only with keys "reply" and "after_effects_jsx".
            """.strip(),
        ),
        (
            "human",
            """
The previous script was too destructive, too weak, or invalid for an incremental edit.

Task mode: {task_mode}

User request:
{message}

Conversation:
{conversation}

Previous script:
{bad_script}

Rewrite it so it preserves the existing AE project intent and performs the requested edit correctly.
If the user asked to add one more text layer, add just one matching layer.
If the user asked to smooth an animation, keep the same layers and improve easing/keyframes.
            """.strip(),
        ),
    ]
)


def format_conversation(conversation: List[dict]) -> str:
    if not conversation:
        return "[]"
    return json.dumps(conversation, ensure_ascii=False, indent=2)


def normalize_llm_content(result: Any) -> str:
    content = getattr(result, "content", result)

    if isinstance(content, str):
        return content.strip()

    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict):
                if "text" in item:
                    parts.append(str(item["text"]))
                elif "content" in item:
                    parts.append(str(item["content"]))
                else:
                    parts.append(json.dumps(item, ensure_ascii=False))
            else:
                parts.append(str(item))
        return "\n".join(parts).strip()

    if isinstance(content, dict):
        for key in ("text", "content", "message"):
            if key in content:
                value = content[key]
                if isinstance(value, str):
                    return value.strip()
                return json.dumps(value, ensure_ascii=False)
        return json.dumps(content, ensure_ascii=False)

    return str(content).strip()


def strip_code_fences(text: str) -> str:
    if not text:
        return text
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.replace("```json", "", 1).replace("```javascript", "", 1).replace("```jsx", "", 1).replace("```", "", 1).strip()
    return cleaned


def extract_json_object(text: str) -> str:
    """
    Tries to pull the first JSON object out of a messy model response.
    Handles:
    - markdown fences
    - extra commentary before/after JSON
    - partial wrappers
    """
    cleaned = strip_code_fences(text)

    try:
        json.loads(cleaned)
        return cleaned
    except Exception:
        pass

    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start != -1 and end != -1 and end > start:
        candidate = cleaned[start:end + 1].strip()
        try:
            json.loads(candidate)
            return candidate
        except Exception:
            pass

    return cleaned


def parse_model_json(raw: str) -> Tuple[str, str]:
    """
    Expect JSON:
    {
      "reply": "...",
      "after_effects_jsx": "..."
    }
    """
    raw = extract_json_object(raw)

    try:
        data = json.loads(raw)
        if isinstance(data, dict):
            reply = str(data.get("reply", "Here is an After Effects script draft."))
            jsx = str(data.get("after_effects_jsx", raw))
            return reply, jsx
    except Exception:
        pass

    return "Here is an After Effects script draft.", raw


def looks_destructive(script: str) -> bool:
    low = script.lower()
    destructive_patterns = (
        "app.project.items.remove",
        "app.project.close(",
        "removeall",
        ".remove()",
        "delete",
        "purge",
        "clear()",
        "reset",
        "readfromfile",
        "removeallkeyframes",
    )
    return any(pattern in low for pattern in destructive_patterns)


def looks_like_valid_ae_script(script: str) -> bool:
    low = script.lower()

    required_any = (
        "app.project.items.addcomp",
        "app.project.items.add",
        "app.beginundogroup(",
        "app.endundogroup(",
    )

    has_required_any = any(token in low for token in required_any)

    has_structure = (
        "app.beginundogroup(" in low
        and "app.endundogroup(" in low
        and ("addcomp" in low or "addtext" in low or "addsolid" in low or "addshape" in low)
    )

    return has_required_any and has_structure


def generate_script(message: str, conversation: str, task_mode: str) -> tuple[str, str]:
    llm = get_llm()
    chain = PROMPT | llm
    result = chain.invoke(
        {
            "message": message,
            "conversation": conversation,
            "task_mode": task_mode,
        }
    )

    raw = normalize_llm_content(result)
    return parse_model_json(raw)


def repair_script_if_needed(
    message: str,
    conversation: str,
    task_mode: str,
    bad_script: str,
) -> tuple[str, str]:
    llm = get_llm()
    repair_chain = REPAIR_PROMPT | llm
    result = repair_chain.invoke(
        {
            "message": message,
            "conversation": conversation,
            "task_mode": task_mode,
            "bad_script": bad_script,
        }
    )
    raw = normalize_llm_content(result)
    return parse_model_json(raw)


@app.get("/", response_class=HTMLResponse)
async def home() -> str:
    return HTML_PAGE


@app.get("/api/download/latest.jsx")
async def download_latest_jsx() -> Response:
    content = LATEST_SCRIPT.get("content", "")
    if not content.strip():
        raise HTTPException(status_code=404, detail="No script has been generated yet.")

    filename = LATEST_SCRIPT.get("filename", "after_effects_script.jsx")
    return Response(
        content=content,
        media_type="application/javascript",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


@app.post("/api/chat", response_model=ChatResponse)
async def chat(payload: ChatRequest) -> ChatResponse:
    try:
        conversation_text = format_conversation(payload.conversation)
        task_mode = detect_task_mode(payload.message)

        reply, after_effects_jsx = generate_script(
            message=payload.message,
            conversation=conversation_text,
            task_mode=task_mode,
        )

        # Repair if the script is destructive or too weak / invalid.
        needs_repair = False

        if task_mode != "start_over" and looks_destructive(after_effects_jsx):
            needs_repair = True

        if not looks_like_valid_ae_script(after_effects_jsx):
            needs_repair = True

        if needs_repair:
            reply, after_effects_jsx = repair_script_if_needed(
                message=payload.message,
                conversation=conversation_text,
                task_mode=task_mode,
                bad_script=after_effects_jsx,
            )

        # If still bad, keep the latest output anyway but show a warning status.
        validation_ok = looks_like_valid_ae_script(after_effects_jsx)
        destructive = looks_destructive(after_effects_jsx)

        LATEST_SCRIPT["content"] = after_effects_jsx
        LATEST_SCRIPT["filename"] = "after_effects_script.jsx"

        if validation_ok and not destructive:
            script_status = "Generated JSX script and saved it for download."
        elif destructive:
            script_status = "Generated script, but it still looks destructive. Review before using."
        else:
            script_status = "Generated script, but validation is weak. Review before using."

        return ChatResponse(
            reply=reply,
            after_effects_jsx=after_effects_jsx,
            script_filename=LATEST_SCRIPT["filename"],
            script_status=script_status,
            download_url="/api/download/latest.jsx",
        )

    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


HTML_PAGE = r"""
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>AE AI Starter</title>
  <style>
    :root {
      --bg: #0f172a;
      --panel: #111827;
      --panel-2: #1f2937;
      --text: #e5e7eb;
      --muted: #94a3b8;
      --accent: #38bdf8;
      --border: #334155;
      --success: #86efac;
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      font-family: Inter, system-ui, -apple-system, Segoe UI, Roboto, Arial, sans-serif;
      background: linear-gradient(180deg, #020617, #0f172a 50%, #111827);
      color: var(--text);
      min-height: 100vh;
    }
    .wrap {
      max-width: 1100px;
      margin: 0 auto;
      padding: 24px;
      display: grid;
      grid-template-columns: 1.1fr 0.9fr;
      gap: 20px;
    }
    .card {
      background: rgba(15, 23, 42, 0.8);
      border: 1px solid var(--border);
      border-radius: 18px;
      box-shadow: 0 20px 50px rgba(0,0,0,0.25);
      overflow: hidden;
      backdrop-filter: blur(8px);
    }
    .header {
      padding: 18px 20px;
      border-bottom: 1px solid var(--border);
    }
    .header h1 {
      margin: 0 0 6px;
      font-size: 22px;
    }
    .header p {
      margin: 0;
      color: var(--muted);
      font-size: 14px;
    }
    .chat {
      height: 65vh;
      display: flex;
      flex-direction: column;
    }
    .messages {
      flex: 1;
      padding: 18px;
      overflow-y: auto;
    }
    .msg {
      max-width: 88%;
      padding: 12px 14px;
      border-radius: 16px;
      margin: 0 0 12px;
      line-height: 1.45;
      white-space: pre-wrap;
      word-wrap: break-word;
    }
    .user {
      background: #0ea5e9;
      color: white;
      margin-left: auto;
      border-bottom-right-radius: 6px;
    }
    .bot {
      background: var(--panel-2);
      border: 1px solid var(--border);
      border-bottom-left-radius: 6px;
    }
    .composer {
      display: flex;
      gap: 10px;
      padding: 16px;
      border-top: 1px solid var(--border);
      background: rgba(2, 6, 23, 0.35);
    }
    textarea {
      flex: 1;
      resize: none;
      min-height: 54px;
      max-height: 150px;
      padding: 14px 16px;
      border-radius: 14px;
      border: 1px solid var(--border);
      background: #0b1220;
      color: var(--text);
      outline: none;
      font: inherit;
    }
    button, .linkbtn {
      border: 0;
      border-radius: 14px;
      padding: 0 18px;
      background: var(--accent);
      color: #082f49;
      font-weight: 700;
      cursor: pointer;
      display: inline-flex;
      align-items: center;
      justify-content: center;
      text-decoration: none;
      min-height: 40px;
    }
    button:disabled { opacity: 0.65; cursor: not-allowed; }
    .side {
      padding: 18px;
    }
    .side h2 {
      margin: 0 0 10px;
      font-size: 18px;
    }
    .side p, .side li {
      color: var(--muted);
      font-size: 14px;
      line-height: 1.6;
    }
    pre {
      margin: 12px 0 0;
      padding: 16px;
      background: #020617;
      border: 1px solid var(--border);
      border-radius: 14px;
      overflow: auto;
      font-size: 13px;
      line-height: 1.5;
      color: #cbd5e1;
      white-space: pre-wrap;
      word-wrap: break-word;
      min-height: 210px;
    }
    .status {
      margin-top: 10px;
      font-size: 13px;
      color: var(--success);
    }
    .small {
      font-size: 12px;
      color: var(--muted);
    }
    .actions {
      display: flex;
      gap: 10px;
      margin-top: 12px;
      flex-wrap: wrap;
    }
    @media (max-width: 900px) {
      .wrap { grid-template-columns: 1fr; }
      .chat { height: 60vh; }
    }
  </style>
</head>
<body>
  <div class="wrap">
    <div class="card chat">
      <div class="header">
        <h1>AE AI Starter</h1>
        <p>Generate ExtendScript (.jsx) files for AE and download them later</p>
      </div>
      <div class="messages" id="messages">
        <div class="msg bot">“Let's build AE scripts”</div>
      </div>
      <div class="composer">
        <textarea id="input" placeholder="Describe the animation or layer change..."></textarea>
        <button id="send">Send</button>
      </div>
    </div>

    <div class="card side">
      <h2>Generated JSX</h2>
      <p class="small">
        It only generates a valid .jsx script and saves it for download.
      </p>

      <div class="actions">
        <a id="downloadBtn" class="linkbtn" href="/api/download/latest.jsx" download>Download latest JSX</a>
      </div>

      <h2>Latest script</h2>
      <pre id="script">(nothing yet)</pre>

      <h2>Reply</h2>
      <pre id="reply">(nothing yet)</pre>

      <div class="status" id="status">Status: waiting</div>
    </div>
  </div>

  <script>
    const messages = document.getElementById("messages");
    const input = document.getElementById("input");
    const send = document.getElementById("send");
    const scriptBox = document.getElementById("script");
    const replyBox = document.getElementById("reply");
    const statusBox = document.getElementById("status");
    const downloadBtn = document.getElementById("downloadBtn");
    const conversation = [];

    function addMessage(text, cls) {
      const div = document.createElement("div");
      div.className = `msg ${cls}`;
      div.textContent = text;
      messages.appendChild(div);
      messages.scrollTop = messages.scrollHeight;
    }

    async function doSend() {
      const text = input.value.trim();
      if (!text) return;

      addMessage(text, "user");
      conversation.push({ role: "user", content: text });
      input.value = "";
      send.disabled = true;

      try {
        const res = await fetch("/api/chat", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ message: text, conversation })
        });

        const data = await res.json();
        if (!res.ok) {
          throw new Error(data.detail || "Request failed");
        }

        conversation.push({ role: "assistant", content: data.reply });
        addMessage(data.reply, "bot");
        replyBox.textContent = data.reply || "(empty)";
        scriptBox.textContent = data.after_effects_jsx || "(empty)";
        statusBox.textContent = "Status: " + (data.script_status || "JSX generated");
        if (data.download_url) {
          downloadBtn.href = data.download_url;
          downloadBtn.setAttribute("download", data.script_filename || "after_effects_script.jsx");
        }
      } catch (err) {
        addMessage("Error: " + err.message, "bot");
        statusBox.textContent = "Status: error";
      } finally {
        send.disabled = false;
        input.focus();
      }
    }

    send.addEventListener("click", doSend);
    input.addEventListener("keydown", (e) => {
      if (e.key === "Enter" && !e.shiftKey) {
        e.preventDefault();
        doSend();
      }
    });
  </script>
</body>
</html>
"""


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
