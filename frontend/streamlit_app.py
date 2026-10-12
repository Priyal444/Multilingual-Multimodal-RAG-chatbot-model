"""Polyglot — Streamlit chat UI.

Run the API first, then:

    streamlit run frontend/streamlit_app.py

The UI is a thin client: every decision (language, retrieval, model settings)
is made server-side, so this file contains presentation logic only.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any

import streamlit as st

# `streamlit run frontend/streamlit_app.py` puts *this* directory on sys.path,
# not the project root, so both import spellings have to work.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
try:
    from frontend.api_client import APIError, PolyglotClient
except ImportError:  # pragma: no cover - depends on how Streamlit was invoked
    from api_client import APIError, PolyglotClient

DEFAULT_API = os.environ.get("POLYGLOT_API_URL", "http://localhost:8000")

LANGUAGE_CHOICES = {
    "Auto-detect": None,
    "English": "en", "हिन्दी (Hindi)": "hi", "Español": "es", "Français": "fr",
    "Deutsch": "de", "中文 (Chinese)": "zh", "日本語 (Japanese)": "ja",
    "العربية (Arabic)": "ar", "Português": "pt", "Русский (Russian)": "ru",
    "한국어 (Korean)": "ko", "Italiano": "it", "বাংলা (Bengali)": "bn",
    "தமிழ் (Tamil)": "ta", "తెలుగు (Telugu)": "te", "मराठी (Marathi)": "mr",
    "ગુજરાતી (Gujarati)": "gu", "ಕನ್ನಡ (Kannada)": "kn", "اردو (Urdu)": "ur",
    "Türkçe": "tr", "Tiếng Việt": "vi", "ไทย (Thai)": "th", "Bahasa Indonesia": "id",
    "Nederlands": "nl", "Polski": "pl", "Українська": "uk", "עברית (Hebrew)": "he",
}

st.set_page_config(page_title="Polyglot", page_icon="🌐", layout="wide")


# ---------------------------------------------------------------------------
# state
# ---------------------------------------------------------------------------


def init_state() -> None:
    defaults: dict[str, Any] = {
        "api_url": DEFAULT_API,
        "access_token": None,
        "user": None,
        "conversation_id": None,
        "messages": [],
        "language": None,
        "use_rag": True,
        "stream": True,
        "last_sources": [],
        "pending_images": [],
        "uploader_key": 0,
    }
    for key, value in defaults.items():
        st.session_state.setdefault(key, value)


def client() -> PolyglotClient:
    return PolyglotClient(st.session_state.api_url, st.session_state.access_token)


def reset_chat() -> None:
    st.session_state.conversation_id = None
    st.session_state.messages = []
    st.session_state.last_sources = []


# ---------------------------------------------------------------------------
# auth screen
# ---------------------------------------------------------------------------


def render_auth() -> None:
    st.title("🌐 Polyglot")
    st.caption(
        "A multimodal, multilingual assistant — text, images, voice, and your own documents."
    )

    st.session_state.api_url = st.text_input("API URL", value=st.session_state.api_url)

    login_tab, register_tab = st.tabs(["Sign in", "Create account"])

    with login_tab:
        with st.form("login"):
            email = st.text_input("Email")
            password = st.text_input("Password", type="password")
            if st.form_submit_button("Sign in", use_container_width=True):
                try:
                    tokens = client().login(email, password)
                    st.session_state.access_token = tokens["access_token"]
                    st.session_state.user = client().me()
                    st.rerun()
                except APIError as exc:
                    st.error(str(exc))

    with register_tab:
        with st.form("register"):
            email = st.text_input("Email", key="r_email")
            password = st.text_input(
                "Password", type="password", key="r_pw", help="At least 8 characters"
            )
            full_name = st.text_input("Name (optional)", key="r_name")
            label = st.selectbox("Preferred reply language", list(LANGUAGE_CHOICES))
            if st.form_submit_button("Create account", use_container_width=True):
                try:
                    tokens = client().register(
                        email, password, full_name, LANGUAGE_CHOICES[label]
                    )
                    st.session_state.access_token = tokens["access_token"]
                    st.session_state.user = client().me()
                    st.rerun()
                except APIError as exc:
                    st.error(str(exc))


# ---------------------------------------------------------------------------
# sidebar
# ---------------------------------------------------------------------------


def render_sidebar() -> None:
    api = client()

    with st.sidebar:
        user = st.session_state.user or {}
        st.markdown(f"**{user.get('full_name') or user.get('email', 'Signed in')}**")
        if st.button("Sign out", use_container_width=True):
            st.session_state.access_token = None
            st.session_state.user = None
            reset_chat()
            st.rerun()

        st.divider()

        # --- turn settings ---
        st.subheader("Reply language")
        current = st.session_state.language
        index = list(LANGUAGE_CHOICES.values()).index(current) if current in LANGUAGE_CHOICES.values() else 0
        label = st.selectbox(
            "Language",
            list(LANGUAGE_CHOICES),
            index=index,
            label_visibility="collapsed",
            help="Auto-detect mirrors whatever language you write in.",
        )
        st.session_state.language = LANGUAGE_CHOICES[label]

        st.session_state.stream = st.toggle("Stream responses", st.session_state.stream)
        st.session_state.use_rag = st.toggle(
            "Use my documents", st.session_state.use_rag, help="Answer from uploaded files."
        )

        st.divider()

        # --- documents ---
        st.subheader("📄 Documents")
        uploaded = st.file_uploader(
            "Upload PDF, DOCX, TXT, MD or CSV",
            type=["pdf", "docx", "txt", "md", "csv"],
            key=f"doc_upload_{st.session_state.uploader_key}",
            label_visibility="collapsed",
        )
        if uploaded is not None:
            with st.spinner(f"Indexing {uploaded.name}…"):
                try:
                    doc = api.upload_document(
                        uploaded.name,
                        uploaded.getvalue(),
                        uploaded.type or "application/octet-stream",
                    )
                    if doc["status"] == "ready":
                        st.success(f"Indexed {doc['chunk_count']} chunks.")
                    else:
                        st.error(doc.get("error") or "Indexing failed.")
                    st.session_state.uploader_key += 1
                    st.rerun()
                except APIError as exc:
                    st.error(str(exc))

        try:
            docs = api.list_documents()["items"]
        except APIError:
            docs = []

        for doc in docs:
            icon = {"ready": "✅", "failed": "❌"}.get(doc["status"], "⏳")
            col_a, col_b = st.columns([5, 1])
            col_a.caption(f"{icon} {doc['filename']} · {doc['chunk_count']} chunks")
            if col_b.button("🗑", key=f"del_doc_{doc['id']}", help="Delete"):
                api.delete_document(doc["id"])
                st.rerun()

        st.divider()

        # --- conversations ---
        st.subheader("💬 Conversations")
        if st.button("＋ New conversation", use_container_width=True):
            reset_chat()
            st.rerun()

        try:
            conversations = api.list_conversations()["items"]
        except APIError:
            conversations = []

        for conversation in conversations:
            selected = conversation["id"] == st.session_state.conversation_id
            col_a, col_b = st.columns([5, 1])
            if col_a.button(
                ("• " if selected else "") + conversation["title"][:34],
                key=f"conv_{conversation['id']}",
                use_container_width=True,
            ):
                load_conversation(conversation["id"])
                st.rerun()
            if col_b.button("🗑", key=f"del_conv_{conversation['id']}", help="Delete"):
                api.delete_conversation(conversation["id"])
                if selected:
                    reset_chat()
                st.rerun()

        st.divider()
        with st.expander("System status"):
            try:
                ready = api.ready()
                caps = ready["checks"]["capabilities"]
                st.write(f"**Model** · `{caps['llm']['model']}`")
                st.write(f"**Database** · {ready['checks']['database']['dialect']}")
                st.write(f"**Cache** · {ready['checks']['cache']['backend']}")
                st.write(f"**Vector store** · {caps['rag']['vector_store']}")
                st.write(
                    "**Retrieval** · "
                    + ("semantic" if caps["rag"]["semantic"] else "lexical fallback")
                )
                st.write("**Speech-to-text** · " + ("on" if caps["speech_to_text"] else "off"))
                st.write("**Text-to-speech** · " + ("on" if caps["text_to_speech"] else "off"))
            except APIError as exc:
                st.error(str(exc))


def load_conversation(conversation_id: str) -> None:
    detail = client().get_conversation(conversation_id)
    st.session_state.conversation_id = conversation_id
    st.session_state.messages = [
        {"role": m["role"], "content": m["content"], "language": m.get("language")}
        for m in detail["messages"]
    ]
    st.session_state.last_sources = []


# ---------------------------------------------------------------------------
# chat
# ---------------------------------------------------------------------------


def render_history() -> None:
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            for image in message.get("images", []):
                st.image(image, width=260)
            if message["role"] == "assistant" and message.get("language"):
                st.caption(f"↳ {message['language']}")


def render_sources() -> None:
    if not st.session_state.last_sources:
        return
    with st.expander(f"📎 {len(st.session_state.last_sources)} source(s) used"):
        for index, source in enumerate(st.session_state.last_sources, start=1):
            st.caption(
                f"[{index}] {source['filename']} · part {source['part']} · "
                f"relevance {source['score']:.2f}"
            )


def send_turn(text: str, images: list[tuple[str, bytes, str]], audio: tuple | None) -> None:
    api = client()
    display = text or ("🎤 voice note" if audio else "🖼 image")

    st.session_state.messages.append(
        {
            "role": "user",
            "content": display,
            "images": [data for _n, data, _t in images],
        }
    )
    with st.chat_message("user"):
        st.markdown(display)
        for _name, data, _type in images:
            st.image(data, width=260)

    with st.chat_message("assistant"):
        try:
            # Binary uploads must go through the multipart endpoint; streaming
            # is only wired for the JSON path.
            if audio or images:
                with st.spinner("Thinking…"):
                    result = api.chat_multipart(
                        text,
                        conversation_id=st.session_state.conversation_id,
                        language=st.session_state.language,
                        image_files=images or None,
                        audio_file=audio,
                        use_rag=st.session_state.use_rag,
                    )
                reply = result["reply"]
                st.markdown(reply)
                st.session_state.conversation_id = result["conversation_id"]
                st.session_state.last_sources = result.get("sources", [])
                language = result["reply_language"]

            elif st.session_state.stream:
                placeholder = st.empty()
                buffer = ""
                language = st.session_state.language or ""
                for event, data in api.chat_stream(
                    text,
                    conversation_id=st.session_state.conversation_id,
                    language=st.session_state.language,
                    use_rag=st.session_state.use_rag,
                ):
                    if event == "start":
                        st.session_state.conversation_id = data["conversation_id"]
                        language = data["reply_language"]
                    elif event == "delta":
                        buffer += data["text"]
                        placeholder.markdown(buffer + " ▌")
                    elif event == "done":
                        st.session_state.last_sources = data.get("sources", [])
                        language = data["reply_language"]
                    elif event == "error":
                        st.error(data["message"])
                        return
                placeholder.markdown(buffer)
                reply = buffer

            else:
                with st.spinner("Thinking…"):
                    result = api.chat(
                        text,
                        conversation_id=st.session_state.conversation_id,
                        language=st.session_state.language,
                        use_rag=st.session_state.use_rag,
                    )
                reply = result["reply"]
                st.markdown(reply)
                st.session_state.conversation_id = result["conversation_id"]
                st.session_state.last_sources = result.get("sources", [])
                language = result["reply_language"]

        except APIError as exc:
            st.error(str(exc))
            return

        st.caption(f"↳ {language}")
        st.session_state.messages.append(
            {"role": "assistant", "content": reply, "language": language}
        )

        if st.button("🔊 Listen", key=f"tts_{len(st.session_state.messages)}"):
            try:
                st.audio(api.speak(reply, language), format="audio/mpeg")
            except APIError as exc:
                st.error(str(exc))


def render_chat() -> None:
    st.title("🌐 Polyglot")
    st.caption("Ask in any language. Attach an image, record a voice note, or query your documents.")

    render_history()
    render_sources()

    with st.expander("📎 Attach an image or record a voice note"):
        col_a, col_b = st.columns(2)
        with col_a:
            images = st.file_uploader(
                "Images",
                type=["png", "jpg", "jpeg", "gif", "webp"],
                accept_multiple_files=True,
                key=f"img_{st.session_state.uploader_key}",
            )
        with col_b:
            audio = None
            if hasattr(st, "audio_input"):
                audio = st.audio_input(
                    "Voice note", key=f"aud_{st.session_state.uploader_key}"
                )
            else:  # older Streamlit
                audio = st.file_uploader(
                    "Voice note",
                    type=["wav", "mp3", "m4a", "ogg", "webm"],
                    key=f"aud_{st.session_state.uploader_key}",
                )

    prompt = st.chat_input("Type your message…")

    if prompt or ((images or audio) and st.button("Send attachment", type="primary")):
        image_payload = [
            (f.name, f.getvalue(), f.type or "image/png") for f in (images or [])
        ]
        audio_payload = (
            (getattr(audio, "name", "voice-note.wav"), audio.getvalue(),
             getattr(audio, "type", None) or "audio/wav")
            if audio
            else None
        )
        send_turn(prompt or "", image_payload, audio_payload)
        st.session_state.uploader_key += 1
        st.rerun()


# ---------------------------------------------------------------------------


def main() -> None:
    init_state()
    if not st.session_state.access_token:
        render_auth()
        return
    render_sidebar()
    render_chat()


if __name__ == "__main__":
    main()
