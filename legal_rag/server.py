"""Twilio ConversationRelay adapter.

Two endpoints:
  * POST /voice           -> returns TwiML that connects the call to the WS
  * WS   /ws/voice        -> receives 'prompt' events, runs the RAG pipeline,
                             streams the verified answer back as 'text' tokens

Design note: for a legal-citation agent we verify BEFORE speaking, so we do not
stream raw LLM tokens straight through. We generate + verify the full answer,
then stream it to Twilio sentence-by-sentence — Twilio still starts speaking the
first sentence while later ones are in flight, which keeps latency reasonable
without ever letting an unverified citation reach the caller.

Requires: fastapi, uvicorn, twilio. Run:
    ANTHROPIC_API_KEY=... uvicorn legal_rag.server:app --host 0.0.0.0 --port 8080
Expose over TLS (wss://) — ConversationRelay requires it.
"""
from __future__ import annotations

import json
import os
import re

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Request, Response

from .pipeline import LegalRAG
from .llm import AnthropicLLM


PUBLIC_WSS_URL = os.environ.get("PUBLIC_WSS_URL", "wss://your-host/ws/voice")
INDEX_DIR = os.environ.get("LEGAL_RAG_INDEX", "index")

app = FastAPI()

# Build the pipeline once at startup (index load + model warm).
_rag: LegalRAG | None = None


def get_rag() -> LegalRAG:
    global _rag
    if _rag is None:
        _rag = LegalRAG.from_index(
            INDEX_DIR,
            AnthropicLLM(model=os.environ.get("LEGAL_RAG_MODEL", "claude-sonnet-5")),
            k=5,
            source_filter=os.environ.get("LEGAL_RAG_SOURCE") or None,
        )
    return _rag


@app.post("/voice")
async def voice(_: Request) -> Response:
    from twilio.twiml.voice_response import VoiceResponse, Connect
    vr = VoiceResponse()
    connect = Connect()
    connect.conversation_relay(
        url=PUBLIC_WSS_URL,
        welcome_greeting="Legal information line. What's your question?",
        voice="en-US-Neural2-F",
        language="en-US",
        transcription_provider="deepgram",
        speech_model="nova-2-phonecall",
        interrupt_by_dtmf=True,
    )
    vr.append(connect)
    return Response(content=str(vr), media_type="text/xml")


_SENTENCE = re.compile(r".+?(?:[.!?](?:\s|$)|$)", re.DOTALL)


def _sentences(text: str) -> list[str]:
    return [s.strip() for s in _SENTENCE.findall(text) if s.strip()]


@app.websocket("/ws/voice")
async def ws_voice(ws: WebSocket) -> None:
    await ws.accept()
    rag = get_rag()
    try:
        while True:
            event = json.loads(await ws.receive_text())
            etype = event.get("type")

            if etype == "prompt":
                query = event.get("voicePrompt", "")
                answer = rag.answer(query)  # retrieve + generate + verify

                # Stream verified answer to TTS sentence-by-sentence.
                sents = _sentences(answer.spoken_text)
                for i, sent in enumerate(sents):
                    await ws.send_text(json.dumps({
                        "type": "text",
                        "token": sent + " ",
                        "last": i == len(sents) - 1,
                    }))

            elif etype == "interrupt":
                # Caller talked over the agent; nothing buffered server-side to
                # cancel here since we send whole sentences. Just continue.
                continue

            elif etype == "error":
                # Log and keep the socket alive.
                print(f"[cr] error: {event.get('description')}")

            # 'connected' / 'dtmf' events fall through.
    except WebSocketDisconnect:
        pass
