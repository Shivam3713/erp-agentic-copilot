# =============================================================================
# main.py - A minimal FastAPI backend that streams AI chat responses
# =============================================================================
# WHAT THIS FILE DOES (big picture):
#   1. Starts a small web server (using FastAPI)
#   2. Exposes one endpoint: POST /chat
#   3. When someone sends a message to /chat, it forwards that message to
#      Google's Gemini AI model and streams the AI's response back,
#      word-by-word (or chunk-by-chunk), instead of waiting for the whole
#      answer to be ready first.
#
# WHY STREAMING?
#   Normally, an API waits for the ENTIRE response before sending anything
#   back. That means a slow model with a long answer = a long, blank wait
#   for the user. Streaming sends each piece of the response as soon as
#   it's generated, so the user sees text appearing progressively (like
#   ChatGPT/Gemini's typing effect in a browser).
# =============================================================================


# ----- IMPORTS ---------------------------------------------------------------
# Imports bring in code that already exists (built by others) so we don't
# have to write everything from scratch.

import os
# 'os' is Python's built-in module for interacting with the operating system.
# We use it here to read environment variables (like our secret API key)
# instead of hardcoding them directly into the code (which is unsafe/insecure,
# especially if you ever push this code to GitHub).

from fastapi import FastAPI
# FastAPI is the web framework. It's the toolkit that lets us define
# "routes" (URLs like /chat) and handles incoming HTTP requests for us.

from fastapi.responses import StreamingResponse
# StreamingResponse is a special FastAPI response type that lets us send
# data back to the client in small pieces over time, instead of all at once.

from pydantic import BaseModel
# Pydantic is a data-validation library. FastAPI uses it to automatically
# check that incoming request data has the right shape/types (e.g. making
# sure the client actually sent a "message" field, and that it's a string).

from google import genai
# This is Google's official SDK (software development kit) for talking to
# their Gemini AI models. It handles all the low-level HTTP requests to
# Google's servers for us.

from dotenv import load_dotenv
# python-dotenv lets us store secret values (like API keys) in a separate
# ".env" file instead of writing them directly in our code. This keeps
# secrets out of source control (e.g. Git/GitHub).

load_dotenv()
# This line actually reads the ".env" file in your project folder and loads
# its contents into the environment variables that os.environ can see.
# IMPORTANT: this must run BEFORE you try to read any variables from .env
# (like we do below with GEMINI_API_KEY), otherwise they won't exist yet.


# ----- APP SETUP --------------------------------------------------------------

app = FastAPI(title="Copilot Project - Day 1")
# This creates the actual web application object. Think of 'app' as the
# central hub — every route (URL endpoint) you define gets attached to it.
# 'title' is just metadata shown in FastAPI's auto-generated docs page
# (visit http://127.0.0.1:8000/docs while your server is running to see it).

client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
# This creates a "client" object — your connection/credentials to Google's
# Gemini API. os.environ["GEMINI_API_KEY"] looks up the API key you stored
# in your .env file (e.g. GEMINI_API_KEY=AIzaSy...).
#
# NOTE: os.environ["KEY"] will throw a hard error (KeyError) and CRASH the
# app immediately at startup if the variable is missing. That's actually
# useful here — you want to know immediately if your key isn't configured,
# rather than have it fail confusingly later mid-request.


# ----- REQUEST SCHEMA ----------------------------------------------------------

class ChatRequest(BaseModel):
    # This defines what shape of data we EXPECT the client to send us in
    # the body of their POST request. By inheriting from BaseModel, Pydantic
    # (via FastAPI) will automatically:
    #   - Reject requests that don't have a "message" field
    #   - Reject requests where "message" isn't a string
    #   - Return a clear, automatic error message to the client if invalid
    message: str
    # This means: "the JSON body must contain a key called 'message' whose
    # value is a string." Example valid request body:
    #   { "message": "Hello!" }


# ----- ROUTE / ENDPOINT ----------------------------------------------------------

@app.post("/chat")
# This "decorator" tells FastAPI: "when an HTTP POST request comes in to
# the URL path /chat, run the function defined right below this line."
async def chat_endpoint(request: ChatRequest):
    # 'async def' means this function can run asynchronously — it can pause
    # while waiting on slow operations (like a network call to Google's
    # servers) WITHOUT blocking the entire server from handling other
    # requests at the same time. This is essential for a server that
    # might be handling many users' chats simultaneously.
    #
    # 'request: ChatRequest' means FastAPI will automatically:
    #   1. Parse the incoming JSON body
    #   2. Validate it matches the ChatRequest shape (i.e. has "message")
    #   3. Give us a ready-to-use Python object: request.message

    async def stream_generator():
        # This is a "generator function" defined INSIDE chat_endpoint.
        # Instead of returning one value and finishing, a generator can
        # "yield" (send out) multiple values over time, pausing in between.
        # This is exactly the mechanism StreamingResponse needs to send
        # data to the client piece by piece.

        try:
            # We wrap the risky/external part (the API call to Gemini) in a
            # try/except block. "Risky" here means: anything that depends on
            # a network connection, an external server, or credentials could
            # fail unexpectedly (bad API key, model unavailable, network
            # drop, rate limits, etc). try/except lets us catch that failure
            # gracefully instead of crashing the whole server ungracefully.

            response_stream = await client.aio.models.generate_content_stream(
                # 'await' pauses this function here until Gemini's servers
                # respond, but WITHOUT blocking other requests being handled
                # by the server in the meantime (that's the benefit of async).
                model='gemini-3.6-flash',
                # This tells Google WHICH AI model to use. Model names/
                # versions change over time as Google releases new ones and
                # deprecates old ones — check https://ai.google.dev/gemini-api/docs/models
                # if you start getting 404 "model not found" errors again later.
                contents=request.message,
                # This is the actual text we're sending to the AI — the
                # user's message extracted from the validated request body.
            )

            async for chunk in response_stream:
                # 'async for' loops over the streamed response as pieces
                # (chunks) arrive from Gemini, one at a time, as they're
                # generated by the model (instead of waiting for the full
                # answer to be ready).

                if chunk.text:
                    # Not every chunk necessarily contains text (some might
                    # be empty, metadata-only, etc), so we check first to
                    # avoid sending empty/junk data to the client.

                    yield f"data:{chunk.text}\n\n"
                    # This sends one small piece of text back to whoever
                    # called our /chat endpoint, immediately, without
                    # waiting for the rest of the response.
                    #
                    # The format "data:<content>\n\n" is the Server-Sent
                    # Events (SSE) standard — a simple text-based protocol
                    # browsers and HTTP clients know how to read as a
                    # continuous stream of small messages.

            yield "data: [DONE]\n\n"
            # After the loop finishes (meaning Gemini has finished
            # generating its full response), we send one final special
            # message: [DONE]. This is a common convention so the
            # client (e.g. a frontend chat UI) knows "the stream is over,
            # stop listening for more chunks."

        except Exception as e:
            # This 'except' block catches ANY error that happens inside the
            # 'try' block above — e.g. invalid API key, model not found,
            # network failure, Google's servers being down, rate limiting,
            # etc.
            #
            # WHY THIS MATTERS: without this, if Gemini's API call fails
            # partway through, the connection just abruptly dies (which is
            # the "curl: (18) transfer closed with outstanding read data
            # remaining" error you saw earlier) — the client gets no useful
            # information about what went wrong.
            #
            # With this except block, we instead send a proper SSE message
            # describing the error, so any frontend built on top of this
            # can display something like "Something went wrong: <reason>"
            # to the user instead of just hanging or silently failing.

            error_message = str(e)
            # Convert whatever exception object Python raised into a
            # human-readable string so we can send it as text.

            yield f"data: [ERROR] {error_message}\n\n"
            # Send the error back to the client in the same SSE format,
            # tagged clearly with [ERROR] so the frontend can distinguish
            # it from normal AI-generated text chunks.

            yield "data: [DONE]\n\n"
            # Still send [DONE] even after an error, so the client knows
            # for sure that no more data is coming and can stop waiting.

    return StreamingResponse(stream_generator(), media_type="text/event-stream")
    # This is what actually gets returned to whoever called POST /chat.
    # - stream_generator() is passed in (not called with () and awaited
    #   directly) because StreamingResponse itself handles pulling values
    #   out of the generator over time as they're yielded.
    # - media_type="text/event-stream" tells the client "this response is
    #   formatted as Server-Sent Events," which is the standard MIME type
    #   for SSE and lets browsers/clients parse it correctly.