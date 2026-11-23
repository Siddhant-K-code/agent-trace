#!/usr/bin/env python3
"""Make OpenAI and Anthropic SDK calls that agent-strace records.

``scenario.py sdk`` runs this in a clean environment with ``AGENT_TRACE_DIR``
set and the conformance mock server's base URL as its argument. The calls go
through the SDK integrations (``instrument_openai`` and
``instrument_anthropic``) inside one ``run_session``, so they land in one
session, whose ID is printed:

* an OpenAI Chat Completions call, and the same call streamed
* an OpenAI Responses API call
* an Anthropic Messages call, and the same call streamed
* an OpenAI Chat Completions call the mock server answers with HTTP 400

The SDKs make real HTTP requests, so what the integrations record is what
they record against a provider: the client's base URL, and the response ID
and model the server returns.
"""

from __future__ import annotations

import os
import sys

from agent_trace.integrations._base import run_session
from agent_trace.integrations.anthropic import instrument_anthropic
from agent_trace.integrations.openai import instrument_openai
from agent_trace.store import TraceStore

AGENT = "semconv-conformance-sdk"
PROMPT = [{"role": "user", "content": "Say hello"}]


def main(base_url: str) -> int:
    import anthropic
    import openai

    instrument_openai(agent_name=AGENT)
    instrument_anthropic(agent_name=AGENT)
    store = TraceStore(os.environ["AGENT_TRACE_DIR"])
    oai = openai.OpenAI(api_key="test", base_url=f"{base_url}/v1", max_retries=0)
    claude = anthropic.Anthropic(api_key="test", base_url=base_url, max_retries=0)
    with run_session(store, AGENT) as session_id:
        oai.chat.completions.create(model="gpt-4o-mini", messages=PROMPT, max_tokens=50)
        for _ in oai.chat.completions.create(
            model="gpt-4o-mini", messages=PROMPT, max_tokens=50, stream=True,
            stream_options={"include_usage": True},
        ):
            pass
        oai.responses.create(model="gpt-4o-mini", input="Say hello", max_output_tokens=50)
        claude.messages.create(model="claude-sonnet-4-5", messages=PROMPT, max_tokens=50)
        for _ in claude.messages.create(
            model="claude-sonnet-4-5", messages=PROMPT, max_tokens=50, stream=True,
        ):
            pass
        try:
            oai.chat.completions.create(
                model="gpt-4o-mini", max_tokens=50,
                messages=[{"role": "user", "content": "[MOCK_BAD_REQUEST]"}],
            )
        except openai.BadRequestError:
            pass
        else:
            raise SystemExit("the mock server did not fail the [MOCK_BAD_REQUEST] call")
    print(session_id)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1]))
