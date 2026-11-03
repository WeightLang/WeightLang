"""Local, stateless text subset of Open Responses. Run with python -m weightlang.responses."""

from __future__ import annotations

import argparse
import copy
import hmac
import json
import logging
import os
import time
import uuid
from http.server import BaseHTTPRequestHandler, HTTPServer

from .bootstrap import ByteModel


class RequestError(ValueError):
    def __init__(self, message, param=None, status=400):
        super().__init__(message)
        self.param, self.status = param, status


def validate(body, model_name):
    if not isinstance(body, dict):
        raise RequestError("request must be a JSON object")
    allowed = {"model", "input", "instructions", "max_output_tokens", "stream", "store", "temperature"}
    for key in body:
        if key not in allowed:
            raise RequestError("unsupported parameter in local text subset", key)
    if body.get("model") != model_name:
        raise RequestError("model not found", "model", 404)
    if type(body.get("stream", False)) is not bool:
        raise RequestError("stream must be boolean", "stream")
    if body.get("store", False) is not False:
        raise RequestError("only store=false is supported", "store")
    temperature = body.get("temperature", 0)
    if type(temperature) not in (int, float) or temperature != 0:
        raise RequestError("only greedy decoding (temperature=0) is supported", "temperature")
    budget = body.get("max_output_tokens", 128)
    if type(budget) is not int or not 1 <= budget <= 4096:
        raise RequestError("must be an integer between 1 and 4096", "max_output_tokens")
    messages = []
    if "instructions" in body:
        if not isinstance(body["instructions"], str):
            raise RequestError("instructions must be a string", "instructions")
        messages.append({"role": "system", "content": body["instructions"]})
    items = body.get("input")
    if isinstance(items, str):
        items = [{"role": "user", "content": items}]
    if not isinstance(items, list) or not items:
        raise RequestError("input must be a string or non-empty message list", "input")
    for item in items:
        if not isinstance(item, dict) or item.get("type", "message") != "message":
            raise RequestError("only text messages are supported", "input")
        if set(item) - {"type", "role", "content", "id", "status"}:
            raise RequestError("unsupported message field", "input")
        role = item.get("role")
        if role not in ("system", "developer", "user", "assistant"):
            raise RequestError("unsupported message role", "input")
        content = item.get("content")
        if isinstance(content, list):
            texts = []
            for part in content:
                expected = "output_text" if role == "assistant" else "input_text"
                if not isinstance(part, dict) or part.get("type") != expected or not isinstance(part.get("text"), str):
                    raise RequestError("unsupported text content", "input")
                if set(part) - {"type", "text", "annotations", "logprobs"} or part.get("annotations") or part.get("logprobs"):
                    raise RequestError("unsupported text content fields", "input")
                texts.append(part["text"])
            content = "".join(texts)
        if not isinstance(content, str):
            raise RequestError("message content must be text", "input")
        messages.append({"role": role, "content": content})
    return messages, budget


def events(backend, body, ids, budget):
    """Both HTTP modes consume the same live generation and lifecycle."""
    response = {"id": "resp_" + uuid.uuid4().hex, "object": "response", "created_at": int(time.time()),
                "status": "in_progress", "model": body["model"], "output": [], "error": None,
                "incomplete_details": None, "usage": None, "instructions": body.get("instructions"),
                "max_output_tokens": budget, "temperature": 0, "tools": [], "tool_choice": "none",
                "parallel_tool_calls": False, "store": False, "previous_response_id": None,
                "metadata": {}, "text": {"format": {"type": "text"}}, "truncation": "disabled",
                "top_p": 1.0, "reasoning": {"effort": None, "summary": None}, "background": False,
                "service_tier": "default", "completed_at": None, "presence_penalty": 0.0,
                "frequency_penalty": 0.0, "top_logprobs": 0, "max_tool_calls": None,
                "safety_identifier": None, "prompt_cache_key": None}
    sequence = 0

    def event(kind, **payload):
        nonlocal sequence
        result = {"type": kind, "sequence_number": sequence, **copy.deepcopy(payload)}
        sequence += 1
        return result

    yield event("response.created", response=response)
    yield event("response.in_progress", response=response)
    item = {"id": "msg_" + uuid.uuid4().hex, "type": "message", "role": "assistant",
            "status": "in_progress", "content": []}
    response["output"].append(item)
    yield event("response.output_item.added", output_index=0, item=item)
    part = {"type": "output_text", "text": "", "annotations": [], "logprobs": []}
    item["content"].append(part)
    location = {"item_id": item["id"], "output_index": 0, "content_index": 0}
    yield event("response.content_part.added", **location, part=part)
    count, eos = 0, False
    try:
        for chunk, consumed, eos in backend.generate(ids, budget):
            count += consumed
            part["text"] += chunk
            if chunk:
                yield event("response.output_text.delta", **location, delta=chunk, logprobs=[])
    except Exception:
        logging.exception("local model generation failed")
        response["status"] = "failed"
        response["error"] = {"code": "server_error", "message": "local model generation failed"}
        yield event("response.failed", response=response)
        return
    item["status"] = response["status"] = "completed" if eos else "incomplete"
    if not eos:
        response["incomplete_details"] = {"reason": "max_output_tokens"}
    response["completed_at"] = int(time.time()) if eos else None
    response["usage"] = {"input_tokens": len(ids), "output_tokens": count, "total_tokens": len(ids) + count,
                         "input_tokens_details": {"cached_tokens": 0}, "output_tokens_details": {"reasoning_tokens": 0}}
    yield event("response.output_text.done", **location, text=part["text"], logprobs=[])
    yield event("response.content_part.done", **location, part=part)
    yield event("response.output_item.done", output_index=0, item=item)
    yield event("response." + response["status"], response=response)


def make_server(backend, *, model_name="weightlang-bootstrap", host="127.0.0.1", port=8000, api_key):
    if not api_key:
        raise ValueError("a local API key is required")

    class Handler(BaseHTTPRequestHandler):
        # Serialized requests keep one CPU model and its memory bounded.
        def setup(self):
            super().setup()
            self.connection.settimeout(30)

        def log_message(self, *_):
            pass

        def send_json(self, status, value):
            data = json.dumps(value, ensure_ascii=False).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_POST(self):
            try:
                if self.path != "/v1/responses":
                    raise RequestError("endpoint not found", status=404)
                if not hmac.compare_digest(self.headers.get("Authorization", "").encode(), ("Bearer " + api_key).encode()):
                    raise RequestError("invalid API key", status=401)
                if self.headers.get_content_type() != "application/json":
                    raise RequestError("Content-Type must be application/json", status=415)
                if self.headers.get("Transfer-Encoding"):
                    raise RequestError("chunked request bodies are unsupported")
                try:
                    length = int(self.headers.get("Content-Length", "0"))
                except ValueError:
                    raise RequestError("invalid Content-Length") from None
                if not 0 < length <= 65536:
                    raise RequestError("request body must contain 1..65536 bytes", status=413)
                try:
                    body = json.loads(self.rfile.read(length))
                except (ValueError, UnicodeError):
                    raise RequestError("invalid JSON") from None
                messages, budget = validate(body, model_name)
                try:
                    ids = backend.prepare(messages, budget)
                except ValueError as exc:
                    raise RequestError(str(exc), "input") from exc
            except RequestError as exc:
                self.send_json(exc.status, {"error": {"message": str(exc), "type": "invalid_request_error",
                                                     "param": exc.param, "code": None}})
                return
            try:
                stream = events(backend, body, ids, budget)
                if body.get("stream", False):
                    self.send_response(200)
                    self.send_header("Content-Type", "text/event-stream")
                    self.send_header("Cache-Control", "no-cache")
                    self.send_header("Connection", "close")
                    self.end_headers()
                    for event in stream:
                        self.wfile.write((f"event: {event['type']}\ndata: " + json.dumps(event) + "\n\n").encode())
                        self.wfile.flush()
                else:
                    final = None
                    for event in stream:
                        if "response" in event:
                            final = event["response"]
                    self.send_json(500 if final["status"] == "failed" else 200, final)
            except (BrokenPipeError, ConnectionResetError, TimeoutError):
                # A disconnected client must not leave generation running in a worker.
                return

    return HTTPServer((host, port), Handler)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--model-name", default="weightlang-bootstrap")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    key = os.environ.get("WEIGHTLANG_API_KEY")
    if not key:
        parser.error("set WEIGHTLANG_API_KEY to a local access token")
    import torch
    torch.set_num_threads(4)
    server = make_server(ByteModel(args.model), model_name=args.model_name, port=args.port, api_key=key)
    print(f"Local text subset listening at http://127.0.0.1:{server.server_port}/v1/responses", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
