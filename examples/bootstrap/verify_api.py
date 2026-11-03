"""Verify a real artifact through OpenAI SDK and a locally supplied Open Responses schema."""

import argparse
import hashlib
import json
import threading
from pathlib import Path

import openai
import torch
from jsonschema import Draft202012Validator

from weightlang.bootstrap import ByteModel
from weightlang.responses import make_server


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--schema", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    if args.report.exists() or args.report.is_symlink():
        parser.error("report already exists; select a new path")
    schema = json.loads(args.schema.read_text())
    schemas = schema["components"]["schemas"]
    def validate(name, value):
        Draft202012Validator({"$ref": "#/components/schemas/" + name,
                              "components": schema["components"]}).validate(value)
    event_names = {v["properties"]["type"]["enum"][0]: k for k, v in schemas.items()
                   if k.endswith("StreamingEvent") and "enum" in v.get("properties", {}).get("type", {})}
    torch.set_num_threads(4)
    backend = ByteModel(args.model)
    server = make_server(backend, port=0, api_key="local-verification-only")
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    client = openai.OpenAI(base_url=f"http://127.0.0.1:{server.server_port}/v1",
                           api_key="local-verification-only", max_retries=0, timeout=30)
    request = {"model": "weightlang-bootstrap", "input": "WGT input f32[2], linear(out=4) -> tanh. Output type only.",
               "max_output_tokens": 32, "store": False}
    try:
        response = client.responses.create(**request)
        validate("ResponseResource", response.model_dump(mode="json", exclude_unset=True))
        assert response.status == "completed", response.status
        assert response.output_text == "f32[4]", response.output_text
        received = []
        with client.responses.stream(**request) as stream:
            for event in stream:
                value = event.model_dump(mode="json", exclude_unset=True)
                validate(event_names[event.type], value)
                received.append(value)
            final = stream.get_final_response()
        assert final.output_text == response.output_text
        assert final.usage == response.usage
        assert [e["sequence_number"] for e in received] == list(range(len(received)))
        limited = client.responses.create(**{**request, "max_output_tokens": 1})
        validate("ResponseResource", limited.model_dump(mode="json", exclude_unset=True))
        assert limited.status == "incomplete"
        assert limited.incomplete_details.reason == "max_output_tokens"
        with client.responses.create(**{**request, "max_output_tokens": 1}, stream=True) as stream:
            truncated = [e.model_dump(mode="json", exclude_unset=True) for e in stream]
        for event in truncated:
            validate(event_names[event["type"]], event)
        assert truncated[-1]["type"] == "response.incomplete"
        replay = client.responses.create(**{**request, "input": [
            {"role": "user", "content": request["input"]},
            *[item.model_dump(mode="json", exclude_unset=True) for item in response.output],
            {"role": "user", "content": "Output type only."}]})
        validate("ResponseResource", replay.model_dump(mode="json", exclude_unset=True))
        for extra in [{"input": "x" * 384}, {"tools": []}]:
            try:
                client.responses.create(**{**request, **extra})
            except openai.BadRequestError:
                pass
            else:
                raise AssertionError("unsupported request was accepted")
        report = {"sdk": f"openai=={openai.__version__}", "model": str(args.model),
                  "weights_sha256": backend.artifact.manifest["weights"]["sha256"],
                  "schema_sha256": hashlib.sha256(args.schema.read_bytes()).hexdigest(),
                  "scope": "local stateless text subset, not complete Open Responses compliance",
                  "request": request, "response": response.model_dump(mode="json", exclude_unset=True),
                  "stream": received, "incomplete_stream": truncated,
                  "checks": {"sdk_json": True, "sdk_stream_helper": True, "schema": True,
                             "usage_matches": True, "history_replay_transport": True,
                             "context_overflow_rejected": True, "tools_rejected": True}}
        args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
        print(json.dumps(report["checks"]))
    finally:
        client.close()
        server.shutdown()
        thread.join()
        server.server_close()


if __name__ == "__main__":
    main()
