import importlib.util
import json
import threading
from pathlib import Path
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import pytest

from weightlang.bootstrap import TOKENIZER, encode, prompt_tokens
from weightlang.responses import RequestError, events, make_server, validate


class Backend:
    def prepare(self, messages, budget):
        return prompt_tokens(messages)

    def generate(self, ids, budget):
        yield "你", 3, False
        yield "好", 3, False
        yield "", 1, True


def test_tokenizer_roles_and_unicode():
    assert bytes(i - 3 for i in encode("你好🦀")).decode() == "你好🦀"
    assert prompt_tokens([{"role": "user", "content": "assistant"}]) == [1, 259, *encode("assistant"), 263, 260]
    assert TOKENIZER["vocab_size"] == 264


@pytest.mark.parametrize("extra", [{"tools": []}, {"store": True}, {"temperature": 1},
                                  {"stream": 1}, {"max_output_tokens": True},
                                  {"previous_response_id": "resp_1"}, {"text": {"format": {"type": "json_schema"}}}])
def test_unsupported_rejected(extra):
    with pytest.raises(RequestError):
        validate({"model": "test", "input": "hi", **extra}, "test")


def test_replay_messages_and_instructions():
    messages, _ = validate({"model": "test", "instructions": "system", "input": [
        {"role": "user", "content": [{"type": "input_text", "text": "hi"}]},
        {"type": "message", "id": "msg_1", "status": "completed", "role": "assistant", "content": [
            {"type": "output_text", "text": "hello", "annotations": [], "logprobs": []}]},
        {"role": "user", "content": "next"}]}, "test")
    assert [m["content"] for m in messages] == ["system", "hi", "hello", "next"]


def test_stream_lifecycle_and_usage():
    result = list(events(Backend(), {"model": "test"}, [1, 259], 16))
    assert [e["sequence_number"] for e in result] == list(range(len(result)))
    assert result[0]["response"]["output"] == []
    assert result[2]["item"]["content"] == []
    assert result[-1]["type"] == "response.completed"
    final = result[-1]["response"]
    assert final["output"][0]["content"][0]["text"] == "你好"
    assert final["usage"]["output_tokens"] == 7
    assert final["usage"]["total_tokens"] == 9


def test_incomplete_and_failure():
    class Limited(Backend):
        def generate(self, ids, budget):
            yield "a", 1, False
    final = list(events(Limited(), {"model": "test"}, [1], 1))[-1]
    assert final["type"] == "response.incomplete"
    assert final["response"]["incomplete_details"]["reason"] == "max_output_tokens"
    class Broken(Backend):
        def generate(self, ids, budget):
            raise RuntimeError("private details")
            yield
    final = list(events(Broken(), {"model": "test"}, [1], 1))[-1]
    assert final["type"] == "response.failed"
    assert "private details" not in json.dumps(final)


def test_http_json_sse_and_errors():
    server = make_server(Backend(), model_name="test", port=0, api_key="test-local")
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    def request(body, key="test-local"):
        return urlopen(Request(f"http://127.0.0.1:{server.server_port}/v1/responses",
            data=json.dumps(body).encode(), headers={"Content-Type": "application/json", "Authorization": f"Bearer {key}"}), timeout=5)
    try:
        body = {"model": "test", "input": "hi", "max_output_tokens": 16}
        with request(body) as r:
            assert json.load(r)["output"][0]["content"][0]["text"] == "你好"
        with request({**body, "stream": True}) as r:
            assert r.headers["Content-Type"] == "text/event-stream"
            chunks = r.read().decode().strip().split("\n\n")
            for chunk in chunks:
                event, data = chunk.split("\n")
                assert event[7:] == json.loads(data[6:])["type"]
        for extra, key, status in [({}, "wrong", 401), ({"tools": []}, "test-local", 400),
                                    ({"model": "missing"}, "test-local", 404)]:
            with pytest.raises(HTTPError) as error:
                request({**body, **extra}, key)
            assert error.value.code == status
            assert "error" in json.load(error.value)
    finally:
        server.shutdown()
        thread.join()
        server.server_close()


def test_dataset_split_and_loss_mask():
    path = Path(__file__).resolve().parents[1] / "examples/bootstrap/train.py"
    spec = importlib.util.spec_from_file_location("bootstrap_train", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    data = module.dataset()
    families = [{r["family"] for r in data[s]} for s in ("train", "validation", "test")]
    assert not families[0] & families[1] and not families[0] & families[2] and not families[1] & families[2]
    row = data["train"][0]
    x, y = module.tensors([row], 384)
    n = len(prompt_tokens([{"role": "user", "content": row["prompt"]}]))
    assert (y[0, :n - 1] == -100).all()
    assert y[0, n - 1] == encode(row["answer"])[0]
    assert y[0, n + len(encode(row["answer"])) - 1] == 2


def test_byte_decoder_and_context_budget():
    import torch
    from weightlang.bootstrap import ByteModel
    class Artifact:
        def __init__(self, sequence):
            self.sequence = iter(sequence)
        def predict(self, x):
            scores = torch.full((32, 264), -100.0)
            scores[:, next(self.sequence)] = 100.0
            return scores
    backend = ByteModel.__new__(ByteModel)
    backend.context = 32
    backend.artifact = Artifact([*encode("你"), 2])
    chunks = list(backend.generate([1], 8))
    assert "".join(c[0] for c in chunks) == "你"
    assert sum(c[1] for c in chunks) == 4
    assert chunks[-1][2] is True
    backend.artifact = Artifact(encode("你"))
    chunks = list(backend.generate([1], 1))
    assert "".join(c[0] for c in chunks) == "�"
    assert sum(c[1] for c in chunks) == 1
    assert chunks[-1][2] is False
    with pytest.raises(ValueError, match="exceeds"):
        backend.prepare([{"role": "user", "content": "x" * 30}], 8)


@pytest.mark.parametrize("input", [None, [], [{"role": "tool", "content": "x"}],
    [{"role": "user", "content": [{"type": "input_image", "image_url": "x"}]}],
    [{"type": "function_call_output", "call_id": "a", "output": "x"}]])
def test_invalid_inputs(input):
    with pytest.raises(RequestError):
        validate({"model": "test", "input": input}, "test")
