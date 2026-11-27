"""Score policy passages locally with the cross-encoder used in the retrieval benchmark."""
import hashlib
import os
import threading
import time
from functools import cache
from pathlib import Path
from tempfile import NamedTemporaryFile

import httpx
import numpy as np
import onnxruntime as ort
from tokenizers import Tokenizer


MODEL = "cross-encoder/ms-marco-MiniLM-L6-v2"
REVISION = "233902d25c440f23af6f7d6e94d2946bac0bee0a"
MODEL_DIR = Path(os.getenv("CLAIMS_MODEL_DIR", Path(__file__).resolve().parents[2] / ".local" / "models")) / "policy-reranker" / REVISION
ASSETS = {
    "tokenizer.json": "d241a60d5e8f04cc1b2b3e9ef7a4921b27bf526d9f6050ab90f9267a1f9e5c66",
    "onnx/model_quint8_avx2.onnx": "c80a8b34256ea453093d612e3ac48d3d965a0c0a48c7906709af8b8e28461bf9",
}
_load_lock = threading.Lock()
_retry_after = 0.0


def model_asset(name):
    path = MODEL_DIR / Path(name).name
    expected = ASSETS[name]
    if path.is_file():
        with path.open("rb") as stream:
            if hashlib.file_digest(stream, "sha256").hexdigest() == expected:
                return path

    path.parent.mkdir(parents=True, exist_ok=True)
    url = f"https://huggingface.co/{MODEL}/resolve/{REVISION}/{name}"
    with NamedTemporaryFile(dir=path.parent, delete=False) as stream:
        temporary = Path(stream.name)
        try:
            digest = hashlib.sha256()
            with httpx.stream("GET", url, follow_redirects=True, timeout=20) as response:
                response.raise_for_status()
                for data in response.iter_bytes():
                    digest.update(data)
                    stream.write(data)
            stream.close()
            if digest.hexdigest() != expected:
                raise ValueError("Policy reranker asset failed checksum verification")
            os.replace(temporary, path)
        finally:
            stream.close()
            temporary.unlink(missing_ok=True)
    return path


@cache
def load_reranker():
    global _retry_after
    if time.monotonic() < _retry_after:
        raise RuntimeError("Policy reranker initialization is waiting to retry")
    try:
        tokenizer = Tokenizer.from_file(str(model_asset("tokenizer.json")))
        tokenizer.enable_truncation(max_length=512)
        tokenizer.enable_padding(pad_id=0, pad_token="[PAD]")
        options = ort.SessionOptions()
        options.intra_op_num_threads = 3
        model = ort.InferenceSession(str(model_asset("onnx/model_quint8_avx2.onnx")), sess_options=options,
                                     providers=["CPUExecutionProvider"])
        return tokenizer, model
    except Exception:
        _retry_after = time.monotonic() + 60
        raise


def rerank_scores(query, passages):
    with _load_lock:
        tokenizer, model = load_reranker()
    batch = tokenizer.encode_batch([(query, passage) for passage in passages])
    inputs = {name: np.asarray([getattr(item, field) for item in batch], dtype=np.int64)
              for name, field in (("input_ids", "ids"), ("attention_mask", "attention_mask"), ("token_type_ids", "type_ids"))}
    scores = model.run(None, inputs)[0].reshape(-1)
    if len(scores) != len(passages) or not np.isfinite(scores).all():
        raise ValueError("Policy reranker returned invalid scores")
    return scores.tolist()
