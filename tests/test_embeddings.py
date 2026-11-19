from types import SimpleNamespace

import numpy as np
import torch

from src.embeddings import HuggingFaceEmbedder, mean_pool, normalize_embeddings


class TinyTokenizer:
    def __call__(self, texts, **kwargs):
        ids = [[1, 2, 0] if text == "first" else [2, 0, 0] for text in texts]
        input_ids = torch.tensor(ids, dtype=torch.long)
        return {"input_ids": input_ids, "attention_mask": (input_ids != 0).long()}


class TinyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.config = SimpleNamespace(hidden_size=3)

    def forward(self, input_ids, attention_mask):
        values = input_ids.to(torch.float32).unsqueeze(-1)
        hidden = torch.cat((values, values * 2, values * 3), dim=-1)
        return SimpleNamespace(last_hidden_state=hidden)


def test_mean_pool_masks_padding_and_normalizes() -> None:
    hidden = torch.tensor([[[1.0, 0.0], [0.0, 2.0], [9.0, 9.0]]])
    mask = torch.tensor([[1, 1, 0]])
    pooled = mean_pool(hidden, mask)
    normalized = normalize_embeddings(pooled)
    assert torch.allclose(pooled, torch.tensor([[0.5, 1.0]]))
    assert torch.allclose(torch.linalg.vector_norm(normalized, dim=1), torch.ones(1))


def test_embedder_batches_output_as_normalized_float32() -> None:
    embedder = HuggingFaceEmbedder(
        "unused-test-model",
        batch_size=2,
        device="cpu",
        tokenizer=TinyTokenizer(),
        model=TinyModel(),
    )
    vectors = embedder.embed_documents(["first", "second"])
    assert vectors.shape == (2, 3)
    assert vectors.dtype == np.float32
    assert np.allclose(np.linalg.norm(vectors, axis=1), 1.0)

