from __future__ import annotations

import scripts.issue2254_probe_context_followup as base
import scripts.issue2254_probe_context_qwen35 as q35


class _TemplateStub:
    def __init__(self):
        self.kwargs = None

    def apply_chat_template(self, messages, **kwargs):
        self.kwargs = kwargs
        assert messages[-1]["role"] == "user"
        return "<|im_start|>user\nQ<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"

    def __call__(self, text, add_special_tokens=False):
        assert add_special_tokens is False
        return {"input_ids": list(range(max(4, len(text) // 8)))}


def test_qwen35_shape_and_depth_matched_layer_configs():
    assert q35.MODEL_ID == "Qwen/Qwen3.5-9B"
    assert q35.N_LAYERS == 32
    assert q35.HIDDEN_DIM == 4096
    assert q35.LAYER_CONFIGS["mid"] == (16, 20, 23)
    assert q35.LAYER_CONFIGS["all"] == tuple(range(32))
    assert q35.TRANSFER_OPERATING_POINTS["evil"]["single"]["layer_config"] == "L16"
    assert q35.TRANSFER_OPERATING_POINTS["hallucination"]["single"]["layer_config"] == "L20"


def test_qwen35_render_forces_thinking_off():
    tokenizer = _TemplateStub()
    rendered = q35.render_qwen35(tokenizer, {"system": None, "user": "Q"})
    assert "<think>\n\n</think>" in rendered
    assert tokenizer.kwargs["enable_thinking"] is False
    assert tokenizer.kwargs["add_generation_prompt"] is True
    assert len(q35.ids_qwen35(tokenizer, {"system": None, "user": "Q"})) >= 4


def test_qwen35_full_grid_is_complete_unique_and_in_range():
    cells = q35.build_cells(n_random=8)
    assert len(cells) == 93
    ids = [base.cell_id(cell) for cell in cells]
    assert len(ids) == len(set(ids))
    for cell in cells:
        if cell["kind"] != "alpha0":
            layers = q35.LAYER_CONFIGS[cell["layer_config"]]
            assert layers
            assert min(layers) >= 0 and max(layers) < q35.N_LAYERS


def test_qwen35_sharding_partitions_every_cell_once():
    cells = q35.build_cells(n_random=8)
    shards = [cells[index::4] for index in range(4)]
    assert sorted(base.cell_id(cell) for shard in shards for cell in shard) == sorted(
        base.cell_id(cell) for cell in cells
    )
