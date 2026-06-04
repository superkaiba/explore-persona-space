"""#475 diagnostic: which loader produces a working text-logit model for Qwen3.5-27B?

Tests several from_pretrained incantations on the real VLM, each followed by a
tiny text-only forward to confirm valid logits over the full vocab. Prints a
RESULT/LOGITS line per loader and a FAIL+traceback on error. No f-strings with
backslashes (py3.11 forbids them). Frees GPU memory between loaders.
"""

import os
import traceback

import torch

MID = "Qwen/Qwen3.5-27B"
TOK = os.environ.get("HF_TOKEN")
IDS = None  # set after first tokenizer load


def _forward_logits(m, name):
    ids = torch.tensor([[100, 200, 300, 400, 500]], device="cuda:0")
    with torch.no_grad():
        out = m(input_ids=ids)
    logits = out.logits
    print(
        "LOGITS",
        name,
        "shape=" + str(tuple(logits.shape)),
        "finite=" + str(bool(torch.isfinite(logits).all().item())),
    )


def _report(name, m):
    children = [n for n, _ in m.named_children()]
    print(
        "RESULT",
        name,
        "class=" + type(m).__name__,
        "lm_head=" + str(hasattr(m, "lm_head")),
        "children=" + ",".join(children[:10]),
    )
    _forward_logits(m, name)


def t_causal_sdpa():
    from transformers import AutoModelForCausalLM

    m = AutoModelForCausalLM.from_pretrained(
        MID,
        torch_dtype=torch.bfloat16,
        device_map={"": 0},
        trust_remote_code=True,
        attn_implementation="sdpa",
        token=TOK,
    )
    _report("AutoCausalLM_sdpa", m)
    del m
    torch.cuda.empty_cache()


def t_causal_no_attn():
    from transformers import AutoModelForCausalLM

    m = AutoModelForCausalLM.from_pretrained(
        MID,
        torch_dtype=torch.bfloat16,
        device_map={"": 0},
        trust_remote_code=True,
        token=TOK,
    )
    _report("AutoCausalLM_default_attn", m)
    del m
    torch.cuda.empty_cache()


def t_image_text_to_text():
    from transformers import AutoModelForImageTextToText

    m = AutoModelForImageTextToText.from_pretrained(
        MID,
        torch_dtype=torch.bfloat16,
        device_map={"": 0},
        trust_remote_code=True,
        attn_implementation="sdpa",
        token=TOK,
    )
    _report("AutoImageTextToText_sdpa", m)
    del m
    torch.cuda.empty_cache()


def t_causal_textconfig():
    from transformers import AutoConfig, AutoModelForCausalLM

    cfg = AutoConfig.from_pretrained(MID, trust_remote_code=True, token=TOK)
    text_cfg = cfg.text_config
    m = AutoModelForCausalLM.from_pretrained(
        MID,
        config=text_cfg,
        torch_dtype=torch.bfloat16,
        device_map={"": 0},
        trust_remote_code=True,
        attn_implementation="sdpa",
        token=TOK,
    )
    _report("AutoCausalLM_textconfig", m)
    del m
    torch.cuda.empty_cache()


def main():
    import transformers

    print("transformers", transformers.__version__, "torch", torch.__version__)
    for name, fn in [
        ("causal_sdpa", t_causal_sdpa),
        ("causal_default_attn", t_causal_no_attn),
        ("image_text_to_text", t_image_text_to_text),
        ("causal_textconfig", t_causal_textconfig),
    ]:
        print("==== TEST", name, "====")
        try:
            fn()
            print("OK", name)
        except Exception as e:
            print("FAIL", name, type(e).__name__, str(e)[:300])
            traceback.print_exc()
    print("DIAG_ALLDONE")


if __name__ == "__main__":
    main()
