"""Export parsley's fp32 joint checkpoint as single-sentence encoder and word heads."""
import argparse
import json
from pathlib import Path
import shutil

import torch
from torch import nn

from dataset_joint import collate, encode_records
from predict_joint import load_checkpoint

SOURCE_REVISION = "09a1f8b32248cc303132026ec488f4ad65085ecb"
ENCODER_INPUTS = ["input_ids", "attention_mask", "char_to_sub", "char_ids", "char_features", "lang_id"]
ENCODER_OUTPUTS = ["boundary_logits", "chars", "sub_at_char"]
HEAD_INPUTS = ["chars", "sub_at_char", "starts", "ends"]
HEAD_OUTPUTS = ["pos_logits", "lemma_logits", "arc_scores", "rel_scores"]


class Encoder(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, input_ids, attention_mask, char_to_sub, char_ids, char_features, lang_id):
        batch = dict(input_ids=input_ids, attention_mask=attention_mask, char_to_sub=char_to_sub,
                     char_ids=char_ids, char_features=char_features, lang_ids=lang_id,
                     char_mask=torch.ones_like(char_ids, dtype=torch.bool))
        out = self.model.encode_chars(batch, unpadded=True)
        return tuple(out[k] for k in ENCODER_OUTPUTS)


class Heads(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, chars, sub_at_char, starts, ends):
        out = self.model.word_heads(dict(chars=chars, sub_at_char=sub_at_char), starts, ends,
                                    torch.ones_like(starts, dtype=torch.bool))
        return tuple(out[k] for k in HEAD_OUTPUTS)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    torch.set_num_threads(4)
    model, tokenizer, vocab = load_checkpoint(args.checkpoint)
    model.encoder.set_attn_implementation("eager")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    batch = collate(encode_records([dict(text="This is parsley.", lang="eng")], tokenizer, vocab,
                                  training=False), tokenizer.pad_token_id)
    inputs = tuple(batch[k if k != "lang_id" else "lang_ids"] for k in ENCODER_INPUTS)
    encoder, heads = Encoder(model).eval(), Heads(model).eval()
    with torch.inference_mode():
        _, chars, sub = encoder(*inputs)
        for name, wrapper, values, names, outputs, axes in [
            ("encoder", encoder, inputs, ENCODER_INPUTS, ENCODER_OUTPUTS,
             {k: {1: "subwords" if k in ("input_ids", "attention_mask") else "characters"}
              for k in ENCODER_INPUTS[:-1] + ENCODER_OUTPUTS}),
            ("heads", heads, (chars, sub, torch.tensor([[0, 5, 8, 15]]), torch.tensor([[4, 7, 15, 16]])),
             HEAD_INPUTS, HEAD_OUTPUTS,
             {**{k: {1: "characters"} for k in HEAD_INPUTS[:2]},
              **{k: {1: "words"} for k in HEAD_INPUTS[2:] + HEAD_OUTPUTS},
              **{k: {1: "words", 2: "heads"} for k in HEAD_OUTPUTS[2:]}}),
        ]:
            torch.onnx.export(wrapper, values, str(out / f"{name}.onnx"), input_names=names,
                              output_names=outputs, dynamic_axes=axes, opset_version=17,
                              dynamo=False, external_data=True)
            # Large encoders exceed protobuf's 2 GiB limit. Keep their tensors in
            # one named sidecar rather than hundreds of exporter-generated files.
            if name == "encoder":
                import onnx
                from onnx.external_data_helper import _get_all_tensors
                graph = onnx.load(out / "encoder.onnx", load_external_data=False)
                sidecars = {entry.value for tensor in _get_all_tensors(graph)
                            for entry in tensor.external_data if entry.key == "location"}
                onnx.external_data_helper.load_external_data_for_model(graph, str(out))
                (out / "encoder.onnx.data").unlink(missing_ok=True)
                onnx.save_model(graph, out / "encoder.onnx", save_as_external_data=True,
                                all_tensors_to_one_file=True, location="encoder.onnx.data",
                                size_threshold=1024, convert_attribute=True)
                for sidecar in sidecars - {"encoder.onnx.data"}:
                    (out / sidecar).unlink()
            print(f"Exported {name}", flush=True)
    shutil.copyfile(Path(args.checkpoint) / "tokenizer/tokenizer.json", out / "tokenizer.json")
    shutil.copyfile(Path(args.checkpoint) / "vocab.json", out / "vocab.json")
    (out / "config.json").write_text(json.dumps(dict(
        source_repo="anchpop/lexide-parsley", source_revision=SOURCE_REVISION,
        source_path="training-runs/joint-v2-24L/best", precision="fp32",
        max_subwords=model.encoder.config.max_position_embeddings - 2,
        char_buckets=model.char_buckets), indent=2) + "\n")


if __name__ == "__main__":
    main()
