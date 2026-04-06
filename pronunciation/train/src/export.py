"""ONNX export and INT8 quantization for on-device deployment."""

import argparse

import torch
from onnxruntime.quantization import QuantType, quantize_dynamic

from .model import ConformerCTCModel
from .vocab import IPAVocab


def export_to_onnx(
    model: ConformerCTCModel,
    output_path: str,
    n_mels: int = 80,
    opset_version: int = 17,
) -> None:
    model.eval()
    dummy_mel = torch.randn(1, 500, n_mels)
    dummy_lengths = torch.tensor([500])

    torch.onnx.export(
        model,
        (dummy_mel, dummy_lengths),
        output_path,
        input_names=["mels", "mel_lengths"],
        output_names=["log_probs", "output_lengths"],
        dynamic_axes={
            "mels": {0: "batch", 1: "time"},
            "mel_lengths": {0: "batch"},
            "log_probs": {0: "time", 1: "batch"},
            "output_lengths": {0: "batch"},
        },
        opset_version=opset_version,
    )
    print(f"Exported ONNX model to {output_path}")


def quantize_int8(onnx_path: str, output_path: str) -> None:
    quantize_dynamic(onnx_path, output_path, weight_type=QuantType.QInt8)
    print(f"Quantized model to {output_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--vocab_path", type=str, required=True)
    parser.add_argument("--output", type=str, default="model.onnx")
    parser.add_argument("--quantized_output", type=str, default="model_int8.onnx")
    # Model args (must match training)
    parser.add_argument("--model_dim", type=int, default=512)
    parser.add_argument("--num_layers", type=int, default=24)
    parser.add_argument("--num_heads", type=int, default=8)
    parser.add_argument("--ffn_dim", type=int, default=2048)
    parser.add_argument("--conv_kernel_size", type=int, default=31)
    args = parser.parse_args()

    vocab = IPAVocab.from_file(args.vocab_path)

    model = ConformerCTCModel(
        vocab_size=vocab.size,
        model_dim=args.model_dim,
        num_heads=args.num_heads,
        ffn_dim=args.ffn_dim,
        num_layers=args.num_layers,
        depthwise_conv_kernel_size=args.conv_kernel_size,
    )

    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    model.load_state_dict(checkpoint["model"])

    export_to_onnx(model, args.output)
    quantize_int8(args.output, args.quantized_output)


if __name__ == "__main__":
    main()
