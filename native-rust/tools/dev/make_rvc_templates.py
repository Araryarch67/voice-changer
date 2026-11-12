"""Maintainer-only: regenerate the RVC graph templates in native-rust/templates/.

Users never run this; `voice-changer-native --export-rvc` fills these templates from a
checkpoint in Rust. A template is the reference PyTorch exporter's ONNX graph for one
architecture, with every weight removed and a recipe recording which checkpoint tensor
(or weight-norm product) each initializer comes from. Weights are random here, so the
recipe is found by exact value matching. Run from the repository root:
  .venv/bin/python native-rust/tools/dev/make_rvc_templates.py
"""
import json
from pathlib import Path
import sys
import tempfile

import numpy as np
import onnx
import torch
from onnx import numpy_helper

# Standard RVC v2 architectures: (spec_channels, ..., upsample_rates, initial channel, kernels, speakers, gin, sr).
BASE = [192, 192, 768, 2, 6, 3, 0, "1", [3, 7, 11], [[1, 3, 5], [1, 3, 5], [1, 3, 5]]]
CONFIGS = {
    32000: [513, 32, *BASE, [10, 8, 2, 2], 512, [20, 16, 4, 4], 109, 256, 32000],
    40000: [1025, 32, *BASE, [10, 10, 2, 2], 512, [16, 16, 4, 4], 109, 256, 40000],
    48000: [1025, 32, *BASE, [12, 10, 2, 2], 512, [24, 20, 4, 4], 109, 256, 48000],
}


def main():
    torch.set_num_threads(2)
    repo = Path(__file__).resolve().parents[3]
    sys.path.insert(0, str(repo / "reference" / "server"))
    from voice_changer.RVC.onnxExporter.SynthesizerTrnMs768NSFsid_ONNX import SynthesizerTrnMs768NSFsid_ONNX

    class RealtimeSynthesizer(SynthesizerTrnMs768NSFsid_ONNX):
        def forward(self, phone, phone_lengths, pitch, nsff0, sid, convert_length):
            g = self.emb_g(sid).unsqueeze(-1)
            m_p, logs_p, x_mask = self.enc_p(phone, pitch, phone_lengths)
            z_p = (m_p + torch.exp(logs_p) * torch.randn_like(m_p) * 0.66666) * x_mask
            z = self.flow(z_p, x_mask, g=g, reverse=True)
            audio = self.dec.infer_realtime(z * x_mask, nsff0, g=g, convert_length=convert_length[0])
            return torch.clip(audio[0, 0], -1.0, 1.0)

    out_dir = repo / "native-rust" / "templates"
    out_dir.mkdir(exist_ok=True)
    for sr, config in CONFIGS.items():
        for realtime in (False, True):
            torch.manual_seed(sr + realtime)
            model_type = RealtimeSynthesizer if realtime else SynthesizerTrnMs768NSFsid_ONNX
            model = model_type(*config, is_half=False).float().eval()
            with torch.no_grad():
                for value in model.state_dict().values():
                    if value.is_floating_point():
                        value.copy_(torch.randn_like(value))
            path = out_dir / f"rvc-v2-{sr}{'-realtime' if realtime else ''}.template.onnx"
            export(model, config, realtime, path)


def export(model, config, realtime, path):
    frames = 64
    inputs = (torch.zeros(1, frames, 768), torch.tensor([frames]),
              torch.full((1, frames), 100, dtype=torch.int64),
              torch.full((1, frames), 220.0), torch.tensor([0]))
    input_names = ["feats", "p_len", "pitch", "pitchf", "sid"]
    if realtime:
        inputs += (torch.tensor([24 * (config[-1] // 100)]),)
        input_names.append("convert_length")
    with tempfile.TemporaryDirectory() as tmp:
        temporary = Path(tmp) / "model.onnx"
        torch.onnx.export(model, inputs, str(temporary), opset_version=17,
            input_names=input_names, output_names=["audio"],
            dynamic_axes={"feats": {1: "frames"}, "pitch": {1: "frames"}, "pitchf": {1: "frames"}, "audio": {0: "samples"}},
            do_constant_folding=True)
        exported = onnx.load(str(temporary))
    onnx.checker.check_model(exported)

    state = {k: v.numpy() for k, v in model.state_dict().items() if v.is_floating_point() and not k.startswith("enc_q.")}
    candidates = {}
    for key, value in state.items():
        candidates.setdefault(value.astype(np.float32).tobytes(), ["sd", key])
        if value.ndim == 2:
            candidates.setdefault(np.ascontiguousarray(value.T).astype(np.float32).tobytes(), ["sd_t", key])
        if key.endswith(".weight_g"):
            base = key[: -len("_g")]
            normed = torch._weight_norm(torch.from_numpy(state[base + "_v"]), torch.from_numpy(value), 0).numpy()
            candidates.setdefault(normed.astype(np.float32).tobytes(), ["wn", base])
    recipe = []
    for tensor in exported.graph.initializer:
        array = numpy_helper.to_array(tensor)
        source = candidates.get(array.astype(np.float32).tobytes()) if array.dtype == np.float32 else None
        if source is None and array.dtype == np.float32:
            # Constant folding rounds weight-norm products slightly differently from torch._weight_norm.
            for bytes_, candidate in candidates.items():
                value = np.frombuffer(bytes_, np.float32)
                if candidate[0] == "wn" and value.size == array.size and np.allclose(value, array.ravel(), rtol=1e-5, atol=1e-6):
                    if value.reshape(array.shape).shape == tuple(state[candidate[1] + "_v"].shape):
                        source = candidate
                        break
        if source is None:
            raise ValueError(f"{path.name}: initializer {tensor.name} {array.shape} is not derived from a checkpoint tensor")
        recipe.append([tensor.name, *source])
        tensor.ClearField("raw_data")
        tensor.ClearField("float_data")
    # Weight-derived data must only live in initializers, never in Constant nodes.
    weights = {bytes(k) for k in candidates}
    for node in exported.graph.node:
        for attribute in node.attribute:
            if attribute.type == onnx.AttributeProto.TENSOR:
                value = numpy_helper.to_array(attribute.t)
                if value.size > 16 and value.astype(np.float32).tobytes() in weights:
                    raise ValueError(f"{path.name}: Constant {node.name} holds checkpoint data")
    template = dict(format=1, config=config, realtime=realtime, initializers=recipe, state_keys=list(model.state_dict().keys()))
    entry = exported.metadata_props.add()
    entry.key = "vc_template"
    entry.value = json.dumps(template, separators=(",", ":"))
    onnx.save(exported, str(path))
    print(f"{path.name}: {len(recipe)} initializers, {path.stat().st_size} bytes")


if __name__ == "__main__":
    main()
