"""Maintainer-only: regenerate the so-vits-svc graph templates in native-rust/templates/.

Like make_rvc_templates.py: each template is the ONNX graph of the reference's
SoVitsSvc40 SynthesizerTrn inference path for one architecture, with every weight
removed and a recipe saying which checkpoint tensor (its transpose, or its weight-norm
product) fills each initializer. `voices import` fills them from a G_*.pth in Rust.

The graph takes ContentVec features `feats` [1,T,ssl_dim], `f0` [1,T] in Hz (unvoiced
frames filled in), `uv` [1,T] (1 voiced), the coarse pitch `pitch` [1,T] (1..255),
`sid` [1] and the encoder's noise `noise` [1,192,T] already multiplied by the noise
scale, and returns `audio` [samples]. The NSF source keeps its own random phase and
noise, as in the reference.

Run from the repository root with the reference's SoVitsSvc40 folder available:
  uv run --no-project --with torch --with numpy --with scipy --with matplotlib --with onnx \\
      python native-rust/tools/dev/make_sovits_templates.py [REFERENCE_SERVER_DIR]
"""
import json
from pathlib import Path
import sys
import tempfile
import zlib

import numpy as np
import onnx
import torch
from onnx import numpy_helper

BASE = dict(inter_channels=192, hidden_channels=192, filter_channels=768, n_heads=2, n_layers=6,
            kernel_size=3, p_dropout=0.1, resblock="1", resblock_kernel_sizes=[3, 7, 11],
            resblock_dilation_sizes=[[1, 3, 5], [1, 3, 5], [1, 3, 5]], upsample_rates=[8, 8, 2, 2, 2],
            upsample_initial_channel=512, upsample_kernel_sizes=[16, 16, 4, 4, 4])
# (ssl_dim, gin_channels): 4.0's default and 4.1's default (vec768l12).
VARIANTS = [(256, 256), (768, 768)]
# The order Rust compares a config.json against: BASE's values, then ssl_dim and gin.
CONFIG_KEYS = list(BASE) + ["ssl_dim", "gin_channels"]
SPEAKERS = 4


def template_name(ssl_dim, gin):
    return f"sovits-{ssl_dim}-{gin}.template.onnx.zz"


def model_class(server):
    sys.path.insert(0, str(Path(server).resolve()))
    from voice_changer.SoVitsSvc40.models.models import SynthesizerTrn

    class Infer(SynthesizerTrn):
        def forward(self, feats, f0, uv, pitch, sid, noise):
            c = feats.transpose(1, 2)
            g = self.emb_g(sid).unsqueeze(-1)
            x_mask = torch.ones_like(c[:, :1, :])
            x = self.pre(c) * x_mask + self.emb_uv(uv).transpose(1, 2)
            e = self.enc_p
            h = x + e.f0_emb(pitch).transpose(1, 2)
            h = e.enc_(h * x_mask, x_mask)
            stats = e.proj(h) * x_mask
            m, logs = torch.split(stats, e.out_channels, dim=1)
            z_p = (m + noise * torch.exp(logs)) * x_mask
            z = self.flow(z_p, x_mask, g=g, reverse=True)
            return self.dec(z * x_mask, g=g, f0=f0)[0, 0]

    return Infer


def build(cls, ssl_dim, gin):
    return cls(1025, 32, **BASE, gin_channels=gin, ssl_dim=ssl_dim, n_speakers=SPEAKERS,
               sampling_rate=44100).float().eval()


def main():
    server = sys.argv[1] if len(sys.argv) > 1 else "reference/server"
    cls = model_class(server)
    out_dir = Path(__file__).resolve().parents[2] / "templates"
    torch.set_num_threads(2)
    for ssl_dim, gin in VARIANTS:
        torch.manual_seed(ssl_dim + gin)
        model = build(cls, ssl_dim, gin)
        with torch.no_grad():
            for value in model.state_dict().values():
                if value.is_floating_point():
                    value.copy_(torch.randn_like(value))
        export(model, ssl_dim, gin, out_dir / template_name(ssl_dim, gin))


def export(model, ssl_dim, gin, path):
    frames = 40
    inputs = (torch.zeros(1, frames, ssl_dim), torch.full((1, frames), 220.0),
              torch.ones(1, frames, dtype=torch.int64), torch.full((1, frames), 100, dtype=torch.int64),
              torch.tensor([0]), torch.zeros(1, 192, frames))
    names = ["feats", "f0", "uv", "pitch", "sid", "noise"]
    axes = {"feats": {1: "frames"}, "f0": {1: "frames"}, "uv": {1: "frames"}, "pitch": {1: "frames"},
            "noise": {2: "frames"}, "audio": {0: "samples"}}
    with tempfile.TemporaryDirectory() as tmp:
        temporary = Path(tmp) / "model.onnx"
        torch.onnx.export(model, inputs, str(temporary), opset_version=17, input_names=names,
                          output_names=["audio"], dynamic_axes=axes, do_constant_folding=True, dynamo=False)
        exported = onnx.load(str(temporary))
    onnx.checker.check_model(exported)
    skip = ("enc_q.", "f0_decoder.")
    state = {k: v.numpy() for k, v in model.state_dict().items() if v.is_floating_point() and not k.startswith(skip)}
    # Candidates by value, each with the shape it produces: a vector's transpose has
    # the same bytes, so the shape decides between them.
    candidates = {}
    for key, value in state.items():
        candidates.setdefault(value.astype(np.float32).tobytes(), []).append((["sd", key], value.shape))
        if value.ndim == 2:
            transposed = np.ascontiguousarray(value.T)
            candidates.setdefault(transposed.astype(np.float32).tobytes(), []).append((["sd_t", key], transposed.shape))
        if key.endswith(".weight_g"):
            base = key[: -len("_g")]
            normed = torch._weight_norm(torch.from_numpy(state[base + "_v"]), torch.from_numpy(value), 0).numpy()
            candidates.setdefault(normed.astype(np.float32).tobytes(), []).append((["wn", base], normed.shape))

    def lookup(array):
        for source, shape in candidates.get(array.astype(np.float32).tobytes(), []):
            if tuple(shape) == array.shape:
                return source
        return None

    recipe = []
    for tensor in exported.graph.initializer:
        array = numpy_helper.to_array(tensor)
        source = lookup(array) if array.dtype == np.float32 else None
        if source is None and array.dtype == np.float32:
            for bytes_, options in candidates.items():
                value = np.frombuffer(bytes_, np.float32)
                for candidate, shape in options:
                    if candidate[0] == "wn" and tuple(shape) == array.shape and np.allclose(value, array.ravel(), rtol=1e-5, atol=1e-6):
                        source = candidate
                        break
                if source is not None:
                    break
        if source is None:
            if array.dtype != np.float32 or array.size <= 16:
                continue  # graph constants (shapes, small scalars) stay in the template
            raise ValueError(f"{path.name}: initializer {tensor.name} {array.shape} is not derived from a checkpoint tensor")
        recipe.append([tensor.name, *source])
        tensor.ClearField("raw_data")
        tensor.ClearField("float_data")
    weights = {bytes(k) for k in candidates}
    for node in exported.graph.node:
        for attribute in node.attribute:
            if attribute.type == onnx.AttributeProto.TENSOR:
                value = numpy_helper.to_array(attribute.t)
                if value.size > 16 and value.astype(np.float32).tobytes() in weights:
                    raise ValueError(f"{path.name}: Constant {node.name} holds checkpoint data")
    config = [BASE[k] for k in BASE] + [ssl_dim, gin]
    template = dict(format=1, config=config, channels=ssl_dim, f0=True, initializers=recipe,
                    state_keys=list(model.state_dict().keys()))
    entry = exported.metadata_props.add()
    entry.key = "vc_template"
    entry.value = json.dumps(template, separators=(",", ":"))
    path.write_bytes(zlib.compress(exported.SerializeToString(), 9))
    print(f"{path.name}: {len(recipe)} initializers, {path.stat().st_size} bytes")


if __name__ == "__main__":
    main()
