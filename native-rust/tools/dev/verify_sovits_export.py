"""Maintainer-only: verify so-vits-svc import against the reference's PyTorch model.

For each template architecture: a random-weight generator checkpoint (G_test.pth, fp16
and fp32, a speaker count unlike the template's) and its config.json -> the Rust CLI's
`voices import` (in a scratch models folder with stand-in shared models) -> ONNX Runtime
with the NSF source's random nodes zeroed, compared with the reference SynthesizerTrn
with randomness patched to zero, at several lengths. Run from the repository root after
`cargo build --release` in native-rust:

  uv run --no-project --with "torch==2.8.*" --with numpy --with scipy --with matplotlib \\
      --with onnx --with onnxruntime python native-rust/tools/dev/verify_sovits_export.py \\
      WORK_DIR BINARY [REFERENCE_SERVER_DIR]
"""
import json
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import torch

here = Path(__file__).resolve().parent
sys.path.insert(0, str(here))
import make_sovits_templates as T  # noqa: E402
from onnx import helper, numpy_helper  # noqa: E402


def zero_random(model):
    """The graph with its random nodes replaced by zeros (as verify_rvc_export.py)."""
    nodes = []
    for n in model.graph.node:
        if n.op_type in ("RandomNormalLike", "RandomUniformLike"):
            z = n.output[0] + "_zero"
            nodes.append(helper.make_node("Constant", [], [z], value=numpy_helper.from_array(np.array(0.0, np.float32))))
            nodes.append(helper.make_node("Mul", [n.input[0], z], [n.output[0]]))
        elif n.op_type in ("RandomNormal", "RandomUniform"):
            shape = [a.ints for a in n.attribute if a.name == "shape"][0]
            nodes.append(helper.make_node("Constant", [], [n.output[0]], value=numpy_helper.from_array(np.zeros(list(shape), np.float32))))
        else:
            nodes.append(n)
    del model.graph.node[:]
    model.graph.node.extend(nodes)
    return model


class NoRandom:
    """PyTorch's random functions patched to zeros."""
    def __enter__(self):
        self.saved = (torch.randn_like, torch.rand_like, torch.rand, torch.randn)
        torch.randn_like = lambda x, **k: torch.zeros_like(x, **k)
        torch.rand_like = lambda x, **k: torch.zeros_like(x, **k)
        torch.rand = lambda *s, **k: torch.zeros(*s, **{a: b for a, b in k.items() if a in ("dtype", "device")})
        torch.randn = lambda *s, **k: torch.zeros(*s, **{a: b for a, b in k.items() if a in ("dtype", "device")})

    def __exit__(self, *a):
        torch.randn_like, torch.rand_like, torch.rand, torch.randn = self.saved

work = Path(sys.argv[1]).resolve()
binary = sys.argv[2]
server = sys.argv[3] if len(sys.argv) > 3 else "reference/server"
cls = T.model_class(server)
fixtures = here.parents[1] / "tests" / "fixtures"
shutil.rmtree(work, ignore_errors=True)
(work / "models" / "shared").mkdir(parents=True)
shutil.copy(fixtures / "embedder.onnx", work / "models" / "shared" / "contentvec.onnx")
shutil.copy(fixtures / "pitch.onnx", work / "models" / "shared" / "rmvpe.onnx")

results = []
rng = np.random.default_rng(5)
for ssl_dim, gin in T.VARIANTS:
    for half in (False, True):
        speakers = 7
        torch.manual_seed(ssl_dim + gin + half)
        model = cls(1025, 32, **T.BASE, gin_channels=gin, ssl_dim=ssl_dim, n_speakers=speakers,
                    sampling_rate=44100).float().eval()
        with torch.no_grad():
            for v in model.state_dict().values():
                if v.is_floating_point():
                    v.copy_(torch.randn_like(v) * 0.05)
        state = model.state_dict()
        if half:
            state = {k: v.half() if v.is_floating_point() else v for k, v in state.items()}
        source = work / f"src-{ssl_dim}-{int(half)}"
        source.mkdir()
        torch.save({"model": state, "iteration": 1, "learning_rate": 1e-4}, source / "G_100.pth")
        config = {"model": dict(T.BASE, gin_channels=gin, ssl_dim=ssl_dim, n_speakers=speakers,
                                n_layers_q=3, use_spectral_norm=False),
                  "data": {"sampling_rate": 44100, "hop_length": 512, "filter_length": 2048,
                           "max_wav_value": 32768.0},
                  "spk": {f"singer{i}": i for i in range(speakers)}}
        if ssl_dim == 768:
            config["model"].update(speech_encoder="vec768l12", vocoder_name="nsf-hifigan",
                                   vol_embedding=False, use_automatic_f0_prediction=True)
        (source / "config.json").write_text(json.dumps(config))
        name = f"verify-{ssl_dim}-{int(half)}"
        out = subprocess.run([binary, "voices", "import", str(source / "G_100.pth"), "--name", name, "--json"],
                             cwd=work, capture_output=True, text=True)
        assert out.returncode == 0, out.stderr + out.stdout
        manifest = json.loads((work / "models" / name / "model.json").read_text())
        exported = onnx.load(str(work / "models" / name / manifest["model"]))
        meta = json.loads({p.key: p.value for p in exported.metadata_props}["metadata"])["sovits"]
        model.load_state_dict({k: v.float() for k, v in state.items()})
        session = ort.InferenceSession(zero_random(exported).SerializeToString(), providers=["CPUExecutionProvider"])
        for frames in (6, 40, 87):
            f0 = rng.uniform(120, 500, (1, frames)).astype(np.float32)
            feeds = {"feats": rng.standard_normal((1, frames, ssl_dim)).astype(np.float32), "f0": f0,
                     "uv": rng.integers(0, 2, (1, frames)).astype(np.int64),
                     "pitch": rng.integers(1, 256, (1, frames)).astype(np.int64),
                     "sid": np.array([5], np.int64),
                     "noise": (rng.standard_normal((1, 192, frames)) * 0.3).astype(np.float32)}
            with torch.no_grad(), NoRandom():
                expected = model(*[torch.from_numpy(feeds[k]) for k in ("feats", "f0", "uv", "pitch", "sid", "noise")]).numpy()
            got = session.run(None, feeds)[0]
            row = dict(variant=f"ssl {ssl_dim} gin {gin} fp16={half} frames={frames}",
                       manifest_ok=manifest["engine"] == "so-vits-svc" and manifest["channels"] == ssl_dim
                       and manifest["embedder_output"] == ("units9" if ssl_dim == 256 else "unit12"),
                       meta_ok=meta["speakers"] == speakers and meta["ssl_dim"] == ssl_dim and len(meta["names"]) == speakers,
                       shape=got.shape == expected.shape, diff=float(np.abs(got - expected).max()),
                       peak=float(np.abs(expected).max()))
            results.append(row)
            print(json.dumps(row), flush=True)
bad = [r for r in results if not (r["manifest_ok"] and r["meta_ok"] and r["shape"] and r["diff"] < 1e-3)]
print("ALL OK" if not bad else f"FAILED {bad}")
