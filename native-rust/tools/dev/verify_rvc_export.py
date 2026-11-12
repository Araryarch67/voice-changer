"""Maintainer-only: verify Rust --export-rvc on synthetic checkpoints of every variant.

Run from the repository root after `cargo build --release` in native-rust:
  .venv/bin/python native-rust/tools/dev/verify_rvc_export.py . /tmp/vc-verify native-rust/target/release/voice-changer-native

For each architecture: random fp16 checkpoint -> Rust export (plain + realtime) -> ORT with
random nodes zeroed, compared with PyTorch forward with randomness patched to zero, and
(plain) with the reference export2onnx procedure (no constant folding + onnxsim).
"""
import json, subprocess, sys, tempfile
from pathlib import Path
import numpy as np, onnx, onnxruntime as ort, torch
from onnx import helper, numpy_helper

repo = Path(sys.argv[1]).resolve(); work = Path(sys.argv[2]); binary = sys.argv[3]
work.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(repo / "native-rust/tools/dev"))
import make_rvc_templates as T

def zero_random(model):
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
    del model.graph.node[:]; model.graph.node.extend(nodes)
    return model

def run(model, feeds):
    s = ort.InferenceSession(zero_random(model).SerializeToString(), providers=["CPUExecutionProvider"])
    names = {i.name for i in s.get_inputs()}
    return s.run(None, {k: v for k, v in feeds.items() if k in names})[0]

class NoRandom:
    def __enter__(self):
        self.saved = (torch.randn_like, torch.rand_like, torch.rand, torch.randn)
        torch.randn_like = lambda x, **k: torch.zeros_like(x, **k)
        torch.rand_like = lambda x, **k: torch.zeros_like(x, **k)
        torch.rand = lambda *s, **k: torch.zeros(*s, **{a: b for a, b in k.items() if a in ("dtype", "device")})
        torch.randn = lambda *s, **k: torch.zeros(*s, **{a: b for a, b in k.items() if a in ("dtype", "device")})
    def __exit__(self, *a):
        torch.randn_like, torch.rand_like, torch.rand, torch.randn = self.saved

results = []
rng = np.random.default_rng(1)
for (version, sr), config in T.CONFIGS.items():
    for f0 in (True, False):
        channels = T.CHANNELS[version]
        torch.manual_seed(1000 + sr + f0)
        plain = T.model_class(repo, version, f0, False)(*config, is_half=False).float().eval()
        with torch.no_grad():
            for v in plain.state_dict().values():
                if v.is_floating_point():
                    v.copy_(torch.randn_like(v) * 0.05)
        # Real RVC checkpoints: fp16 weights without enc_q; old v1 files have no "version".
        weight = {k: v.half() for k, v in plain.state_dict().items() if not k.startswith("enc_q.")}
        ckpt = {"weight": weight, "config": config, "info": "synthetic", "sr": f"{sr // 1000}k", "f0": int(f0)}
        if version == "v2":
            ckpt["version"] = "v2"
        pth = work / f"{version}-{sr}-{int(f0)}.pth"
        torch.save(ckpt, pth)
        plain.load_state_dict({k: v.float() for k, v in weight.items()}, strict=False)
        frames = 40
        feeds = {"feats": rng.standard_normal((1, frames, channels)).astype(np.float32),
                 "p_len": np.array([frames], np.int64), "sid": np.array([3], np.int64),
                 "pitch": rng.integers(1, 255, (1, frames)).astype(np.int64),
                 "pitchf": rng.uniform(100, 400, (1, frames)).astype(np.float32),
                 "convert_length": np.array([frames * int(np.prod(config[12]))], np.int64)}
        t = {k: torch.from_numpy(v) for k, v in feeds.items()}
        for realtime in (False, True):
            out = work / f"{pth.stem}-{int(realtime)}.onnx"
            info = json.loads(subprocess.run([binary, "--export-rvc", str(pth), str(out)] + (["--realtime"] if realtime else []),
                                             check=True, capture_output=True, text=True).stdout)
            exported = onnx.load(str(out)); onnx.checker.check_model(exported)
            meta = json.loads(exported.metadata_props[-1].value)
            model = T.model_class(repo, version, f0, realtime)(*config, is_half=False).float().eval()
            model.load_state_dict({k: v.float() for k, v in weight.items()}, strict=False)
            args = [t["feats"], t["p_len"]] + ([t["pitch"], t["pitchf"]] if f0 else []) + [t["sid"]] + ([t["convert_length"]] if realtime else [])
            with torch.no_grad(), NoRandom():
                expected = model(*args).numpy()
            got = run(exported, feeds)
            row = dict(variant=f"{version} {sr} f0={f0} rt={realtime}", meta_ok=meta["embChannels"] == channels and meta["f0"] == f0
                       and meta["useFinalProj"] == (version == "v1") and info["channels"] == channels,
                       shape=got.shape == expected.shape, torch_diff=float(np.abs(got - expected).max()), peak=float(np.abs(expected).max()))
            if not realtime:
                # Reference export2onnx: fp32, no constant folding, then onnxsim.
                from onnxsim import simplify
                with tempfile.TemporaryDirectory() as tmp:
                    ref = Path(tmp) / "ref.onnx"
                    names = ["feats", "p_len"] + (["pitch", "pitchf"] if f0 else []) + ["sid"]
                    torch.onnx.export(plain, tuple(args), str(ref), dynamic_axes={"feats": [1], "pitch": [1], "pitchf": [1]},
                                      do_constant_folding=False, opset_version=17, input_names=names, output_names=["audio"])
                    simple, ok = simplify(onnx.load(str(ref)))
                row["reference_diff"] = float(np.abs(run(simple, feeds) - got).max())
            results.append(row); print(json.dumps(row), flush=True)
            out.unlink()
        pth.unlink()
bad = [r for r in results if not (r["meta_ok"] and r["shape"] and r["torch_diff"] < 1e-3 and r.get("reference_diff", 0) < 1e-3)]
print("ALL OK" if not bad else f"FAILED {bad}")
