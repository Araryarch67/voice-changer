"""Maintainer-only: writes Diffusion-SVC comparison fixtures from the reference's own
PyTorch classes, for `cargo test --test diffusion_svc -- --ignored`.

For each architecture below it builds a combo checkpoint (.ptc) with random weights,
runs the reference naive model, the diffusion model's conditioning, one denoiser step
and the whole shallow diffusion (DPM-Solver++, with torch.randn patched to zeros so
the result is deterministic), and writes inputs and outputs as raw float32 beside it.
No downloaded checkpoint is loaded. Run from the repository root:

  uv run --no-project --with torch --with einops --with local-attention --with numpy --with pyyaml --with tqdm \\
      python native-rust/tools/dev/verify_diffusion_svc.py OUT_DIR [REFERENCE_SERVER_DIR]

REFERENCE_SERVER_DIR defaults to reference/server and must hold
voice_changer/DiffusionSVC (a sparse checkout may need `git archive` of it).
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch

ARCHITECTURES = {
    # name: (speakers, pitch_aug, naive layers, naive chans, diff layers, diff chans, diff hidden, k_step_max)
    "single": (1, False, 3, 256, 4, 128, 256, 100),
    "speakers": (3, True, 2, 128, 3, 64, 128, 20),
}
RUNS = {
    # name: [(k_step, speedup), ...]
    "single": [(100, 10), (100, 50), (40, 4)],
    "speakers": [(20, 10), (20, 5)],
}


class DotDict(dict):
    """Like the dict subclass Diffusion-SVC's tools use for configs."""


def config(kind, speakers, pitch_aug, layers, chans, hidden, k_step_max):
    model = dict(type=kind, n_spk=speakers, use_pitch_aug=pitch_aug, n_layers=layers, n_chans=chans,
                 use_speaker_encoder=False)
    if kind == "Diffusion":
        model.update(n_hidden=hidden, k_step_max=k_step_max)
    return DotDict(data=dict(encoder="contentvec768l12", encoder_out_channels=768, sampling_rate=44100,
                             block_size=512, speaker_encoder_out_channels=256),
                   model=model, vocoder=dict(type="nsf-hifigan", ckpt="pretrain/nsf_hifigan/model"))


def randomize(model, seed):
    torch.manual_seed(seed)
    with torch.no_grad():
        for name, value in model.named_parameters():
            if name.endswith("norm.weight") or ".norm." in name and name.endswith("weight") or name.endswith("stack.1.weight") \
                    or name.endswith("net.0.weight"):
                value.copy_(1 + 0.1 * torch.randn_like(value))
            else:
                fan_in = value[0].numel() if value.dim() > 1 else value.numel()
                value.copy_(torch.randn_like(value) / max(fan_in, 1) ** 0.5)


def main():
    out = Path(sys.argv[1])
    server = Path(sys.argv[2]) if len(sys.argv) > 2 else Path("reference/server")
    sys.path.insert(0, str(server.resolve()))
    # The vocoder module only matters for audio; stub it so torchaudio and librosa are
    # not needed to build the models.
    import types
    stub = types.ModuleType("vocoder")
    stub.Vocoder = object
    sys.modules["voice_changer.DiffusionSVC.inferencer.diffusion_svc_model.diffusion.vocoder"] = stub
    from voice_changer.DiffusionSVC.inferencer.diffusion_svc_model.diffusion.unit2mel import load_svc_model
    torch.set_num_threads(2)
    for name, (speakers, pitch_aug, n_layers, n_chans, d_layers, d_chans, d_hidden, k_max) in ARCHITECTURES.items():
        folder = out / name
        folder.mkdir(parents=True, exist_ok=True)
        naive_args = config("Naive", speakers, pitch_aug, n_layers, n_chans, None, None)
        diff_args = config("Diffusion", speakers, pitch_aug, d_layers, d_chans, d_hidden, k_max)

        class Args(dict):
            def __getattr__(self, key):
                value = self[key]
                return Args(value) if isinstance(value, dict) else value

        naive = load_svc_model(Args(naive_args), 128).eval()
        diff = load_svc_model(Args(diff_args), 128).eval()
        randomize(naive, 1)
        randomize(diff, 2)
        torch.save(dict(diff_config_dict=diff_args, naive_config_dict=dict(naive_args),
                        diff_model=dict(model=diff.state_dict()), naive_model=dict(model=naive.state_dict())),
                   folder / "combo.ptc")

        frames = 150
        rng = np.random.default_rng(3)
        units = rng.standard_normal((1, frames, 768)).astype(np.float32)
        f0 = (220 * (1 + 0.1 * np.sin(np.arange(frames) / 9))).astype(np.float32).reshape(1, frames, 1)
        volume = np.abs(0.2 * np.sin(np.arange(frames) / 13)).astype(np.float32).reshape(1, frames, 1)
        speaker = 1 if speakers > 1 else 0
        spk_id = torch.LongTensor([[speaker + 1]])
        t_units, t_f0, t_volume = map(torch.from_numpy, (units, f0, volume))
        aug = torch.zeros(1, 1)
        with torch.no_grad():
            naive_mel = naive(t_units, t_f0, t_volume, spk_id=spk_id, aug_shift=aug, infer=True)
            captured = {}
            hook = diff.decoder.register_forward_pre_hook(lambda m, args: captured.setdefault("cond", args[0]))
            # A run without gt_spec only to capture the conditioning.
            diff(t_units, t_f0, t_volume, spk_id=spk_id, aug_shift=aug, gt_spec=naive_mel, infer=True,
                 infer_speedup=k_max // 2, method="dpm-solver", k_step=k_max, use_tqdm=False)
            hook.remove()
            condition = captured["cond"].transpose(1, 2).contiguous()
            spec = torch.from_numpy(rng.standard_normal((1, 1, 128, frames)).astype(np.float32))
            step = torch.tensor([37.5])
            noise = diff.decoder.denoise_fn(spec, step, condition)
            mels = []
            real_randn_like = torch.randn_like
            torch.randn_like = torch.zeros_like
            try:
                for k_step, speedup in RUNS[name]:
                    mels.append(diff(t_units, t_f0, t_volume, spk_id=spk_id, aug_shift=aug, gt_spec=naive_mel,
                                     infer=True, infer_speedup=speedup, method="dpm-solver", k_step=k_step,
                                     use_tqdm=False).numpy())
            finally:
                torch.randn_like = real_randn_like
        for file, array in dict(units=units, f0=f0, volume=volume, naive=naive_mel.numpy(),
                                condition=condition.numpy(), denoise_in=spec.numpy(),
                                denoise_out=noise.numpy()).items():
            array.astype(np.float32).tofile(folder / f"{file}.f32")
        for i, mel in enumerate(mels):
            mel.astype(np.float32).tofile(folder / f"mel{i}.f32")
        (folder / "meta.json").write_text(json.dumps(dict(
            frames=frames, speaker=speaker, step=37.5, hidden=d_hidden,
            runs=[dict(k_step=k, speedup=s) for k, s in RUNS[name]])))
        print(name, "naive peak", float(naive_mel.abs().max()), "noise peak", float(noise.abs().max()),
              "mel peaks", [float(np.abs(m).max()) for m in mels])


if __name__ == "__main__":
    main()
