"""Maintainer-only: writes DDSP-SVC comparison fixtures from the reference's own
PyTorch classes, for `cargo test --test ddsp_svc -- --ignored`.

For each synthesizer (Sins, CombSub, CombSubFast) it builds a DDSP model and a
shallow-diffusion model with random weights, saves them as DDSP-SVC does (each a
`model_*.pt` beside its `config.yaml`), and runs the reference on fixed inputs:
Unit2Control's controls, the synthesized audio (with `torch.rand_like` returning a
recorded noise), NSF-HiFiGAN's mel of that audio and the shallow diffusion of it
(DPM-Solver++, with `torch.randn_like` patched to zeros). It also records the
volume extractor and the loudness mask on random audio. No downloaded checkpoint
is loaded; modules only needed for audio files or other encoders are stubbed.
Run from the repository root:

  uv run --no-project --with torch --with numpy --with pyyaml --with tqdm --with einops \\
      --with local-attention --with librosa \\
      python native-rust/tools/dev/verify_ddsp_svc.py OUT_DIR [REFERENCE_SERVER_DIR]

REFERENCE_SERVER_DIR defaults to reference/server and must hold
voice_changer/DDSP_SVC (a sparse checkout may need `git archive` of it).
"""
import importlib
import json
import sys
import types
from pathlib import Path

import numpy as np
import torch
import yaml

CASES = {
    # name: (type, model settings, speakers, diffusion (layers, chans, hidden))
    "sins": ("Sins", dict(n_harmonics=128, n_mag_allpass=256, n_mag_noise=80), 1, (3, 64, 128)),
    "combsub": ("CombSub", dict(n_mag_allpass=256, n_mag_harmonic=512, n_mag_noise=256), 2, (2, 32, 64)),
    "combsubfast": ("CombSubFast", dict(), 1, (2, 32, 64)),
}
FRAMES = 40
BLOCK = 512
K_STEP, SPEEDUP = 120, 20


def stub_missing():
    """Modules the reference imports for other encoders, pitch detectors or audio
    files; none of them runs here."""

    class Anything:
        def __init__(self, *args, **kwargs):
            pass

    for name in ["pyworld", "parselmouth", "torchcrepe", "transformers", "fairseq", "gin", "torchaudio",
                 "torchaudio.transforms", "sklearn", "sklearn.cluster", "matplotlib", "matplotlib.pylab",
                 "soundfile"]:
        try:
            importlib.import_module(name)
        except Exception:
            sys.modules[name] = types.ModuleType(name)
    sys.modules["transformers"].HubertModel = Anything
    sys.modules["transformers"].Wav2Vec2FeatureExtractor = Anything
    sys.modules["fairseq"].checkpoint_utils = types.SimpleNamespace()
    sys.modules["torchaudio.transforms"].Resample = Anything
    sys.modules["torchaudio"].transforms = sys.modules["torchaudio.transforms"]
    sys.modules["sklearn.cluster"].KMeans = Anything
    sys.modules["matplotlib"].use = lambda *args, **kwargs: None
    sys.modules["matplotlib"].pylab = sys.modules["matplotlib.pylab"]


def randomize(model, seed):
    torch.manual_seed(seed)
    with torch.no_grad():
        for name, value in model.named_parameters():
            if "norm" in name and name.endswith("weight") or name.endswith("stack.1.weight"):
                value.copy_(1 + 0.1 * torch.randn_like(value))
            elif name.endswith("weight_g"):
                continue
            else:
                fan_in = value[0].numel() if value.dim() > 1 else value.numel()
                value.copy_(torch.randn_like(value) / max(fan_in, 1) ** 0.5)


def data_config():
    return dict(sampling_rate=44100, block_size=BLOCK, encoder="contentvec768l12", encoder_out_channels=768,
                encoder_sample_rate=16000, encoder_hop_size=320)


def save(folder, model, config):
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "config.yaml").write_text(yaml.safe_dump(config))
    path = folder / "model_1000.pt"
    torch.save(dict(global_step=1000, model=model.state_dict()), path)
    return path


def main():
    out = Path(sys.argv[1])
    server = Path(sys.argv[2]) if len(sys.argv) > 2 else Path("reference/server")
    sys.path.insert(0, str(server.resolve()))
    stub_missing()
    from voice_changer.DDSP_SVC.models.ddsp import vocoder as ddsp
    from voice_changer.DDSP_SVC.models.diffusion.unit2mel import Unit2Mel
    from voice_changer.DDSP_SVC.models.nsf_hifigan.nvSTFT import STFT
    torch.set_num_threads(2)
    stft = STFT(44100, 128, 2048, 2048, 512, 40, 16000)

    rng = np.random.default_rng(7)
    audio = (0.3 * rng.standard_normal(20_000) * np.abs(np.sin(np.arange(20_000) / 3000))).astype(np.float32)
    volume = ddsp.Volume_Extractor(BLOCK).extract(audio)
    threshold = -45
    mask = (volume > 10 ** (float(threshold) / 20)).astype("float")
    mask = np.pad(mask, (4, 4), constant_values=(mask[0], mask[-1]))
    mask = np.array([np.max(mask[n: n + 9]) for n in range(len(mask) - 8)])
    mask = ddsp.upsample(torch.from_numpy(mask).float().unsqueeze(-1).unsqueeze(0), BLOCK).squeeze(-1)
    out.mkdir(parents=True, exist_ok=True)
    for file, array in dict(audio=audio, volume=volume, mask=mask.numpy()).items():
        np.asarray(array, dtype=np.float32).tofile(out / f"{file}.f32")

    for name, (kind, settings, speakers, (d_layers, d_chans, d_hidden)) in CASES.items():
        folder = out / name
        config = dict(data=data_config(), model=dict(type=kind, n_spk=speakers, **settings),
                      enhancer=dict(type="nsf-hifigan", ckpt="pretrain/nsf_hifigan/model"))
        if kind == "Sins":
            model = ddsp.Sins(44100, BLOCK, n_unit=768, n_spk=speakers, **settings)
        elif kind == "CombSub":
            model = ddsp.CombSub(44100, BLOCK, n_unit=768, n_spk=speakers, **settings)
        else:
            model = ddsp.CombSubFast(44100, BLOCK, n_unit=768, n_spk=speakers)
        randomize(model, 1)
        model_path = save(folder / "ddsp", model.eval(), config)
        model, _ = ddsp.load_model(str(model_path))

        diff_config = dict(data=data_config(),
                           model=dict(type="Diffusion", n_spk=speakers, use_pitch_aug=True, n_layers=d_layers,
                                      n_chans=d_chans, n_hidden=d_hidden),
                           vocoder=dict(type="nsf-hifigan", ckpt="pretrain/nsf_hifigan/model"))
        diff = Unit2Mel(768, speakers, True, 128, d_layers, d_chans, d_hidden)
        randomize(diff, 2)
        save(folder / "diff", diff.eval(), diff_config)

        units = (0.5 * rng.standard_normal((1, FRAMES, 768))).astype(np.float32)
        f0 = (200 + 120 * np.sin(np.arange(FRAMES) / 6)).astype(np.float32).reshape(1, FRAMES, 1)
        vol = (0.02 + 0.2 * np.abs(np.sin(np.arange(FRAMES) / 7))).astype(np.float32).reshape(1, FRAMES, 1)
        uniform = rng.random((1, FRAMES * BLOCK)).astype(np.float32)
        speaker = 1 if speakers > 1 else 0
        spk_id = torch.LongTensor([[speaker + 1]])
        t_units, t_f0, t_vol, t_uniform = map(torch.from_numpy, (units, f0, vol, uniform))

        real_rand_like, real_randn_like = torch.rand_like, torch.randn_like
        torch.rand_like = lambda x, **kwargs: t_uniform.to(x).reshape(x.shape)
        torch.randn_like = torch.zeros_like
        try:
            with torch.no_grad():
                signal, phase, _ = model(t_units, t_f0, t_vol, spk_id=spk_id, infer=True)
                phase_frames = phase[:, ::BLOCK, :] if kind == "Sins" else phase
                controls = model.unit2ctrl(t_units, t_f0, phase_frames, t_vol, spk_id=spk_id)
                controls = torch.cat([controls[key] for key in model.unit2ctrl.output_splits], -1)
                mel = stft.get_mel(signal).transpose(1, 2)
                diffused = diff(t_units, t_f0, t_vol, spk_id=spk_id, gt_spec=mel, infer=True,
                                infer_speedup=SPEEDUP, method="dpm-solver", k_step=K_STEP, use_tqdm=False)
        finally:
            torch.rand_like, torch.randn_like = real_rand_like, real_randn_like

        for file, array in dict(units=units, f0=f0, volume=vol, uniform=uniform, phase=phase_frames.numpy(),
                                controls=controls.numpy(), signal=signal.numpy(), mel=mel.numpy(),
                                diffused=diffused.numpy()).items():
            np.asarray(array, dtype=np.float32).tofile(folder / f"{file}.f32")
        (folder / "meta.json").write_text(json.dumps(dict(frames=FRAMES, speaker=speaker, k_step=K_STEP,
                                                          speedup=SPEEDUP)))
        print(name, "controls", tuple(controls.shape), "signal peak", float(signal.abs().max()),
              "mel range", float(mel.min()), float(mel.max()), "diffused peak", float(diffused.abs().max()))


if __name__ == "__main__":
    main()
