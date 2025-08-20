# --- AI Processing Functions (XTTS + RVC) ---
from __future__ import annotations

import os
from typing import Optional, Tuple
import soundfile as sf
import torch
from TTS.api import TTS as XTTS
from TTS.tts.configs.xtts_config import XttsConfig
from TTS.tts.models.xtts import Xtts
from rvc_python.infer import RVCInference

def rvc_pipe_convert(
    input_wav_path: str,
    model_pth_path: str,
    f0_up_key: int = 0,
    f0_method: str = "rmvpe",
    index_rate: float = 1.0,
    filter_radius: int = 3,
    rms_mix_rate: float = 0.5,
    protect: float = 0.33,
) -> Optional[str]:
    """Convert voice using daswer123/rvc-python.

    Args:
        input_wav_path: Path to input WAV (e.g., XTTS output)
        model_pth_path: Path to RVC .pth model

    Returns:
        Path to the converted WAV file or None on failure.
    """
    try:
        # Choose device automatically
        device = f"cuda:0" if torch.cuda.is_available() else "cpu"

        # Prepare output path next to input
        in_dir, in_name = os.path.split(input_wav_path)
        base, _ = os.path.splitext(in_name)
        out_path = os.path.join(in_dir or os.getcwd(), f"{base}_rvc.wav")

        # Workaround for PyTorch >= 2.6 weights_only change affecting fairseq checkpoints (hubert)
        try:
            from torch.serialization import add_safe_globals  # type: ignore
            from fairseq.data.dictionary import Dictionary  # type: ignore
            add_safe_globals([Dictionary])
        except Exception:
            pass

        # Initialize rvc-python
        rvc = RVCInference(device=device)
        rvc.load_model(model_pth_path)

        # Try to pass processing parameters if supported; fall back to defaults
        kwargs = {}
        # Common flags used by rvc-python (aligning with CLI/README)
        kwargs.update({
            "method": f0_method,           # harvest/crepe/rmvpe/pm
            "pitch": int(f0_up_key),       # semitone shift
            "index_rate": float(index_rate),
            "filter_radius": int(filter_radius),
            # resample_sr not exposed in our function; leave default (0)
            "rms_mix_rate": float(rms_mix_rate),
            "protect": float(protect),
        })

        try:
            rvc.infer_file(input_wav_path, out_path, **kwargs)
        except TypeError:
            # Older versions may not accept kwargs; retry with minimal signature
            rvc.infer_file(input_wav_path, out_path)

        return out_path if os.path.exists(out_path) else None
    except Exception as e:
        print(f"Error in rvc_pipe_convert: {e}")
        import traceback

        traceback.print_exc()
        return None


def process_text(
    text: str,
    speaker_wav: str,
    tts,  # TTS model instance (XTTS)
    output_dir: str,
) -> Tuple[object, int, str]:
    """Generate speech from text using XTTS and return audio data.

    Args:
        text: Text to convert to speech
        speaker_wav: Path to speaker reference wav
        tts: XTTS model instance
        output_dir: Directory to save the generated audio

    Returns:
        Tuple of (audio_data, sample_rate, output_file_path)
    """
    try:
        os.makedirs(output_dir, exist_ok=True)
        output_file = os.path.join(output_dir, f"tts_output_{int(torch.randint(0, 1_000_000, (1,)).item())}.wav")

        # Generate speech
        tts.tts_to_file(text=text, speaker_wav=speaker_wav, language="en", file_path=output_file)

        # Load the generated audio
        audio_data, sample_rate = sf.read(output_file)

        return audio_data, sample_rate, output_file

    except Exception as e:
        print(f"Error in process_text: {str(e)}")
        import traceback

        traceback.print_exc()
        return None, None, ""


def convert_voice(
    tts_wav_path: str,
    model_pth_path: str,
    f0_up_key: int = 0,
    f0_method: str = "rmvpe",
    index_rate: float = 1.0,
    filter_radius: int = 3,
    rms_mix_rate: float = 0.5,
    protect: float = 0.33,
) -> Optional[str]:
    """Convert voice using rvc-python on a generated TTS WAV file.

    Returns path to converted file or None.
    """
    return rvc_pipe_convert(
        input_wav_path=tts_wav_path,
        model_pth_path=model_pth_path,
        f0_up_key=f0_up_key,
        f0_method=f0_method,
        index_rate=index_rate,
        filter_radius=filter_radius,
        rms_mix_rate=rms_mix_rate,
        protect=protect,
    )


def init_xtts_model_with_deepspeed(
    config_path: str,
    vocab_path: str,
    checkpoint_path: str,
    device: torch.device,
    use_deepspeed: bool = True,
):
    """Initialize XTTS.

    - If use_deepspeed is True: load the low-level XTTS with the provided config/vocab/checkpoint and enable DeepSpeed.
    - If use_deepspeed is False: return the high-level XTTS API (original Coqui API) via `TTS.api.TTS`.

    Returns the initialized model.
    """
    if use_deepspeed:
        if not (os.path.exists(config_path) and os.path.exists(vocab_path) and os.path.exists(checkpoint_path)):
            raise FileNotFoundError("XTTS config/vocab/checkpoint path is invalid")

        config = XttsConfig()
        config.load_json(config_path)
        model = Xtts.init_from_config(config)
        model.load_checkpoint(
            config,
            checkpoint_path=checkpoint_path,
            vocab_path=vocab_path,
            use_deepspeed=True,
        )
        if device.type == "cuda":
            model.cuda()
        return model
    else:
        # Original high-level API. Device is handled internally (uses CUDA if available).
        # Model name is standard for XTTS v2 in Coqui-TTS.
        return XTTS(model_name="tts_models/multilingual/multi-dataset/xtts_v2")


def xtts_infer_to_file(
    model,
    text: str,
    speaker_wav: str,
    out_path: str,
    language: str = "en",
    temperature: float = 0.7,
) -> tuple:
    """Run XTTS inference and save to out_path. Returns (audio, sr, out_path).

    Supports both:
    - High-level API: `TTS.api.TTS` with `tts_to_file` (when use_deepspeed=False in initializer)
    - Low-level XTTS model (when use_deepspeed=True)
    """
    # High-level API path
    if hasattr(model, "tts_to_file"):
        # Use tts() to get waveform directly and write with soundfile to avoid internal writer issues
        wav = model.tts(text=text, speaker_wav=speaker_wav, language=language)
        sr = 24000  # XTTS v2 output sample rate
        sf.write(out_path, wav, sr)
        return wav, sr, out_path

    # Low-level API path
    # Compute conditioning latents from speaker reference
    gpt_cond_latent, speaker_embedding = model.get_conditioning_latents(audio_path=[speaker_wav])

    # Inference
    out = model.inference(
        text,
        language,
        gpt_cond_latent,
        speaker_embedding,
        temperature=temperature,
    )
    wav = out["wav"]
    sr = 24000  # XTTS output sample rate
    sf.write(out_path, wav, sr)
    return wav, sr, out_path


if __name__ == "__main__":
    # Hardcoded DeepSpeed test (no CLI, no auto-discovery). Edit these paths to match your setup.
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[HardTest] Device: {DEVICE}")

    # --- EDIT ME: XTTS v2 checkpoint bundle ---
    XTTS_CONFIG = r"D:\program\RVC-TTS-PIPELINE\RVCModels\xtts_v2\config.json"
    XTTS_VOCAB = r"D:\program\RVC-TTS-PIPELINE\RVCModels\xtts_v2\vocab.json"
    XTTS_CHECKPOINT = r"D:\program\RVC-TTS-PIPELINE\RVCModels\xtts_v2\best_model.pth"  # or model.pth

    # --- EDIT ME: Speaker reference wav ---
    SPEAKER_WAV = r"D:\program\RVC-TTS-PIPELINE\voices\lisa\1.wav"

    # --- EDIT ME: RVC model (dir containing <name>.pth and <name>.json, and the base name) ---
    RVC_MODEL_PTH = r"D:\program\RVC-TTS-PIPELINE\RVCModels\lisa-genshin.pth"  # set to your .pth

    # Fixed test params
    OUTDIR = r"D:\program\RVC-TTS-PIPELINE\generated_sounds"
    os.makedirs(OUTDIR, exist_ok=True)
    TTS_OUT = os.path.join(OUTDIR, "xtts_output.wav")
    TEXT = "This is a fixed DeepSpeed XTTS test sentence for the RVC pipeline."
    F0_KEY = 0

    # Choose whether to use DeepSpeed path for XTTS
    USE_DEEPSPEED = False

    # Sanity checks (will raise with clear messages if paths are wrong)
    required_paths = [
        (SPEAKER_WAV, "SPEAKER_WAV"),
        (RVC_MODEL_PTH, "RVC PTH"),
    ]
    if USE_DEEPSPEED:
        required_paths.extend([
            (XTTS_CONFIG, "XTTS_CONFIG"),
            (XTTS_VOCAB, "XTTS_VOCAB"),
            (XTTS_CHECKPOINT, "XTTS_CHECKPOINT"),
        ])
    for p, label in required_paths:
        if not os.path.exists(p):
            raise SystemExit(f"[HardTest] Missing {label}: {p}")

    # XTTS init
    print(f"[HardTest] Initializing XTTS ({'DeepSpeed low-level' if USE_DEEPSPEED else 'high-level API'})...")
    xtts_model = init_xtts_model_with_deepspeed(
        config_path=XTTS_CONFIG,
        vocab_path=XTTS_VOCAB,
        checkpoint_path=XTTS_CHECKPOINT,
        device=DEVICE,
        use_deepspeed=USE_DEEPSPEED,
    )

    print(f"[HardTest] Generating TTS ({'DeepSpeed' if USE_DEEPSPEED else 'high-level API'})...")
    audio_data, sample_rate, tts_path = xtts_infer_to_file(
        model=xtts_model,
        text=TEXT,
        speaker_wav=SPEAKER_WAV,
        out_path=TTS_OUT,
    )
    if audio_data is None:
        raise SystemExit("[HardTest] XTTS inference failed")
    print(f"[HardTest] TTS saved: {tts_path} (sr={sample_rate})")

    # RVC conversion via rvc-python
    print("[HardTest] Converting voice with rvc-python...")
    rvc_out = convert_voice(
        tts_wav_path=TTS_OUT,
        model_pth_path=RVC_MODEL_PTH,
        f0_up_key=F0_KEY,
    )
    if not rvc_out or not os.path.exists(rvc_out):
        raise SystemExit("[HardTest] RVC conversion failed")

    print(f"[HardTest] Converted saved: {rvc_out}")
