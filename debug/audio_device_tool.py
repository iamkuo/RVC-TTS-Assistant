#!/usr/bin/env python
"""
Audio Device Tool
- List output-capable devices
- Select by --index or --name (substring, case-insensitive)
- Set as default output device for this process
- Play a short test tone to confirm

Usage examples:
  python debug/audio_device_tool.py --list
  python debug/audio_device_tool.py --index 5
  python debug/audio_device_tool.py --name "Speakers"
  python debug/audio_device_tool.py            # interactive selection
"""
import argparse
import sys
import time

import numpy as np
import sounddevice as sd


def list_devices():
    devices = sd.query_devices()
    headers = ["Index", "Name", "MaxOut", "HostAPI"]
    print(f"{headers[0]:>5}  {headers[1]:<60}  {headers[2]:>6}  {headers[3]}")
    hostapis = sd.query_hostapis()
    for i, d in enumerate(devices):
        hostapi_name = hostapis[d["hostapi"]]["name"] if isinstance(d.get("hostapi"), int) else "?"
        print(f"{i:>5}  {d['name']:<60}  {d.get('max_output_channels', 0):>6}  {hostapi_name}")
    return devices


def choose_device(devices, index=None, name_sub=None):
    # Resolve by index
    if index is not None:
        if 0 <= index < len(devices) and devices[index].get("max_output_channels", 0) > 0:
            return index
        raise ValueError(f"Invalid --index {index} or device has no output channels")

    # Resolve by name substring
    if name_sub:
        name_sub = name_sub.lower()
        for i, d in enumerate(devices):
            if d.get("max_output_channels", 0) > 0 and name_sub in d["name"].lower():
                return i
        raise ValueError(f"No output device matching substring: {name_sub}")

    # Interactive choose: show only output-capable
    out_idxs = [i for i, d in enumerate(devices) if d.get("max_output_channels", 0) > 0]
    if not out_idxs:
        raise RuntimeError("No output-capable audio devices found")

    print("\nSelect an output device by index (or press Enter to cancel):")
    while True:
        s = input("> ").strip()
        if s == "":
            return None
        try:
            i = int(s)
        except ValueError:
            print("Please enter a valid integer index")
            continue
        if i in out_idxs:
            return i
        print("Index is not an output-capable device; try again")


def set_default_output(index):
    # Keep input device unchanged if present, otherwise None
    current = sd.default.device
    in_idx = current[0] if isinstance(current, tuple) else None
    sd.default.device = (in_idx, index)


def play_test_tone(duration=1.0, samplerate=48000, frequency=440.0):
    t = np.linspace(0, duration, int(samplerate * duration), endpoint=False)
    wave = 0.2 * np.sin(2 * np.pi * frequency * t).astype(np.float32)
    sd.play(wave, samplerate)
    sd.wait()


def main(argv=None):
    parser = argparse.ArgumentParser(description="Audio output device selector and tester")
    parser.add_argument("--list", action="store_true", help="List devices and exit")
    parser.add_argument("--index", type=int, help="Set device by index")
    parser.add_argument("--name", type=str, help="Set device by name substring (case-insensitive)")
    parser.add_argument("--test-only", action="store_true", help="Only play tone on current default device")
    parser.add_argument("--tone-freq", type=float, default=440.0, help="Test tone frequency (Hz)")
    parser.add_argument("--tone-dur", type=float, default=1.0, help="Test tone duration (seconds)")
    parser.add_argument("--samplerate", type=int, default=48000, help="Samplerate for test tone")

    args = parser.parse_args(argv)

    try:
        devices = list_devices()
        if args.list and not (args.index is not None or args.name):
            return 0

        if args.test_only:
            print("Playing test tone on current default output device...")
            play_test_tone(duration=args.tone_dur, samplerate=args.samplerate, frequency=args.tone_freq)
            return 0

        sel = choose_device(devices, index=args.index, name_sub=args.name)
        if sel is None:
            print("Cancelled.")
            return 0

        set_default_output(sel)
        print(f"Using audio output device {sel}: {devices[sel]['name']}")
        print("Playing test tone...")
        play_test_tone(duration=args.tone_dur, samplerate=args.samplerate, frequency=args.tone_freq)
        print("Done.")
        return 0
    except Exception as e:
        print(f"Error: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
