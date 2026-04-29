import io
import tempfile
import random
import librosa
import numpy as np
from pydub import AudioSegment
from pedalboard import (
    Pedalboard, Reverb, PitchShift, LowpassFilter, HighpassFilter,
    Distortion, Delay, Chorus, Phaser, Compressor
)

def ensure_format(audio: AudioSegment) -> AudioSegment:
    return audio.set_channels(2).set_frame_rate(44100).set_sample_width(2)

def get_theme_effects(theme):
    board = Pedalboard()
    speed = 1.0

    if theme == "Dreamy":
        board.append(Reverb(room_size=0.8, wet_level=0.5))
        board.append(PitchShift(-2))
        board.append(LowpassFilter(8000))
        speed = 0.85
    elif theme == "Vintage":
        board.append(Reverb(room_size=0.3, wet_level=0.2))
        board.append(Distortion(drive_db=10))
        board.append(LowpassFilter(6000))
        board.append(HighpassFilter(150))
        speed = 0.9
    elif theme == "Glitchy":
        board.append(Phaser(rate_hz=2.0))
        board.append(Chorus())
        board.append(Distortion(drive_db=8))
        speed = 1.1
    elif theme == "Hyperspeed":
        board.append(Chorus())
        board.append(PitchShift(5))
        board.append(Delay(delay_seconds=0.1))
        speed = 1.3
    elif theme == "Underwater":
        board.append(LowpassFilter(2000))
        board.append(Reverb(room_size=0.9, wet_level=0.4))
        board.append(Compressor())
        speed = 0.8
    elif theme == "Radio":
        board.append(HighpassFilter(400))
        board.append(LowpassFilter(4000))
        board.append(Distortion(drive_db=12))
        board.append(Delay(delay_seconds=0.05))
    elif theme == "Alien":
        board.append(PitchShift(random.choice([-12, 12])))
        board.append(Chorus())
        speed = random.choice([0.75, 1.25])
    elif theme == "Spooky":
        board.append(Reverb(room_size=0.9, wet_level=0.6))
        board.append(PitchShift(-4))
        board.append(Delay(delay_seconds=0.3))
        board.append(LowpassFilter(3000))
        speed = 0.85

    return board, speed

def audiosegment_to_numpy(audio: AudioSegment):
    samples = np.array(audio.get_array_of_samples()).astype(np.float32) / 32768.0
    if audio.channels == 2:
        samples = samples.reshape((-1, 2)).T
    else:
        samples = samples.reshape((1, -1))
    return samples

def ensure_channel_first(samples: np.ndarray) -> np.ndarray:
    if samples.ndim == 2 and samples.shape[0] > samples.shape[1] and samples.shape[1] <= 2:
        return samples.T
    return samples

def numpy_to_audiosegment(samples: np.ndarray, sample_rate=44100):
    samples = ensure_channel_first(samples)

    samples = (samples * 32768).clip(-32768, 32767).astype(np.int16)
    if samples.ndim == 1:
        channels = 1
        interleaved = samples
    else:
        channels = samples.shape[0]
        interleaved = samples.T.flatten()
    return AudioSegment(
        interleaved.tobytes(), frame_rate=sample_rate, sample_width=2, channels=channels
    )

def change_speed(audio: AudioSegment, speed: float) -> AudioSegment:
    if abs(speed - 1.0) <= 0.01:
        return audio

    samples = audiosegment_to_numpy(audio)
    stretched_channels = [
        librosa.effects.time_stretch(channel.astype(np.float32), rate=float(speed))
        for channel in samples
    ]
    min_length = min(len(channel) for channel in stretched_channels)
    stretched = np.vstack([channel[:min_length] for channel in stretched_channels])
    return numpy_to_audiosegment(stretched, sample_rate=audio.frame_rate)

def apply_eq(samples: np.ndarray, sample_rate: int, bass_boost=0, treble_cut=0):
    eq_samples = ensure_channel_first(samples)

    if abs(bass_boost) > 0.01:
        crossover_hz = 250
        bass = Pedalboard([LowpassFilter(crossover_hz)])(eq_samples, sample_rate)
        upper = Pedalboard([HighpassFilter(crossover_hz)])(eq_samples, sample_rate)
        bass_gain = 10 ** (bass_boost / 20.0)
        eq_samples = ensure_channel_first(upper + (bass * bass_gain))

    if treble_cut and treble_cut > 0:
        cutoff_hz = min(float(treble_cut), (sample_rate / 2) - 100)
        if cutoff_hz > 20:
            eq_samples = ensure_channel_first(Pedalboard([LowpassFilter(cutoff_hz)])(eq_samples, sample_rate))

    return eq_samples

def remix_song(song_bytes, crackle_bytes=None, ambient_bytes=None,
               remix_mode="lofi", theme=None, pitch=0, speed=1.0,
               reverb_wet=0.3, bass_boost=0, treble_cut=0,
               crackle_vol=0.1, ambient_vol=0.2,
               preview_seconds=0, progress_callback=None):
    """
    preview_seconds: 如果 > 0，只处理前 N 秒（用于试听）
    progress_callback: 函数，接收 (stage, percent) 参数
    """
    
    def report(stage, percent):
        if progress_callback:
            progress_callback(stage, percent)

    report("加载音频", 0)
    song = ensure_format(AudioSegment.from_file(io.BytesIO(song_bytes)))
    
    # 试听模式：裁剪
    if preview_seconds > 0:
        preview_ms = min(int(preview_seconds * max(speed, 0.01) * 1000), len(song))
        song = song[:preview_ms]
    
    def prepare_bg(bg_bytes, volume):
        if not bg_bytes or volume <= 0:
            return AudioSegment.silent(duration=len(song)).set_channels(2)
        bg = ensure_format(AudioSegment.from_file(io.BytesIO(bg_bytes)))
        gain = -30 * (1.0 - volume)
        bg = bg + gain
        return (bg * ((len(song) // len(bg)) + 1))[:len(song)]

    report("加载背景音", 10)
    board = Pedalboard()

    if remix_mode == "themed":
        board, _ = get_theme_effects(theme)
    elif remix_mode == "chipmunk":
        board.append(PitchShift(pitch))
    elif remix_mode == "nightcore":
        board.append(PitchShift(pitch))
    else:
        board.append(PitchShift(pitch))
        board.append(Reverb(wet_level=reverb_wet))

    report("应用速度变化", 30)
    song = change_speed(song, speed)

    if preview_seconds > 0:
        song = song[:preview_seconds * 1000]

    crackle = prepare_bg(crackle_bytes, crackle_vol)
    ambient = prepare_bg(ambient_bytes, ambient_vol)

    report("应用音频效果", 50)
    samples = audiosegment_to_numpy(song)
    processed = ensure_channel_first(board(samples, song.frame_rate))
    
    report("应用 EQ", 70)
    processed = apply_eq(processed, song.frame_rate, bass_boost=bass_boost, treble_cut=treble_cut)
    song = numpy_to_audiosegment(processed, sample_rate=song.frame_rate)

    report("混合背景音", 85)
    final_mix = song.overlay(crackle).overlay(ambient)
    
    report("音量标准化", 95)
    final_mix = final_mix.normalize()

    report("导出文件", 100)
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as f:
        final_mix.export(f.name, format="mp3")
        return f.name
