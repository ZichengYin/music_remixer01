import io
import os
import random

import librosa
import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
from pydub import AudioSegment

from remix_engine import remix_song

# --- 预设参数（单一来源） ---
PRESETS = {
    "lofi": {"pitch": -3, "speed": 0.92, "reverb": 0.5, "crackle_vol": 0.35, "ambient_vol": 0.45, "bass_boost": 4, "treble_cut": 8500},
    "chipmunk": {"pitch": 7, "speed": 1.15, "reverb": 0.25, "crackle_vol": 0.25, "ambient_vol": 0.3, "bass_boost": -2, "treble_cut": 0},
    "nightcore": {"pitch": 5, "speed": 1.3, "reverb": 0.3, "crackle_vol": 0.15, "ambient_vol": 0.25, "bass_boost": 1, "treble_cut": 11000},
    "Dreamy": {"pitch": -3, "speed": 0.8, "reverb": 0.55, "crackle_vol": 0.25, "ambient_vol": 0.55, "bass_boost": 3, "treble_cut": 7000},
    "Vintage": {"pitch": -1, "speed": 0.88, "reverb": 0.22, "crackle_vol": 0.45, "ambient_vol": 0.35, "bass_boost": 2, "treble_cut": 5500},
    "Glitchy": {"pitch": 1, "speed": 1.12, "reverb": 0.12, "crackle_vol": 0.12, "ambient_vol": 0.15, "bass_boost": -3, "treble_cut": 9500},
    "Hyperspeed": {"pitch": 6, "speed": 1.35, "reverb": 0.18, "crackle_vol": 0.1, "ambient_vol": 0.1, "bass_boost": -4, "treble_cut": 0},
    "Underwater": {"pitch": -2, "speed": 0.78, "reverb": 0.45, "crackle_vol": 0.25, "ambient_vol": 0.6, "bass_boost": 6, "treble_cut": 2500},
    "Radio": {"pitch": 0, "speed": 1.0, "reverb": 0.15, "crackle_vol": 0.38, "ambient_vol": 0.3, "bass_boost": -6, "treble_cut": 4200},
    "Alien": {"pitch": 12, "speed": 1.28, "reverb": 0.35, "crackle_vol": 0.2, "ambient_vol": 0.35, "bass_boost": -1, "treble_cut": 10000},
    "Spooky": {"pitch": -5, "speed": 0.83, "reverb": 0.65, "crackle_vol": 0.35, "ambient_vol": 0.4, "bass_boost": 5, "treble_cut": 3500},
}

MODE_LABELS = {
    "lofi": "Lo-fi 复古",
    "chipmunk": "花栗鼠变声",
    "nightcore": "Nightcore 加速",
    "themed": "主题预设",
}

THEME_KEYS = ["Dreamy", "Vintage", "Glitchy", "Hyperspeed", "Underwater", "Radio", "Alien", "Spooky"]
THEME_LABELS = {
    "Dreamy": "梦幻",
    "Vintage": "复古唱片",
    "Glitchy": "故障电音",
    "Hyperspeed": "极速",
    "Underwater": "水下",
    "Radio": "老式收音机",
    "Alien": "外星",
    "Spooky": "诡异",
}


def load_audio_for_analysis(audio_bytes):
    audio = AudioSegment.from_file(io.BytesIO(audio_bytes)).set_channels(1)
    samples = np.array(audio.get_array_of_samples()).astype(np.float32)
    samples /= float(1 << (8 * audio.sample_width - 1))
    return samples, audio.frame_rate


def detect_bpm(samples, sample_rate):
    tempo = librosa.beat.tempo(y=samples, sr=sample_rate)
    return float(np.atleast_1d(tempo)[0])


def plot_waveform(samples, sample_rate):
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "Arial Unicode MS", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False

    max_points = 20000
    if len(samples) > max_points:
        step = int(np.ceil(len(samples) / max_points))
        samples = samples[::step]

    time_axis = np.arange(len(samples)) / sample_rate
    fig, ax = plt.subplots(figsize=(8, 2.4))
    ax.plot(time_axis, samples, linewidth=0.7, color="#ff8c42")
    ax.set_xlabel("时间（秒）")
    ax.set_ylabel("振幅")
    ax.set_title("音频波形")
    ax.grid(alpha=0.18, color="#ffb561")
    fig.patch.set_facecolor("#2d1b11")
    ax.set_facecolor("#3d2618")
    ax.xaxis.label.set_color("#ffd2a5")
    ax.yaxis.label.set_color("#ffd2a5")
    ax.title.set_color("#ffb561")
    ax.tick_params(colors="#ffd2a5")
    fig.tight_layout()
    return fig


# --- 页面布局 ---
st.set_page_config(page_title="音频混音工作台", layout="centered")
st.markdown(
    """
    <style>
    .stApp {
        background: linear-gradient(180deg, #ff7b2c 0%, #d94b1a 35%, #8b2c0d 70%, #3d1406 100%);
        color: #fff0e0;
    }
    .block-container {
        padding-top: 2.3rem;
        padding-bottom: 3rem;
    }
    h1, h2, h3, label, .stMarkdown, .stMetric {
        color: #ffe6cc !important;
    }
    div[data-testid="stExpander"] {
        background: rgba(45, 25, 15, 0.85);
        border: 1px solid rgba(255, 140, 66, 0.5);
        border-radius: 10px;
    }
    div[data-testid="stFileUploaderDropzone"] {
        background: rgba(60, 35, 25, 0.8);
        border-color: rgba(255, 140, 66, 0.6);
    }
    .stButton > button, .stDownloadButton > button {
        background: #ff7b2c;
        color: #2d1406;
        border: 0;
        border-radius: 8px;
        font-weight: 700;
    }
    .stButton > button:hover, .stDownloadButton > button:hover {
        background: #ff9f5a;
        color: #2d1406;
        border: 0;
    }
    div[data-baseweb="select"] > div {
        background-color: rgba(60, 35, 25, 0.9);
        border-color: rgba(255, 140, 66, 0.5);
    }
    div[data-testid="stMetricValue"] {
        color: #ffbc7a;
    }
    .stSlider > div > div > div {
        background-color: #ff7b2c;
    }
    .stCheckbox > label {
        color: #ffe6cc !important;
    }
    .stAlert {
        background-color: #5c2c1a;
        color: #ffddb0;
    }
    hr {
        border-color: #ff8c42;
    }
    </style>
    """,
    unsafe_allow_html=True,
)
st.title("🌅 音频混音工作台")
st.markdown("上传音乐，调整变调、速度、混响和 EQ，一键生成新的混音版本。")


# --- 应用预设的回调 ---
def apply_preset():
    remix_mode = st.session_state.remix_mode
    preset_key = remix_mode

    if remix_mode == "themed":
        if st.session_state.get("surprise_me", False):
            theme_options = THEME_KEYS.copy()
            previous_preset = st.session_state.get("active_preset")
            if len(theme_options) > 1 and previous_preset in theme_options:
                theme_options.remove(previous_preset)
            preset_key = random.choice(theme_options)
            st.session_state.theme = preset_key
        else:
            preset_key = st.session_state.get("theme", "Dreamy")

    st.session_state.active_preset = preset_key
    preset_values = PRESETS.get(preset_key, {})

    for key, value in preset_values.items():
        st.session_state[key] = value


# --- 会话状态初始化 ---
if "initialized" not in st.session_state:
    st.session_state.initialized = True
    st.session_state.remix_mode = "lofi"
    st.session_state.theme = "Dreamy"
    st.session_state.surprise_me = False
    apply_preset()

if st.session_state.get("remix_mode") not in MODE_LABELS:
    st.session_state.remix_mode = "lofi"
    apply_preset()

if st.session_state.get("theme") not in THEME_KEYS:
    st.session_state.theme = "Dreamy"

# --- 文件上传 ---
with st.expander("上传音频文件", expanded=True):
    main_audio = st.file_uploader("主音轨（MP3）", type=["mp3"])
    col1, col2 = st.columns(2)
    with col1:
        crackle_audio = st.file_uploader("黑胶噪声（可选）", type=["mp3", "wav"])
    with col2:
        ambient_audio = st.file_uploader("氛围背景音（可选）", type=["mp3", "wav"])
    st.caption("提示：可以在 [freesound.org](https://freesound.org) 搜索黑胶噪声或氛围音素材。")

main_audio_bytes = main_audio.getvalue() if main_audio else None

if main_audio_bytes:
    with st.expander("音频分析", expanded=True):
        try:
            analysis_samples, analysis_sr = load_audio_for_analysis(main_audio_bytes)
            bpm = detect_bpm(analysis_samples, analysis_sr)
            st.metric("检测到的 BPM", f"{bpm:.1f}")
            fig = plot_waveform(analysis_samples, analysis_sr)
            st.pyplot(fig)
            plt.close(fig)
        except Exception as e:
            st.warning(f"无法分析音频：{str(e)}")

# --- 模式选择 ---
st.markdown("## 混音设置")
remix_mode = st.selectbox(
    "选择混音模式",
    list(MODE_LABELS.keys()),
    format_func=lambda mode: MODE_LABELS[mode],
    key="remix_mode",
    on_change=apply_preset,
)

is_themed_mode = remix_mode == "themed"
if is_themed_mode:
    theme_col1, theme_col2 = st.columns([2, 1])
    with theme_col1:
        st.selectbox(
            "选择主题",
            THEME_KEYS,
            format_func=lambda theme: THEME_LABELS[theme],
            key="theme",
            on_change=apply_preset,
        )
    with theme_col2:
        st.checkbox("随机主题", key="surprise_me", on_change=apply_preset, help="随机选择一个主题预设。")

# --- 效果控制 ---
with st.expander("效果控制", expanded=True):
    st.slider("变调", -12, 12, key="pitch", disabled=is_themed_mode)
    st.slider("播放速度", 0.5, 2.0, key="speed", step=0.01, disabled=is_themed_mode)
    st.slider("混响强度", 0.0, 1.0, key="reverb", step=0.01, disabled=is_themed_mode)
    st.slider("低频增强（dB）", -12, 12, key="bass_boost", disabled=is_themed_mode)
    st.slider("高频削减（Hz，0 = 关闭）", 0, 12000, key="treble_cut", step=100, disabled=is_themed_mode)
    if is_themed_mode:
        active_theme_label = THEME_LABELS.get(st.session_state.active_preset, st.session_state.active_preset)
        st.info(f"当前主题：{active_theme_label}。变调、速度、混响和 EQ 由主题预设控制。")

# --- 背景音量控制 ---
if crackle_audio or ambient_audio:
    with st.expander("背景音混合", expanded=True):
        if crackle_audio:
            st.slider("黑胶噪声音量", 0.0, 1.0, key="crackle_vol", step=0.01)
        else:
            st.session_state.crackle_vol = 0.0
        if ambient_audio:
            st.slider("氛围背景音量", 0.0, 1.0, key="ambient_vol", step=0.01)
        else:
            st.session_state.ambient_vol = 0.0
else:
    st.session_state.crackle_vol = 0.0
    st.session_state.ambient_vol = 0.0

# --- 处理按钮 ---
st.markdown("---")
if main_audio:
    if st.button("开始处理音频"):
        with st.spinner("正在生成混音..."):
            try:
                output_path = remix_song(
                    song_bytes=main_audio_bytes,
                    crackle_bytes=crackle_audio.read() if crackle_audio else None,
                    ambient_bytes=ambient_audio.read() if ambient_audio else None,
                    remix_mode=st.session_state.remix_mode,
                    pitch=st.session_state.pitch,
                    speed=st.session_state.speed,
                    reverb_wet=st.session_state.reverb,
                    bass_boost=st.session_state.bass_boost,
                    treble_cut=st.session_state.treble_cut,
                    crackle_vol=st.session_state.crackle_vol,
                    ambient_vol=st.session_state.ambient_vol,
                    theme=st.session_state.active_preset,
                )

                with open(output_path, "rb") as f:
                    audio_bytes = f.read()
                    st.success("混音完成！可以试听或下载：")
                    st.audio(audio_bytes, format="audio/mp3")
                    st.download_button("下载混音文件", audio_bytes, "remixed_track.mp3", "audio/mp3")
                os.unlink(output_path)
            except Exception as e:
                st.error(f"处理失败：{str(e)}")
                st.exception(e)
else:
    st.warning("请先上传主音轨。")
