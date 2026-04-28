import io
import os

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
}

MODE_LABELS = {
    "lofi": "Lo-fi 复古",
    "chipmunk": "花栗鼠变声",
    "nightcore": "Nightcore 加速",
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
    ax.plot(time_axis, samples, linewidth=0.7, color="#14b8a6")
    ax.set_xlabel("时间（秒）")
    ax.set_ylabel("振幅")
    ax.set_title("音频波形")
    ax.grid(alpha=0.18)
    fig.tight_layout()
    return fig


# --- 页面布局 ---
st.set_page_config(page_title="音频混音工作台", layout="centered")
st.markdown(
    """
    <style>
    .stApp {
        background: linear-gradient(180deg, #071a1f 0%, #0b1215 46%, #101417 100%);
        color: #edfdfb;
    }
    .block-container {
        padding-top: 2.3rem;
        padding-bottom: 3rem;
    }
    h1, h2, h3, label, .stMarkdown, .stMetric {
        color: #edfdfb !important;
    }
    div[data-testid="stExpander"] {
        background: rgba(12, 32, 37, 0.82);
        border: 1px solid rgba(45, 212, 191, 0.24);
        border-radius: 10px;
    }
    div[data-testid="stFileUploaderDropzone"] {
        background: rgba(15, 23, 42, 0.78);
        border-color: rgba(20, 184, 166, 0.42);
    }
    .stButton > button, .stDownloadButton > button {
        background: #14b8a6;
        color: #061417;
        border: 0;
        border-radius: 8px;
        font-weight: 700;
    }
    .stButton > button:hover, .stDownloadButton > button:hover {
        background: #2dd4bf;
        color: #061417;
        border: 0;
    }
    div[data-baseweb="select"] > div {
        background-color: rgba(15, 23, 42, 0.9);
        border-color: rgba(20, 184, 166, 0.36);
    }
    div[data-testid="stMetricValue"] {
        color: #5eead4;
    }
    </style>
    """,
    unsafe_allow_html=True,
)
st.title("音频混音工作台")
st.markdown("上传音乐，调整变调、速度、混响和 EQ，一键生成新的混音版本。")


# --- 应用预设的回调 ---
def apply_preset():
    remix_mode = st.session_state.remix_mode
    preset_key = remix_mode

    st.session_state.active_preset = preset_key
    preset_values = PRESETS.get(preset_key, {})

    for key, value in preset_values.items():
        st.session_state[key] = value


# --- 会话状态初始化 ---
if "initialized" not in st.session_state:
    st.session_state.initialized = True
    st.session_state.remix_mode = "lofi"
    apply_preset()

if st.session_state.get("remix_mode") not in MODE_LABELS:
    st.session_state.remix_mode = "lofi"
    apply_preset()

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

# --- 效果控制 ---
with st.expander("效果控制", expanded=True):
    st.slider("变调", -12, 12, key="pitch")
    st.slider("播放速度", 0.5, 2.0, key="speed", step=0.01)
    st.slider("混响强度", 0.0, 1.0, key="reverb", step=0.01)
    st.slider("低频增强（dB）", -12, 12, key="bass_boost")
    st.slider("高频削减（Hz，0 = 关闭）", 0, 12000, key="treble_cut", step=100)

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
