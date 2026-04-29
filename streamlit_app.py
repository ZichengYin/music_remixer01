import io
import os
import random
import json

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

# 用户自定义预设存储
if "user_presets" not in st.session_state:
    st.session_state.user_presets = {}

MODE_LABELS = {
    "lofi": "Lo-fi / 低保真",
    "chipmunk": "Chipmunk / 松鼠音",
    "nightcore": "Nightcore / 夜店",
    "themed": "Themed / 主题",
}

THEME_KEYS = ["Dreamy", "Vintage", "Glitchy", "Hyperspeed", "Underwater", "Radio", "Alien", "Spooky"]
THEME_LABELS = {
    "Dreamy": "Dreamy / 梦幻",
    "Vintage": "Vintage / 复古",
    "Glitchy": "Glitchy / 故障",
    "Hyperspeed": "Hyperspeed / 极速",
    "Underwater": "Underwater / 水下",
    "Radio": "Radio / 收音机",
    "Alien": "Alien / 外星",
    "Spooky": "Spooky / 诡异",
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
    ax.plot(time_axis, samples, linewidth=0.7, color="#d95b0e")  # 深橙色线条
    ax.set_xlabel("时间 / Time (s)")
    ax.set_ylabel("振幅 / Amplitude")
    ax.set_title("波形 / Waveform")
    ax.grid(alpha=0.18, color="#d95b0e")
    fig.patch.set_facecolor("#fef7e8")  # 浅米色背景
    ax.set_facecolor("#fef0dc")         # 更浅的米色
    ax.xaxis.label.set_color("#4a2e1a")  # 深棕色文字
    ax.yaxis.label.set_color("#4a2e1a")
    ax.title.set_color("#4a2e1a")
    ax.tick_params(colors="#4a2e1a")
    fig.tight_layout()
    return fig


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


def reset_to_default():
    preset_key = st.session_state.active_preset
    preset_values = PRESETS.get(preset_key, {})
    for key, value in preset_values.items():
        st.session_state[key] = value


def save_current_preset():
    name = st.session_state.new_preset_name.strip()
    if not name:
        st.warning("请输入预设名称 / Please enter a preset name")
        return
    current_params = {
        "pitch": st.session_state.pitch,
        "speed": st.session_state.speed,
        "reverb": st.session_state.reverb,
        "bass_boost": st.session_state.bass_boost,
        "treble_cut": st.session_state.treble_cut,
        "crackle_vol": st.session_state.crackle_vol,
        "ambient_vol": st.session_state.ambient_vol,
    }
    st.session_state.user_presets[name] = current_params
    st.success(f"已保存预设 / Saved: {name}")


def load_user_preset():
    preset_name = st.session_state.load_preset_name
    if preset_name and preset_name in st.session_state.user_presets:
        params = st.session_state.user_presets[preset_name]
        for key, value in params.items():
            st.session_state[key] = value
        st.success(f"已加载预设 / Loaded: {preset_name}")
    elif preset_name:
        st.warning(f"预设不存在 / Preset not found: {preset_name}")


# --- 页面配置 ---
st.set_page_config(page_title="Remix Lab", layout="centered")
st.markdown(
    """
    <style>
    /* 整体浅色背景（米色暖调） */
    .stApp {
        background: linear-gradient(160deg, #fff8df 0%, #ffe9cf 36%, #ffd7c2 68%, #f8c9d4 100%);
    }
    .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }

    /* 所有文字颜色变为深褐色，提高对比度 */
    /* 覆盖 Streamlit 默认类（深色文字） */
    /* Expander 面板浅色半透明，深色边框 */
    div[data-testid="stExpander"] {
        background: rgba(255, 245, 225, 0.85) !important;
        border: 1px solid rgba(150, 80, 30, 0.4) !important;
        border-radius: 12px;
        backdrop-filter: blur(2px);
    }
    div[data-testid="stExpander"] summary {
        background: rgba(0,0,0,0) !important;
        color: #8b3c0c !important;
    }

    /* 上传区域浅色背景 + 深色文字/边框 */
    div[data-testid="stFileUploaderDropzone"],
    div[data-testid="stFileUploaderDropzone"] > div,
    div[data-testid="stFileUploaderDropzone"] div {
        background: #ffe6cc !important;
        border-color: #d97a2b !important;
        border-radius: 10px !important;
    }

    /* 按钮：暖橙色底 + 深色字，悬浮稍亮 */
    .stButton > button, .stDownloadButton > button {
        background: #e8772e !important;
        color: #2c1a0e !important;
        border: none !important;
        border-radius: 8px !important;
        font-weight: 600 !important;
    }
    .stButton > button:hover, .stDownloadButton > button:hover {
        background: #f59142 !important;
        color: #1f1107 !important;
    }

    /* 下拉框、数字输入、滑块背景浅色，深色文字 */
    div[data-baseweb="select"] > div,
    div[data-testid="stSelectbox"] div,
    div[data-testid="stNumberInput"] input,
    div[data-testid="stSlider"] > div {
        background-color: #fff6e8 !important;
        border-color: #d97a2b !important;
        color: #2c1a0e !important;
    }

    /* 滑块轨道颜色加深 */
    div[data-testid="stSlider"] div[data-baseweb="slider"] {
        background-color: #d9b48b !important;
    }
    /* 滑块圆点 */
    div[data-testid="stSlider"] div[role="slider"] {
        background-color: #e8772e !important;
        border-color: #b2500a !important;
    }

    /* Metric 数值颜色 */
    div[data-testid="stMetricValue"] {
        color: #b2500a !important;
        font-weight: 500;
    }

    /* Checkbox 文字颜色 */
    .stCheckbox label span {
        color: #2c1a0e !important;
    }

    /* 滑块标签数值颜色 */
    .stSlider label, .stSlider output {
        color: #4a2e1a !important;
    }

    /* 提示框、警告框等色调统一 */
    .stAlert {
        background-color: #fff0e0 !important;
        color: #2c1a0e !important;
        border-left: 4px solid #e8772e !important;
    }
    hr {
        border-color: #d9b48b !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)
st.title("🌅 Remix Lab / 音频混音工作台")
st.markdown("Upload a track, adjust effects, and generate a remixed version. / 上传音乐，调整效果，生成混音版本。")

# --- 会话状态初始化 ---
if "initialized" not in st.session_state:
    st.session_state.initialized = True
    st.session_state.remix_mode = "lofi"
    st.session_state.theme = "Dreamy"
    st.session_state.surprise_me = False
    st.session_state.active_preset = "lofi"
    st.session_state.preview_mode = False
    apply_preset()

if st.session_state.get("remix_mode") not in MODE_LABELS:
    st.session_state.remix_mode = "lofi"
    apply_preset()

if st.session_state.get("theme") not in THEME_KEYS:
    st.session_state.theme = "Dreamy"

# --- 文件上传 ---
with st.expander("Upload / 上传音频文件", expanded=True):
    main_audio = st.file_uploader("Main track (MP3) / 主音轨", type=["mp3"])
    col1, col2 = st.columns(2)
    with col1:
        crackle_audio = st.file_uploader("Vinyl crackle (optional) / 黑胶噪声", type=["mp3", "wav"])
    with col2:
        ambient_audio = st.file_uploader("Ambient background (optional) / 氛围背景音", type=["mp3", "wav"])

main_audio_bytes = main_audio.getvalue() if main_audio else None

if main_audio_bytes:
    with st.expander("Audio Analysis / 音频分析", expanded=True):
        try:
            analysis_samples, analysis_sr = load_audio_for_analysis(main_audio_bytes)
            bpm = detect_bpm(analysis_samples, analysis_sr)
            st.metric("BPM", f"{bpm:.1f}")
            fig = plot_waveform(analysis_samples, analysis_sr)
            st.pyplot(fig)
            plt.close(fig)
        except Exception as e:
            st.warning(f"无法分析音频 / Analysis failed: {str(e)}")

# --- 模式选择 ---
st.markdown("## Mix Settings / 混音设置")

col_mode, col_output = st.columns([3, 1])
with col_mode:
    remix_mode = st.selectbox(
        "Mode / 模式",
        list(MODE_LABELS.keys()),
        format_func=lambda mode: MODE_LABELS[mode],
        key="remix_mode",
        on_change=apply_preset,
    )

with col_output:
    output_format = st.selectbox("Output / 输出格式", ["mp3", "wav"], index=0)

is_themed_mode = remix_mode == "themed"
if is_themed_mode:
    theme_col1, theme_col2 = st.columns([2, 1])
    with theme_col1:
        st.selectbox(
            "Theme / 主题",
            THEME_KEYS,
            format_func=lambda theme: THEME_LABELS[theme],
            key="theme",
            on_change=apply_preset,
        )
    with theme_col2:
        st.checkbox("Random / 随机主题", key="surprise_me", on_change=apply_preset)

# --- 效果控制 ---
with st.expander("Effects / 效果控制", expanded=True):
    st.slider("Pitch shift / 变调", -12, 12, key="pitch", disabled=is_themed_mode)
    st.slider("Speed / 速度", 0.5, 2.0, key="speed", step=0.01, disabled=is_themed_mode)
    st.slider("Reverb / 混响", 0.0, 1.0, key="reverb", step=0.01, disabled=is_themed_mode)
    st.slider("Bass boost (dB) / 低频增强", -12, 12, key="bass_boost", disabled=is_themed_mode)
    st.slider("Treble cut (Hz, 0 = off) / 高频削减", 0, 12000, key="treble_cut", step=100, disabled=is_themed_mode)
    
    if st.button("Reset to preset / 重置到预设值"):
        reset_to_default()
        st.rerun()
    
    if is_themed_mode:
        active_theme_label = THEME_LABELS.get(st.session_state.active_preset, st.session_state.active_preset)
        st.info(f"Active theme / 当前主题：{active_theme_label}")

# --- 背景音量控制 ---
if crackle_audio or ambient_audio:
    with st.expander("Background mixing / 背景音混合", expanded=True):
        if crackle_audio:
            st.slider("Crackle volume / 噪声音量", 0.0, 1.0, key="crackle_vol", step=0.01)
        else:
            st.session_state.crackle_vol = 0.0
        if ambient_audio:
            st.slider("Ambient volume / 背景音量", 0.0, 1.0, key="ambient_vol", step=0.01)
        else:
            st.session_state.ambient_vol = 0.0
else:
    st.session_state.crackle_vol = 0.0
    st.session_state.ambient_vol = 0.0

# --- 用户预设管理 ---
with st.expander("Save/Load custom presets / 保存/加载自定义预设"):
    col_save, col_load = st.columns(2)
    with col_save:
        st.text_input("Preset name / 预设名称", key="new_preset_name", placeholder="e.g. My Bass Boost")
        st.button("Save current / 保存当前设置", on_click=save_current_preset)
    with col_load:
        preset_options = list(st.session_state.user_presets.keys())
        if preset_options:
            st.selectbox("Load preset / 加载预设", [""] + preset_options, key="load_preset_name", on_change=load_user_preset)
        else:
            st.caption("No saved presets / 暂无保存的预设")

# --- 处理选项 ---
st.markdown("---")
col_preview, col_full = st.columns(2)

with col_preview:
    preview_button = st.button("Preview 10s / 试听前10秒")
with col_full:
    full_button = st.button("Full remix / 完整处理")

progress_placeholder = st.empty()

def update_progress(stage, percent):
    with progress_placeholder.container():
        st.progress(percent / 100)
        st.caption(f"{stage} ({percent}%)")

if main_audio:
    is_preview = False
    if preview_button:
        is_preview = True
        preview_seconds = 10
    elif full_button:
        is_preview = False
        preview_seconds = 0
    else:
        preview_seconds = None
    
    if preview_seconds is not None:
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
                preview_seconds=preview_seconds,
                progress_callback=update_progress,
            )

            with open(output_path, "rb") as f:
                audio_bytes = f.read()
            
            progress_placeholder.empty()
            
            if is_preview:
                st.success("Preview ready / 试听生成完成")
                st.audio(audio_bytes, format=f"audio/{output_format}")
                st.caption("Preview: first 10 seconds only. / 仅前10秒试听")
            else:
                st.success("Remix complete / 混音完成")
                st.audio(audio_bytes, format=f"audio/{output_format}")
                st.download_button(
                    "Download / 下载", 
                    audio_bytes, 
                    f"remixed_track.{output_format}", 
                    f"audio/{output_format}"
                )
            
            os.unlink(output_path)
        except Exception as e:
            progress_placeholder.empty()
            st.error(f"Failed / 处理失败：{str(e)}")
            st.exception(e)
else:
    st.warning("Please upload a main track first. / 请先上传主音轨。")
