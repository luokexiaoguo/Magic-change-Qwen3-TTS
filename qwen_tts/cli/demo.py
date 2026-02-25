#!/usr/bin/env python3
# Copyright (c) Alibaba Cloud.
# SPDX-License-Identifier: Apache-2.0

import argparse
import base64
import io
import os
import re
import sys
import tempfile
import warnings
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from datetime import datetime

import gradio as gr
import numpy as np
import torch
import soundfile as sf

# Import Qwen3TTSModel
from qwen_tts.inference.qwen3_tts_model import Qwen3TTSModel

# Suppress common warnings for cleaner UI
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", message=".*Torch was not compiled with flash attention.*")


@dataclass
class VoiceClonePromptItem:
    ref_code: Optional[torch.Tensor]
    ref_spk_embedding: torch.Tensor
    x_vector_only_mode: bool
    icl_mode: bool
    ref_text: Optional[str] = None


# Language mapping for UI
LANGUAGE_MAP = {
    "自动检测": "auto",
    "中文": "chinese",
    "英语": "english",
    "日语": "japanese",
    "韩语": "korean",
    "法语": "french",
    "德语": "german",
    "意大利语": "italian",
    "葡萄牙语": "portuguese",
    "西班牙语": "spanish",
    "俄语": "russian"
}
LANGUAGE_CHOICES = list(LANGUAGE_MAP.keys())

# Speaker mapping for CustomVoice
SPEAKER_MAP = {
    "薇薇安 (vivian)": "vivian",
    "塞雷娜 (serena)": "serena",
    "埃里克 (eric)": "eric",
    "艾登 (aiden)": "aiden",
    "迪伦 (dylan)": "dylan",
    "瑞安 (ryan)": "ryan",
    "苏熙 (sohee)": "sohee",
    "小野安娜 (ono_anna)": "ono_anna",
    "傅叔 (uncle_fu)": "uncle_fu"
}
SPEAKER_CHOICES = list(SPEAKER_MAP.keys())

# Speaker descriptions for UI
SPEAKER_DESCRIPTIONS = {
    "薇薇安 (vivian)": """
        <div class='spk-desc-animate' style='background: rgba(99, 102, 241, 0.1); padding: 12px; border-radius: 12px; border-left: 4px solid #6366f1; margin-top: 10px;'>
            <h4 style='margin: 0 0 5px 0; color: #6366f1;'>🎤 薇薇安 (Vivian)</h4>
            <p style='margin: 0; font-size: 0.9rem;'><b>特点</b>：明亮且略带磁性的年轻女声。</p>
            <p style='margin: 3px 0; font-size: 0.9rem;'><b>适用场景</b>：时尚解说、元气广播、短视频配音。</p>
            <p style='margin: 0; font-size: 0.9rem; opacity: 0.8;'><b>音色特征</b>：音质清脆，充满活力与现代感。</p>
        </div>
    """,
    "塞雷娜 (serena)": """
        <div class='spk-desc-animate' style='background: rgba(168, 85, 247, 0.1); padding: 12px; border-radius: 12px; border-left: 4px solid #a855f7; margin-top: 10px;'>
            <h4 style='margin: 0 0 5px 0; color: #a855f7;'>🎤 塞雷娜 (Serena)</h4>
            <p style='margin: 0; font-size: 0.9rem;'><b>特点</b>：温暖、柔和且极具亲和力的年轻女声。</p>
            <p style='margin: 3px 0; font-size: 0.9rem;'><b>适用场景</b>：情感电台、治愈系故事、温柔导购。</p>
            <p style='margin: 0; font-size: 0.9rem; opacity: 0.8;'><b>音色特征</b>：语调平缓，听感舒适顺滑。</p>
        </div>
    """,
    "埃里克 (eric)": """
        <div class='spk-desc-animate' style='background: rgba(234, 179, 8, 0.1); padding: 12px; border-radius: 12px; border-left: 4px solid #eab308; margin-top: 10px;'>
            <h4 style='margin: 0 0 5px 0; color: #eab308;'>🎤 埃里克 (Eric)</h4>
            <p style='margin: 0; font-size: 0.9rem;'><b>特点</b>：活泼的成都男声，略带沙哑的明亮感。</p>
            <p style='margin: 3px 0; font-size: 0.9rem;'><b>适用场景</b>：四川方言短视频、生活化对白、特色配音。</p>
            <p style='margin: 0; font-size: 0.9rem; opacity: 0.8;'><b>音色特征</b>：川味韵味浓厚，风趣幽默，辨识度高。</p>
        </div>
    """,
    "艾登 (aiden)": """
        <div class='spk-desc-animate' style='background: rgba(34, 197, 94, 0.1); padding: 12px; border-radius: 12px; border-left: 4px solid #22c55e; margin-top: 10px;'>
            <h4 style='margin: 0 0 5px 0; color: #22c55e;'>🎤 艾登 (Aiden)</h4>
            <p style='margin: 0; font-size: 0.9rem;'><b>特点</b>：阳光开朗的美国男声，中音清晰通透。</p>
            <p style='margin: 3px 0; font-size: 0.9rem;'><b>适用场景</b>：美式英语学习、旅游攻略、运动品牌旁白。</p>
            <p style='margin: 0; font-size: 0.9rem; opacity: 0.8;'><b>音色特征</b>：发音地道，语速自然，充满朝气。</p>
        </div>
    """,
    "迪伦 (dylan)": """
        <div class='spk-desc-animate' style='background: rgba(59, 130, 246, 0.1); padding: 12px; border-radius: 12px; border-left: 4px solid #3b82f6; margin-top: 10px;'>
            <h4 style='margin: 0 0 5px 0; color: #3b82f6;'>🎤 迪伦 (Dylan)</h4>
            <p style='margin: 0; font-size: 0.9rem;'><b>特点</b>：清脆自然、字正腔圆的北京少年男声。</p>
            <p style='margin: 3px 0; font-size: 0.9rem;'><b>适用场景</b>：校园广播、科普教育、充满活力的解说。</p>
            <p style='margin: 0; font-size: 0.9rem; opacity: 0.8;'><b>音色特征</b>：京腔余韵，咬字清晰，充满少年感。</p>
        </div>
    """,
    "瑞安 (ryan)": """
        <div class='spk-desc-animate' style='background: rgba(239, 68, 68, 0.1); padding: 12px; border-radius: 12px; border-left: 4px solid #ef4444; margin-top: 10px;'>
            <h4 style='margin: 0 0 5px 0; color: #ef4444;'>🎤 瑞安 (Ryan)</h4>
            <p style='margin: 0; font-size: 0.9rem;'><b>特点</b>：富有动感、节奏感极强的磁性男声。</p>
            <p style='margin: 3px 0; font-size: 0.9rem;'><b>适用场景</b>：运动赛事解说、动感广告、激昂演说。</p>
            <p style='margin: 0; font-size: 0.9rem; opacity: 0.8;'><b>音色特征</b>：爆发力强，充满激情与力量感。</p>
        </div>
    """,
    "苏熙 (sohee)": """
        <div class='spk-desc-animate' style='background: rgba(236, 72, 153, 0.1); padding: 12px; border-radius: 12px; border-left: 4px solid #ec4899; margin-top: 10px;'>
            <h4 style='margin: 0 0 5px 0; color: #ec4899;'>🎤 苏熙 (Sohee)</h4>
            <p style='margin: 0; font-size: 0.9rem;'><b>特点</b>：温暖、细腻且富有情感深度的韩语女声。</p>
            <p style='margin: 3px 0; font-size: 0.9rem;'><b>适用场景</b>：韩语教学、影视剧配音、深情独白。</p>
            <p style='margin: 0; font-size: 0.9rem; opacity: 0.8;'><b>音色特征</b>：感情充沛，能够精准表达细腻情绪。</p>
        </div>
    """,
    "小野安娜 (ono_anna)": """
        <div class='spk-desc-animate' style='background: rgba(20, 184, 166, 0.1); padding: 12px; border-radius: 12px; border-left: 4px solid #14b8a6; margin-top: 10px;'>
            <h4 style='margin: 0 0 5px 0; color: #14b8a6;'>🎤 小野安娜 (Ono_Anna)</h4>
            <p style='margin: 0; font-size: 0.9rem;'><b>特点</b>：俏皮可爱、音色轻盈灵动的日语女声。</p>
            <p style='margin: 3px 0; font-size: 0.9rem;'><b>适用场景</b>：动漫配音、二次元视频、轻快生活分享。</p>
            <p style='margin: 0; font-size: 0.9rem; opacity: 0.8;'><b>音色特征</b>：语气俏皮，极具辨识度，元气十足。</p>
        </div>
    """,
    "傅叔 (uncle_fu)": """
        <div class='spk-desc-animate' style='background: rgba(120, 113, 108, 0.1); padding: 12px; border-radius: 12px; border-left: 4px solid #78716c; margin-top: 10px;'>
            <h4 style='margin: 0 0 5px 0; color: #78716c;'>🎤 傅叔 (Uncle_Fu)</h4>
            <p style='margin: 0; font-size: 0.9rem;'><b>特点</b>：沉稳厚重、音色圆润的老年男声。</p>
            <p style='margin: 3px 0; font-size: 0.9rem;'><b>适用场景</b>：纪录片旁白、讲座故事、成熟稳重的长辈角色。</p>
            <p style='margin: 0; font-size: 0.9rem; opacity: 0.8;'><b>音色特征</b>：语速缓慢，充满智慧感与岁月积淀。</p>
        </div>
    """
}

class ModelManager:
    """Unified model manager for dynamic loading and switching"""
    def __init__(self, models_dir: str, device: str, dtype: torch.dtype, attn_impl: Optional[str]):
        self.models_dir = models_dir
        self.device = device
        self.dtype = dtype
        self.attn_impl = attn_impl
        self.model = None
        self.kind = None
        
        # Path configuration
        self.paths = {
            "custom_voice": os.path.join(models_dir, "Qwen3-TTS-12Hz-1.7B-CustomVoice"),
            "voice_design": os.path.join(models_dir, "Qwen3-TTS-12Hz-1.7B-VoiceDesign"),
            "voice_clone": os.path.join(models_dir, "Qwen3-TTS-12Hz-1.7B-Base")
        }

    def load(self, kind: str) -> Qwen3TTSModel:
        if self.kind == kind and self.model is not None:
            return self.model
            
        print(f"\n[ModelManager] Switching to {kind.upper()} mode...")
        
        # Unload previous model to free VRAM
        if self.model is not None:
            del self.model
            torch.cuda.empty_cache()
            import gc
            gc.collect()
            
        target_path = self.paths.get(kind)
        if not target_path or not os.path.exists(target_path):
            raise FileNotFoundError(f"Model path not found: {target_path}")
            
        self.model = Qwen3TTSModel.from_pretrained(
            target_path,
            device_map=self.device,
            dtype=self.dtype,
            attn_implementation=self.attn_impl
        )
        self.kind = kind
        print(f"[ModelManager] Successfully loaded {kind}\n")
        return self.model

    def get_supported_languages(self):
        if self.model:
            return self.model.get_supported_languages()
        return ["Auto", "ZH", "EN", "JP", "KO", "FR", "DE"]

    def get_supported_speakers(self):
        if self.model and hasattr(self.model, "get_supported_speakers"):
            return self.model.get_supported_speakers()
        return []


def _audio_to_tuple(audio) -> Optional[Tuple[int, np.ndarray]]:
    if audio is None:
        return None
    if isinstance(audio, tuple) and len(audio) == 2:
        sr, wav = audio
        if isinstance(wav, np.ndarray):
            return (int(sr), wav)
    if hasattr(audio, "name"):
        import soundfile as sf
        wav, sr = sf.read(audio.name, dtype="float32")
        if wav.ndim > 1:
            wav = wav.mean(axis=-1)
        return (int(sr), wav)
    return None


def _wav_to_gradio_audio(wav: np.ndarray, sr: int):
    if wav.ndim == 1:
        wav = wav[np.newaxis, :]
    return (sr, wav.T)


def save_audio_file(wav: np.ndarray, sr: int, output_dir: str = "outputs") -> str:
    """保存音频文件到本地目录"""
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"tts_{timestamp}.wav"
    filepath = os.path.join(output_dir, filename)
    if wav.ndim > 1:
        wav = wav.squeeze()
    sf.write(filepath, wav, sr)
    return filepath


def _dtype_from_str(s: Optional[str]) -> Optional[torch.dtype]:
    if not s:
        return None
    m = {
        "fp32": torch.float32,
        "float32": torch.float32,
        "fp16": torch.float16,
        "float16": torch.float16,
        "bf16": torch.bfloat16,
        "bfloat16": torch.bfloat16,
    }
    return m.get(s.lower(), None)


def _collect_gen_kwargs(args) -> Dict[str, Any]:
    kwargs: Dict[str, Any] = {}
    if args.top_p is not None:
        kwargs["top_p"] = args.top_p
    if args.temperature is not None:
        kwargs["temperature"] = args.temperature
    if args.max_new_tokens is not None:
        kwargs["max_new_tokens"] = args.max_new_tokens
    if args.do_sample is not None:
        kwargs["do_sample"] = args.do_sample
    return kwargs


def _build_choices_and_map(raw_list, is_lang=False):
    """Build display choices and mapping from raw list with Chinese labels."""
    if not raw_list:
        return [], {}
    
    lang_names = {
        "auto": "自动检测",
        "chinese": "中文",
        "english": "英语",
        "german": "德语",
        "italian": "意大利语",
        "portuguese": "葡萄牙语",
        "spanish": "西班牙语",
        "japanese": "日语",
        "korean": "韩语",
        "french": "法语",
        "russian": "俄语",
    }
    
    speaker_names = {
        "vivian": "薇薇安", "serena": "塞雷娜", "emma": "艾玛", "olivia": "奥利维亚",
        "ava": "艾娃", "isabella": "伊莎贝拉", "sophia": "索菲亚", "mia": "米娅",
        "charlotte": "夏洛特", "amelia": "阿米莉亚", "harper": "哈珀", "evelyn": "伊芙琳",
        "abigail": "阿比盖尔", "ella": "艾拉", "elizabeth": "伊丽莎白", "camila": "卡米拉",
        "luna": "露娜", "sofia": "索菲亚", "avery": "艾弗里", "mila": "米拉",
        "aria": "阿里亚", "scarlett": "斯嘉丽", "penelope": "佩内洛普", "layla": "莱拉",
        "chloe": "克洛伊", "victoria": "维多利亚", "madison": "麦迪逊", "eleanor": "埃莉诺",
        "grace": "格蕾丝", "nora": "诺拉", "riley": "莱莉", "zoey": "佐伊",
        "hannah": "汉娜", "hazel": "黑兹尔", "lily": "莉莉", "ellie": "艾莉",
        "violet": "维奥莱特", "aurora": "奥罗拉", "savannah": "萨凡纳", "audrey": "奥黛丽",
        "brooklyn": "布鲁克林", "bella": "贝拉", "claire": "克莱尔", "skylar": "斯凯勒",
        "lucy": "露西", "paisley": "佩斯利", "everly": "埃弗利", "anna": "安娜",
        "caroline": "卡罗琳", "nova": "诺瓦", "genesis": "吉妮西丝", "emilia": "艾米莉亚",
        "kennedy": "肯尼迪", "samantha": "萨曼莎", "maya": "玛雅", "willow": "威洛",
        "kinsley": "金斯利", "naomi": "娜奥米", "aaliyah": "阿莉娅", "elena": "埃琳娜",
        "sarah": "萨拉", "ariana": "阿里安娜", "allison": "艾莉森", "gabriella": "加布里埃拉",
        "alice": "爱丽丝", "madelyn": "玛德琳", "cora": "科拉", "ruby": "鲁比",
        "eva": "伊娃", "serenity": "塞雷妮蒂", "autumn": "奥顿", "adalynn": "阿达琳",
        "gianna": "吉安娜", "valentina": "瓦伦蒂娜", "isla": "艾拉", "eliana": "埃利安娜",
        "quinn": "奎因", "nevaeh": "内瓦", "ivy": "艾薇", "sadie": "赛迪",
        "piper": "派珀", "lydia": "莉迪亚", "alexa": "亚历克萨", "josephine": "约瑟芬",
        "emery": "埃默里", "julia": "朱莉娅", "delilah": "黛利拉", "arianna": "阿里安娜",
        "vivian": "薇薇安", "kaylee": "凯莉", "sophie": "索菲", "brielle": "布里埃尔",
        "madeline": "玛德琳",
    }
    
    display = []
    mapping = {}
    for x in raw_list:
        key = str(x).lower()
        if is_lang and key in lang_names:
            display.append(lang_names[key])
            mapping[lang_names[key]] = x
        elif not is_lang and key in speaker_names:
            display.append(speaker_names[key])
            mapping[speaker_names[key]] = x
        else:
            display.append(str(x))
            mapping[str(x)] = x
    
    return display, mapping


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Qwen3 TTS Gradio Demo")
    parser.add_argument("checkpoint", type=str, nargs="?", help="Path to model checkpoint dir")
    parser.add_argument("--checkpoint-pos", type=str, default=None, help="Path to positional checkpoint dir")
    parser.add_argument("--ip", type=str, default="127.0.0.1", help="Server IP")
    parser.add_argument("--port", type=int, default=8000, help="Server port")
    parser.add_argument("--share", action="store_true", help="Enable Gradio share")
    parser.add_argument("--device", type=str, default="cuda", help="Device (cuda/cpu)")
    parser.add_argument("--dtype", type=str, default=None, help="dtype: fp32/fp16/bf16")
    parser.add_argument("--flash-attn", action="store_true", help="Use flash attention 2")
    parser.add_argument("--concurrency", type=int, default=1, help="Concurrency limit")
    parser.add_argument("--ssl-certfile", type=str, default=None, help="SSL certificate file")
    parser.add_argument("--ssl-keyfile", type=str, default=None, help="SSL key file")
    parser.add_argument("--ssl-verify", action="store_true", help="Verify SSL")
    parser.add_argument("--top-p", type=float, default=None, help="Top-p sampling")
    parser.add_argument("--temperature", type=float, default=None, help="Temperature")
    parser.add_argument("--max-new-tokens", type=int, default=None, help="Max new tokens")
    parser.add_argument("--do-sample", type=lambda x: x.lower() in ("true", "1"), default=None, help="Do sample")
    return parser


def _resolve_checkpoint(args) -> str:
    if args.checkpoint:
        return args.checkpoint
    if args.checkpoint_pos:
        return args.checkpoint_pos
    raise ValueError("Either checkpoint or checkpoint-pos must be provided")


def build_demo(manager: ModelManager, gen_kwargs_default: Dict[str, Any]):
    def _gen_common_kwargs() -> Dict[str, Any]:
        return dict(gen_kwargs_default)

    # Modern Theme & CSS - Unified Studio Design
    css = """
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

    :root {
        --primary-gradient: linear-gradient(135deg, #6366f1 0%, #a855f7 50%, #3b82f6 100%);
        --bg-blur: blur(16px);
        --transition: all 0.4s cubic-bezier(0.4, 0, 0.2, 1);
        --card-shadow: 0 8px 32px 0 rgba(31, 38, 135, 0.15);
        --bg-app: #f5f7ff;
        --glass-bg: rgba(255, 255, 255, 0.7);
        --glass-border: rgba(255, 255, 255, 0.4);
        --text-main: #1e293b;
        --text-muted: #64748b;
        --radius: 20px;
    }

    .dark {
        --bg-app: #0f172a;
        --glass-bg: rgba(30, 41, 59, 0.7);
        --glass-border: rgba(255, 255, 255, 0.1);
        --text-main: #f8fafc;
        --text-muted: #94a3b8;
    }

    * { font-family: 'Inter', system-ui, sans-serif !important; }

    body, .gradio-container {
        background: var(--bg-app) !important;
        color: var(--text-main) !important;
        min-height: 100vh !important;
        display: flex !important;
        flex-direction: column !important;
        margin: 0 !important;
        padding: 0 !important;
    }

    .main-container {
        flex: 1 0 auto !important;
        max-width: 1200px !important;
        margin: 0 auto !important;
        padding: 10px 20px !important;
        width: 100% !important;
    }

    .glass-card {
        background: var(--glass-bg) !important;
        backdrop-filter: var(--bg-blur) !important;
        -webkit-backdrop-filter: var(--bg-blur) !important;
        border: 1px solid var(--glass-border) !important;
        border-radius: var(--radius) !important;
        box-shadow: var(--card-shadow) !important;
        padding: 16px !important;
        margin-bottom: 12px !important;
        transition: var(--transition);
        box-sizing: border-box !important;
    }

    /* Systemic Height Synchronization Rules */
    .sync-height-group {
        display: flex !important;
        flex-direction: column !important;
        min-height: 320px !important;
        transition: height 0.3s ease-out !important;
    }

    /* Centered Titles in Headers */
    .sync-height-group h3, .sync-height-group .header-title {
        text-align: center !important;
        width: 100% !important;
        margin: 0 0 15px 0 !important;
        font-weight: 700 !important;
        line-height: 1.4 !important;
    }

    /* Responsive Heights */
    @media (max-width: 767px) {
        .sync-height-group { min-height: 240px !important; }
    }
    @media (min-width: 768px) and (max-width: 1024px) {
        .sync-height-group { min-height: 280px !important; }
    }
    @media (min-width: 1025px) {
        .sync-height-group { min-height: 320px !important; }
    }

    /* Force Title Visibility */
    #studio-title-main {
        font-size: 2.5rem !important;
        font-weight: 800 !important;
        color: #6366f1 !important; /* Solid Indigo */
        text-align: center !important;
        margin: 10px 0 !important;
        display: block !important;
        visibility: visible !important;
        opacity: 1 !important;
        background: none !important;
        -webkit-text-fill-color: #6366f1 !important;
        text-shadow: 0 2px 4px rgba(0,0,0,0.1) !important;
    }

    .primary-btn {
        background: var(--primary-gradient) !important;
        border: none !important;
        color: white !important;
        font-weight: 700 !important;
        padding: 12px !important;
        border-radius: 12px !important;
        cursor: pointer !important;
        width: 100% !important;
        transition: var(--transition);
    }
    
    .primary-btn:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(99, 102, 241, 0.4);
    }

    #qwen-final-footer {
        flex-shrink: 0 !important;
        margin-top: auto !important;
        padding: 20px !important;
        text-align: center !important;
        border-top: 1px solid var(--glass-border);
        background: var(--glass-bg);
        width: 100%;
    }
    """

    with gr.Blocks(css=css, theme=gr.themes.Default()) as demo:
        # Height Synchronization Logic
        gr.HTML("""
            <script>
            (function() {
                function syncHeights() {
                    const groups = document.querySelectorAll('.sync-height-group');
                    if (groups.length < 2) return;
                    
                    let maxHeight = 0;
                    // Reset to measure natural height
                    groups.forEach(g => {
                        g.style.height = 'auto';
                        // Only measure visible elements
                        if (g.offsetWidth > 0 || g.offsetHeight > 0) {
                            maxHeight = Math.max(maxHeight, g.offsetHeight);
                        }
                    });
                    
                    // Apply max height to all visible sync groups
                    groups.forEach(g => {
                        if (g.offsetWidth > 0 || g.offsetHeight > 0) {
                            g.style.height = maxHeight + 'px';
                        }
                    });
                }

                // Debounce to prevent performance issues
                function debounce(func, wait) {
                    let timeout;
                    return function() {
                        clearTimeout(timeout);
                        timeout = setTimeout(func, wait);
                    };
                }

                const debouncedSync = debounce(syncHeights, 300);

                // Observe for content changes (Gradio dynamic updates)
                const observer = new MutationObserver((mutations) => {
                    debouncedSync();
                });

                document.addEventListener('DOMContentLoaded', () => {
                    const config = { childList: true, subtree: true, characterData: true };
                    const container = document.querySelector('.main-container');
                    if (container) observer.observe(container, config);
                    
                    window.addEventListener('resize', debouncedSync);
                    
                    // Initial sync after Gradio finishes rendering
                    setTimeout(syncHeights, 1500);
                    
                    // Sync when tab changes
                    document.addEventListener('click', (e) => {
                        if (e.target.closest('button')) {
                            setTimeout(syncHeights, 100);
                        }
                    });
                });
            })();
            </script>
        """)

        with gr.Column(elem_classes=["main-container"]):
            # Hero Section
            with gr.Column():
                gr.HTML('<span style="font-size: 48px; display: block; text-align: center; margin-bottom: 0;">🎙️</span>')
                gr.HTML('<h1 id="studio-title-main">Magic-change-Qwen3-TTS Studio</h1>')
                gr.HTML('<p style="text-align: center; font-size: 1.1rem; opacity: 0.8; margin-bottom: 20px;">全能语音创作中心 · 统一模型管理架构</p>')

            with gr.Row():
                # Left Column: Model Selection & Inputs
                with gr.Column(scale=3):
                    with gr.Tabs() as tabs:
                        # Tab 1: Custom Voice
                        with gr.Tab("🎭 预设音色 (CustomVoice)", id="custom_voice"):
                            with gr.Group(elem_classes=["glass-card", "sync-height-group"], elem_id="left-sync-custom"):
                                gr.HTML("<h3 class='header-title'>📝 文本输入 (Text Input)</h3>")
                                text_custom = gr.Textbox(label="", placeholder="输入文字...", lines=5, show_label=False)
                                with gr.Row():
                                    lang_custom = gr.Dropdown(label="语言", choices=LANGUAGE_CHOICES, value="自动检测")
                                    spk_custom = gr.Dropdown(label="音色选择", choices=SPEAKER_CHOICES, value="薇薇安 (vivian)")
                                
                                # Speaker detail description area
                                spk_desc_custom = gr.HTML(SPEAKER_DESCRIPTIONS["薇薇安 (vivian)"])
                                
                                instruct_custom = gr.Textbox(label="情感指令", placeholder="例如：温柔地、开心地...")
                                btn_custom = gr.Button("立即生成 ✨", elem_classes=["primary-btn"])

                        # Tab 2: Voice Design
                        with gr.Tab("🎨 语音设计 (VoiceDesign)", id="voice_design"):
                            with gr.Group(elem_classes=["glass-card", "sync-height-group"], elem_id="left-sync-design"):
                                gr.HTML("<h3 class='header-title'>🎨 文本输入 (Text Input)</h3>")
                                text_design = gr.Textbox(label="", placeholder="输入文字...", lines=5, show_label=False)
                                lang_design = gr.Dropdown(label="语言", choices=LANGUAGE_CHOICES, value="自动检测")
                                instruct_design = gr.Textbox(label="音色描述", placeholder="如：深沉的中年男声，语气沉稳...")
                                btn_design = gr.Button("开始设计 ⚡", elem_classes=["primary-btn"])

                        # Tab 3: Voice Clone
                        with gr.Tab("👥 语音克隆 (VoiceClone)", id="voice_clone"):
                            with gr.Group(elem_classes=["glass-card", "sync-height-group"], elem_id="left-sync-clone"):
                                gr.HTML("<h3 class='header-title'>👥 文本输入 (Text Input)</h3>")
                                text_clone = gr.Textbox(label="", placeholder="输入需要合成的文字...", lines=5, show_label=False)
                                lang_clone = gr.Dropdown(label="语言", choices=LANGUAGE_CHOICES, value="自动检测")
                                ref_audio = gr.Audio(label="参考音频", type="filepath")
                                ref_text = gr.Textbox(label="参考文本", placeholder="请输入参考音频中说话人的原话（ICL 模式必填）...")
                                x_vector_only = gr.Checkbox(label="仅使用说话人向量模式 (免参考文本)", value=False)
                                btn_clone = gr.Button("启动克隆 🚀", elem_classes=["primary-btn"])

                # Right Column: Shared Output & Logs
                with gr.Column(scale=2):
                    with gr.Group(elem_classes=["glass-card", "sync-height-group"], elem_id="right-sync-output"):
                        gr.HTML("<h3 class='header-title'>🔊 渲染输出 (Render Output)</h3>")
                        audio_out = gr.Audio(label="", show_label=False)
                        gr.HTML("<div style='margin-top: 20px;'><h3 class='header-title'>ℹ️ 系统日志 (Logs)</h3></div>")
                        status_out = gr.Textbox(label="", show_label=False, placeholder="准备就绪...", interactive=False, lines=10)

        # Shared Footer
        gr.HTML("""
            <style>
            @keyframes fadeInScale {
                from { opacity: 0; transform: translateY(10px) scale(0.98); }
                to { opacity: 1; transform: translateY(0) scale(1); }
            }
            .spk-desc-animate {
                animation: fadeInScale 0.4s ease-out forwards;
            }
            </style>
            <div id="qwen-final-footer">
                <p>© 2026 我的随手日记 | 基于阿里云 Qwen3 模型开发</p>
                <p style="font-size: 0.7rem;">⚠️ 本工具生成的语音内容由 AI 自动合成，请勿用于非法用途。</p>
            </div>
        """)

        # Backend Logic with Dynamic Loading
        def run_task(kind, text, lang_label, spk_label=None, instruct=None, audio=None, r_text=None, x_vec=False, progress=gr.Progress()):
            try:
                if not text or not text.strip():
                    return None, "请输入合成文本"
                
                # Map labels to internal values
                lang = LANGUAGE_MAP.get(lang_label, "auto")
                spk = SPEAKER_MAP.get(spk_label, spk_label) # Use label directly if not in map (for VoiceDesign/VoiceClone)
                
                # Dynamic Model Switching
                progress(0.1, desc=f"正在检查模型状态...")
                if manager.kind != kind:
                    progress(0.2, desc=f"正在动态加载 {kind.upper()} 模型，请稍候...")
                    manager.load(kind)
                
                tts = manager.model
                progress(0.4, desc="模型就绪，正在分析文本...")
                
                # Run actual inference based on kind
                if kind == "custom_voice":
                    wavs, sr = tts.generate_custom_voice(text=text.strip(), language=lang, speaker=spk, instruct=instruct, **_gen_common_kwargs())
                elif kind == "voice_design":
                    wavs, sr = tts.generate_voice_design(text=text.strip(), language=lang, instruct=instruct, **_gen_common_kwargs())
                else:  # voice_clone
                    if not audio:
                        return None, "❌ 错误：语音克隆模式需要上传参考音频"
                    
                    # Validate ref_text requirement for ICL mode
                    if not bool(x_vec) and (not r_text or not r_text.strip()):
                        return None, "❌ 错误：在当前（ICL）模式下，必须提供参考音频对应的【参考文本】以获得更好的克隆效果。如果是为了免输入文本，请勾选“仅使用说话人向量模式”。"
                        
                    wavs, sr = tts.generate_voice_clone(text=text.strip(), language=lang, ref_audio=audio, ref_text=r_text, x_vector_only_mode=bool(x_vec), **_gen_common_kwargs())
                
                progress(0.8, desc="音频生成完成，正在保存...")
                output_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "outputs")
                saved_path = save_audio_file(wavs[0], sr, output_dir)
                
                return _wav_to_gradio_audio(wavs[0], sr), f"✅ 渲染成功！\n模式: {kind.upper()}\n路径: {saved_path}"
            except Exception as e:
                import traceback
                return None, f"❌ 任务失败: {str(e)}\n{traceback.format_exc()}"

        # Event Bindings
        def update_spk_desc(spk_name):
            return SPEAKER_DESCRIPTIONS.get(spk_name, "")
            
        spk_custom.change(fn=update_spk_desc, inputs=[spk_custom], outputs=[spk_desc_custom])

        btn_custom.click(fn=run_task, inputs=[gr.State("custom_voice"), text_custom, lang_custom, spk_custom, instruct_custom], outputs=[audio_out, status_out])
        btn_design.click(fn=run_task, inputs=[gr.State("voice_design"), text_design, lang_design, gr.State(None), instruct_design], outputs=[audio_out, status_out])
        btn_clone.click(fn=run_task, inputs=[gr.State("voice_clone"), text_clone, lang_clone, gr.State(None), gr.State(None), ref_audio, ref_text, x_vector_only], outputs=[audio_out, status_out])

    return demo


def main(argv=None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    models_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    models_dir = os.path.join(models_dir, "models")
    
    dtype = _dtype_from_str(args.dtype)
    attn_impl = "flash_attention_2" if args.flash_attn else None

    # Initialize Unified Model Manager
    manager = ModelManager(
        models_dir=models_dir,
        device=args.device,
        dtype=dtype or torch.float16,
        attn_impl=attn_impl
    )

    gen_kwargs_default = _collect_gen_kwargs(args)
    demo = build_demo(manager, gen_kwargs_default)

    launch_kwargs: Dict[str, Any] = dict(
        server_name=args.ip,
        server_port=args.port,
        share=args.share,
        ssl_verify=True if args.ssl_verify else False,
    )
    if args.ssl_certfile is not None:
        launch_kwargs["ssl_certfile"] = args.ssl_certfile
    if args.ssl_keyfile is not None:
        launch_kwargs["ssl_keyfile"] = args.ssl_keyfile

    demo.queue(default_concurrency_limit=int(args.concurrency)).launch(**launch_kwargs)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
