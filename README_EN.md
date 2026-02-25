# Magic-change-Qwen3-TTS Studio (Portable Edition)

[中文版](./README.md)

> **Disclaimer**: This project is modified based on [QwenLM/Qwen3-TTS](https://github.com/QwenLM/Qwen3-TTS?tab=readme-ov-file#voice-design), aiming to provide a completely offline, configuration-free, and ready-to-use portable voice creation center.

Qwen3-TTS is a series of open-source speech synthesis models developed by the Alibaba Qwen team. Based on it, this project integrates a complete portable Python environment and optimizes the Web UI interaction experience, supporting high-quality voice generation in more than 10 languages.

---

## 🌟 Core Features

This project integrates the three core capabilities of Qwen3-TTS:

1. **🎭 Preset Voices (CustomVoice)**: Built-in 9 high-quality preset voices (e.g., Vivian, Uncle Fu), supporting precise control of emotions and intonation through natural language instructions (e.g., "gently", "happily").
2. **🎨 Voice Design (VoiceDesign)**: Supports "what you think is what you hear". Just enter a description (e.g., "a deep middle-aged male voice with a steady tone"), and the model can dynamically design a new voice that meets the requirements.
3. **👥 Voice Cloning (VoiceClone)**: Only 3-5 seconds of reference audio are needed to achieve high-fidelity voice cloning. Supports ICL (In-Context Learning) mode to replicate the speaker's subtle rhythm.
4. **⚡ Fast Inference**: Supports Flash Attention 2 acceleration, with first-packet response latency as low as 100ms.
5. **🌍 Multi-language Support**: Fully covers 10 mainstream languages including Chinese, English, Japanese, Korean, German, French, Russian, Portuguese, Spanish, and Italian.

---

## 📦 Installation and Configuration

This project is a **portable version**, no need to install a system-global Python environment or configure complex CUDA environments.

### 1. Hardware Requirements
* **OS**: Windows 10 / 11 (64-bit).
* **GPU**: NVIDIA Graphics Card (RTX 30 series and above recommended for best performance).
* **VRAM**: 8GB and above recommended.
* **Driver**: NVIDIA driver supporting CUDA 12.1 or higher installed.

### 2. Getting the Project

> [!IMPORTANT]
> **Important Note**: Source code cloned directly via Git **does not include the `python312/` portable environment folder**.
> - **Highly Recommended**: Go to the [Releases](https://github.com/luokexiaoguo/Magic-change-Qwen3-TTS/releases) page to download the latest **full portable package** for a ready-to-use experience.
> - **Advanced Users**: If you clone the source code, please install Python 3.12 manually and configure the environment according to `pyproject.toml`.

You can choose from the following methods:

#### Method A: Download Release (Recommended, Ready-to-use)
Go to [GitHub Releases](https://github.com/luokexiaoguo/Magic-change-Qwen3-TTS/releases) to download the full portable package.

#### Method B: Clone via Git (Source only)
```bash
git clone https://github.com/luokexiaoguo/Magic-change-Qwen3-TTS.git
cd Magic-change-Qwen3-TTS
```

1. Ensure the folder path does not contain Chinese characters or spaces (recommended to place in the disk root directory, e.g., `E:\Magic-change-Qwen3-TTS`).

### 3. Download Models
Double-click to run **`下载模型.bat`** (Download Models.bat) in the root directory.
* This is an interactive tool where you can choose to download specific models (e.g., only "Preset Voices") or "All Models" (approx. 10GB).
* Models will be automatically saved in the `models/` folder.

---

## 🚀 Quick Start

1. **Start Service**: Double-click to run **`启动.bat`** (Start.bat).
2. **Access Interface**: The script will automatically open your default browser and navigate to `http://localhost:8001`.
3. **Start Creating**: Select the function tab on the webpage, enter the text, and click "Generate".

---

## 🛠️ Project Structure

```text
Magic-change-Qwen3-TTS/
├── python312/          # Built-in portable Python environment (all dependencies integrated)
├── models/             # Model weights storage directory (generated after running download script)
├── qwen_tts/           # Core code library
├── sox/                # Built-in audio processing tools
├── outputs/            # Generated audio files are automatically saved here
├── 启动.bat            # One-click start script
├── 下载模型.bat        # Model management tool
└── start.py            # Launcher logic
```

---

## 📖 Usage Examples

### Preset Voice Example
* **Text**: The weather is so nice today, let's go for a walk in the park together.
* **Voice**: Vivian (vivian)
* **Emotion Instruction**: Energetic, with a bit of a smile.

### Voice Design Example
* **Voice Description**: A mature and steady female voice, magnetic, speaking at a slightly slower pace.

---

## 📑 API Interface Description

This project primarily provides services through the Web UI. For integration into other systems, you can directly call the `/api/predict` interface provided by Gradio. After starting the service, click the "View API" link at the bottom of the page to view detailed API documentation and sample code.

---

## 🤝 Contribution Guide

1. Feedback on bugs during operation is welcome via Issues.
2. To add new features, please Fork the original repository and submit a Pull Request.
3. Voice descriptions are recommended to be optimized referring to the instructions in [Voice Encyclopedia](file:///e:/Qwen3-TTS/qwen_tts/cli/demo.py).

---

## 📜 License

* The code part of this project follows the [Apache-2.0 License](LICENSE).
* Model weights follow the relevant model license agreements of the Alibaba Qwen team.

---

## 📅 Version History

* **v1.0.0 (2026-02-26)**
    * Initial version released.
    * Integrated portable environment and auto-downloader.
    * Added detailed voice encyclopedia introduction function.
    * Fixed port conflict issues, default to using port 8001.

---

## 📧 Contact

For questions or suggestions, please contact: [luokexiaoguo@foxmail.com]
For issues related to the original model, please visit the official repository: [QwenLM/Qwen3-TTS](https://github.com/QwenLM/Qwen3-TTS)
