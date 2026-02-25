import os
import sys
import subprocess
import time
import webbrowser
import threading
import urllib.request

def wait_and_open_browser(url, max_wait=60):
    """等待服务启动后自动打开浏览器"""
    print(f"⏳ 等待服务启动...")
    start_time = time.time()
    while time.time() - start_time < max_wait:
        try:
            # 尝试连接服务
            urllib.request.urlopen(url, timeout=1)
            print(f"✅ 服务已启动！")
            print(f"🌐 正在打开浏览器...")
            webbrowser.open(url)
            return
        except urllib.error.URLError:
            # 服务尚未启动，继续等待
            time.sleep(0.5)
        except ConnectionResetError:
            # 连接被重置，服务可能正在启动中
            time.sleep(0.5)
        except Exception:
            # 其他异常，继续等待
            time.sleep(0.5)
    print(f"⚠️ 等待超时，请手动打开浏览器访问: {url}")

def main():
    print("=" * 60)
    print("Magic-change-Qwen3-TTS 启动器 (本项目由我的随手日记整理编译)")
    print("=" * 60)
    print("🚀 正在初始化统一语音创作中心...")
    print("💡 您可以在网页中无缝切换预设音色、语音设计和语音克隆模型")
    print("=" * 60)
    
    # 获取本地模型路径
    script_dir = os.path.dirname(os.path.abspath(__file__))
    models_dir = os.path.join(script_dir, "models")
    python_exe = os.path.join(script_dir, "python312", "python.exe")
    demo_py = os.path.join(script_dir, "qwen_tts", "cli", "demo.py")
    
    # 检查基本目录是否存在
    if not os.path.exists(models_dir) or not os.path.exists(os.path.join(models_dir, "Qwen3-TTS-Tokenizer-12Hz")):
        print(f"\n❌ 错误：找不到模型文件！")
        print(f"💡 检测到您尚未下载 AI 模型，请先双击运行项目根目录下的：")
        print(f"   👉 【下载模型.bat】")
        print(f"\n待模型下载完成后，再次运行本项目即可。")
        input("\n按回车键退出...")
        return
    
    print("\n🚀 正在启动服务...")
    print("⏳ 首次加载模型需要一些时间，请耐心等待...")
    print("🌐 服务启动后将自动打开浏览器\n")
    
    # 在后台线程中检测服务并打开浏览器
    browser_thread = threading.Thread(target=wait_and_open_browser, args=("http://localhost:8001", 120))
    browser_thread.daemon = True
    browser_thread.start()
    
    # 设置环境变量
    env = os.environ.copy()
    env['PYTHONPATH'] = script_dir
    
    # 使用python运行demo.py
    # 注意：demo.py 现在支持动态加载，我们可以不传具体模型路径，或者传一个默认路径
    # 这里的 demo.py main 函数期望 argv[1] 是 checkpoint
    default_model = os.path.join(models_dir, "Qwen3-TTS-12Hz-1.7B-CustomVoice")
    
    cmd = [
        python_exe, "-c",
        f"import sys; sys.path.insert(0, r'{script_dir}'); " +
        f"from qwen_tts.cli.demo import main; " +
        f"import sys; sys.argv = ['demo', r'{default_model}', '--ip', '0.0.0.0', '--port', '8001']; " +
        f"main()"
    ]
    
    subprocess.run(cmd, cwd=script_dir, env=env)
    input("\n按回车键退出...")

if __name__ == "__main__":
    main()
