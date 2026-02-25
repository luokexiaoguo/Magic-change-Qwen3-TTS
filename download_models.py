#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
下载 Qwen3-TTS 模型到本地
"""
import os
import sys

def download_models():
    """下载所有需要的模型"""
    models_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")
    os.makedirs(models_dir, exist_ok=True)
    
    print("=" * 60)
    print("Qwen3-TTS 模型下载工具")
    print("=" * 60)
    print("\n请选择要下载的模型：")
    print("1. 预设音色模型 (CustomVoice) - 约 3.5GB")
    print("2. 语音设计模型 (VoiceDesign) - 约 3.5GB")
    print("3. 语音克隆模型 (Base) - 约 3.5GB")
    print("4. 全部模型 - 约 10GB")
    print("5. Tokenizer (必需) - 约 500MB")
    print("=" * 60)
    
    choice = input("\n请选择 (1-5): ").strip()
    
    try:
        from modelscope import snapshot_download
        
        if choice == "1":
            print("\n正在下载预设音色模型...")
            snapshot_download("Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice", 
                            local_dir=os.path.join(models_dir, "Qwen3-TTS-12Hz-1.7B-CustomVoice"))
            print("预设音色模型下载完成！")
            
        elif choice == "2":
            print("\n正在下载语音设计模型...")
            snapshot_download("Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign", 
                            local_dir=os.path.join(models_dir, "Qwen3-TTS-12Hz-1.7B-VoiceDesign"))
            print("语音设计模型下载完成！")
            
        elif choice == "3":
            print("\n正在下载语音克隆模型...")
            snapshot_download("Qwen/Qwen3-TTS-12Hz-1.7B-Base", 
                            local_dir=os.path.join(models_dir, "Qwen3-TTS-12Hz-1.7B-Base"))
            print("语音克隆模型下载完成！")
            
        elif choice == "4":
            print("\n正在下载全部模型，这可能需要很长时间...")
            snapshot_download("Qwen/Qwen3-TTS-Tokenizer-12Hz", 
                            local_dir=os.path.join(models_dir, "Qwen3-TTS-Tokenizer-12Hz"))
            snapshot_download("Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice", 
                            local_dir=os.path.join(models_dir, "Qwen3-TTS-12Hz-1.7B-CustomVoice"))
            snapshot_download("Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign", 
                            local_dir=os.path.join(models_dir, "Qwen3-TTS-12Hz-1.7B-VoiceDesign"))
            snapshot_download("Qwen/Qwen3-TTS-12Hz-1.7B-Base", 
                            local_dir=os.path.join(models_dir, "Qwen3-TTS-12Hz-1.7B-Base"))
            print("全部模型下载完成！")
            
        elif choice == "5":
            print("\n正在下载 Tokenizer...")
            snapshot_download("Qwen/Qwen3-TTS-Tokenizer-12Hz", 
                            local_dir=os.path.join(models_dir, "Qwen3-TTS-Tokenizer-12Hz"))
            print("Tokenizer 下载完成！")
            
        else:
            print("无效选项！")
            return
            
        print(f"\n模型已保存到: {models_dir}")
        
    except Exception as e:
        print(f"\n下载失败: {e}")
        print("请检查网络连接或手动下载模型")
        
    input("\n按回车键退出...")

if __name__ == "__main__":
    download_models()