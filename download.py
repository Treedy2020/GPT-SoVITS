import os
import argparse
from huggingface_hub import snapshot_download as hf_sd

# HF_MODELS
SOVITS_MODEL="lj1995/GPT-SoVITS"

# UVR5_MODELS
UVR5_MODEL="lj1995/VoiceConversionWebUI"

MODELS_MAP = {
    "HF_MODELS":[
        {"id": SOVITS_MODEL, "local_path": "GPT_SoVITS/pretrained_models"},
        {"id": UVR5_MODEL, "local_path": "tools/uvr5", "include": "uvr5_weights/*"},
    ],
    "MODELSCOPE_MODELS": [
        # ASR Models
        {"id": "iic/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-pytorch", "local_path": "tools/asr/models"},
        # Damo VAD Models
        {"id": "iic/speech_fsmn_vad_zh-cn-16k-common-pytorch", "local_path": "tools/asr/models"},
        # Damo Punc Models
        {"id": "iic/punc_ct-transformer_zh-cn-common-vocab272727-pytorch", "local_path": "tools/asr/models"},
    ],
    "FastWhispers": [
        {"id": "Systran/faster-whisper-large-v3", "local_path": "tools/asr/models"},
    ]
}

def env_precheck():
    """Do some precheck for the environment checking."""
    command = [("gcc -v", "gcc"),
               ("g++ -v", "g++"),
               ("cmake --version", "cmake"),
               ("ffmpeg -version", "ffmpeg"),
               ("python -c 'import torch; print(torch.__version__)'", "torch version"),
               ("python -c 'import torchaudio; print(torchaudio.__version__)'", "torchaudio version"),
               ]
    for cmd, name in command:
        print(f"Checking {name}...")
        if os.system(cmd) != 0:
            raise RuntimeError(f"Command {cmd} failed. Please check your environment.")
        else:
            print(f"{name} is installed.")

def download_models(use_fastwhispers=False):
    """Downloads required models from HuggingFace and ModelScope.
    
    This function downloads pretrained models needed for GPT-SoVITS from various sources:
    - HuggingFace models: GPT-SoVITS and UVR5 models
    - ModelScope models: Chinese ASR, VAD and punctuation models
    - FastWhisper models (optional): For English/Japanese ASR
    
    Args:
        use_fastwhispers (bool): Whether to download FastWhisper models for English/Japanese ASR.
                                Defaults to False.
    """
    current_dir = os.getcwd()
    for model_type in MODELS_MAP:
        if model_type == "FastWhispers" and not use_fastwhispers:
            continue
        elif model_type == "MODELSCOPE_MODELS":
            os.chdir("tools/asr/models")
            for model in MODELS_MAP[model_type]:
                model_id = model["id"]
                local_path = model["local_path"]
                os.system(f"git clone https://www.modelscope.cn/{model_id}")
            os.chdir(current_dir)
            
        else:
            for model in MODELS_MAP[model_type]:
                model_id = model["id"]
                local_path = model["local_path"]
                if model_type == "HF_MODELS":
                    if "include" in model:
                        hf_sd(model_id, local_dir=local_path, allow_patterns=model['include'])
                    else:
                        hf_sd(model_id, local_dir=local_path)

    
if __name__ == "__main__":
    # env_precheck()
    parser = argparse.ArgumentParser()
    parser.add_argument("--use_hf_mirror", action="store_true", help="Use HuggingFace mirror site for faster download of models.")
    parser.add_argument("--use_fastwhispers", action="store_true", help="Download FastWhisper models")
    args = parser.parse_args()
    if args.use_hf_mirror:
        os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
    download_models(args.use_fastwhispers)
