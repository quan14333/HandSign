"""Download the recognition assets once so practice can run without network."""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from huggingface_hub import snapshot_download
from recognizer import LOCAL_MODEL_PATH, MODEL_NAME

if __name__ == '__main__':
    snapshot_download(MODEL_NAME, local_dir=LOCAL_MODEL_PATH,
                      allow_patterns=['*.json', '*.safetensors', 'pytorch_model.bin', 'classifier_sequential.pth'])
    print(f'Model saved to {LOCAL_MODEL_PATH}')
