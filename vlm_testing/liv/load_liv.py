import os
from huggingface_hub import hf_hub_download
import torch
from .models.model_liv import LIV


def load_liv(modelid='resnet50', device=None, **kwargs):
    """
    Load LIV model from HuggingFace or local cache.
    
    Args:
        modelid: Model ID (default: 'resnet50')
        device: Device to load model on (default: 'cuda' if available, else 'cpu')
        **kwargs: Additional parameters to pass to LIV initialization:
            - scratch: bool (default: False)
            - grad_text: bool (default: True)
            - metric: str (default: 'cos')
            - lr: float (default: 1e-5)
            - weight_decay: float (default: 0.001)
            - modelid: str (default: 'RN50')
            - clipweight: float (default: 1.0)
            - visionweight: float (default: 1.0)
            - langweight: float (default: 0.0)
            - gamma: float (default: 0.98)
            - num_negatives: int (default: 0)
    
    Returns:
        LIV model wrapped in DataParallel
    """
    assert modelid == 'resnet50'
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    home = os.path.join(os.path.expanduser("~"), ".liv")

    if not os.path.exists(os.path.join(home, modelid)):
        os.makedirs(os.path.join(home, modelid), exist_ok=True)
    folderpath = os.path.join(home, modelid)
    modelpath = os.path.join(home, modelid, "model.pt")

    if not os.path.exists(modelpath):
        # Default reliable download from HuggingFace Hub
        print(f"Downloading LIV model files to {folderpath}...")
        hf_hub_download(repo_id="jasonyma/LIV", filename="model.pt", local_dir=folderpath)
        print("Download complete.")

    # Default parameters matching the provided config
    liv_params = {
        'modelid': kwargs.get('modelid', 'RN50'),
        'device': kwargs.get('device', device),
        'lr': kwargs.get('lr', 1e-5),
        'weight_decay': kwargs.get('weight_decay', 0.001),
        'visionweight': kwargs.get('visionweight', 1.0),
        'langweight': kwargs.get('langweight', 0.0),
        'clipweight': kwargs.get('clipweight', 1.0),
        'gamma': kwargs.get('gamma', 0.98),
        'metric': kwargs.get('metric', 'cos'),
        'num_negatives': kwargs.get('num_negatives', 0),
        'grad_text': kwargs.get('grad_text', True),
        'scratch': kwargs.get('scratch', False),
    }
    
    # Directly instantiate LIV
    rep = LIV(**liv_params)
    rep = torch.nn.DataParallel(rep)
    
    # Load state dict
    state_dict = torch.load(modelpath, map_location=torch.device(device))['liv']
    rep.load_state_dict(state_dict)
    return rep
