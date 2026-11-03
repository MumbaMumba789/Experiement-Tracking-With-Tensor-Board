import torch
from pathlib import Path


def save_model(model, model_name, target_dir="models/saved_models"):

    target_dir = Path(target_dir)

    target_dir.mkdir(parents=True, exist_ok=True)

    model_path = target_dir / f"{model_name}.pth"

    torch.save(model.state_dict(), model_path)

    print(f"Model saved to: {model_path}")


def load_model(model, model_path, device):

    model.load_state_dict(
        torch.load(model_path, map_location=device, weights_only=True)
    )

    model.to(device)

    model.eval()

    return model