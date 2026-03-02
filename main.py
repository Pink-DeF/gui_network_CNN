import eel
import numpy as np
import torch
import torch.nn as nn


eel.init("")

img_height = 28
img_width = 28
model_path = "my_model.pth"


class DigitCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(32, 64, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 5 * 5, 64),
            nn.ReLU(),
            nn.Linear(64, 10),
        )

    def forward(self, x):
        x = self.features(x)
        return self.classifier(x)


def load_pth_model(path: str):
    """Загружает .pth модель в одном из популярных форматов."""
    try:
        scripted_model = torch.jit.load(path, map_location="cpu")
        scripted_model.eval()
        return scripted_model
    except RuntimeError:
        pass

    checkpoint = torch.load(path, map_location="cpu")

    if isinstance(checkpoint, nn.Module):
        checkpoint.eval()
        return checkpoint

    if isinstance(checkpoint, dict):
        state_dict = checkpoint.get("state_dict", checkpoint)
        model = DigitCNN()
        model.load_state_dict(state_dict)
        model.eval()
        return model

    raise ValueError("Не удалось распознать формат .pth модели")


model = load_pth_model(model_path)


def load_image(arr):
    img = np.array(arr, dtype=np.float32).reshape(1, 1, img_height, img_width)
    return torch.from_numpy(img)


def predict_digit(arr):
    img_tensor = load_image(arr)
    with torch.no_grad():
        logits = model(img_tensor)
    return int(torch.argmax(logits, dim=1).item())


@eel.expose
def get_array(n):
    predicted_digit_num = predict_digit(n)
    print(f"Цифра на изображении: {predicted_digit_num}")
    return predicted_digit_num


eel.start("main.html", size=(2560, 1440))
