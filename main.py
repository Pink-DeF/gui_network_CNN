import os

import eel
import numpy as np
import torch
import torch.nn as nn


eel.init("")

img_height = 28
img_width = 28
MODEL_PATH = "my_model.pth"


class DigitCNN(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 5 * 5, 64),
            nn.ReLU(),
            nn.Linear(64, 10),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        return self.classifier(x)


def load_model(path: str) -> nn.Module:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Файл модели не найден: {path}")

    model = DigitCNN()
    state = torch.load(path, map_location="cpu")

    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]

    if isinstance(state, dict):
        model.load_state_dict(state)
    else:
        model = state

    model.eval()
    return model


model = load_model(MODEL_PATH)


def load_image(arr: list) -> torch.Tensor:
    img = np.array(arr, dtype=np.float32)
    img = img.reshape(1, 1, img_height, img_width)
    return torch.from_numpy(img)


def predict_digit(arr: list) -> int:
    img = load_image(arr)
    with torch.no_grad():
        logits = model(img)
        prediction = torch.argmax(logits, dim=1)
    return int(prediction.item())


@eel.expose
def get_array(n: list) -> int:
    predicted_digit_num = predict_digit(n)
    print(f"Цифра на изображении: {predicted_digit_num}")
    return predicted_digit_num


eel.start("main.html", size=(2560, 1440))
