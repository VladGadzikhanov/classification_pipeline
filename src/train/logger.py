import os
import time

import torch


class Log:
    def __init__(self, model_name: str):
        model_name = os.path.join(model_name, "model")
        self.model_name = model_name

        if not os.path.exists(model_name):
            os.makedirs(model_name)
        else:
            f = open(f"{self.model_name}/log.txt", "w")
            f.close()

    def log(self, text: str) -> None:
        print(text)
        f = open(f"{self.model_name}/log.txt", "a+")
        f.write(text + "\n")
        f.close()

    def epoch(self, n: int, phase: str) -> None:
        start = time.strftime("%H:%M:%S")
        self.log(f"Epoch: {n} | phase: {phase} |: {start}")

    def metrics(self, loss: float, score: float, digits: int = 6) -> None:
        self.log(f"Loss: {round(loss, digits)}, Score: {round(score, digits)}")

    def save(self, state, epoch):
        self.log("******** New optimal found, saving state ********")
        torch.save(state, f"{self.model_name}/best_model_{epoch}.pt")  # for ordinary learning
