import torch
import torch.backends.cudnn as cudnn
from torch.nn import functional as F
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.tensorboard import SummaryWriter

from .logger import Log


class TrainerSeparate:
    """
    This class takes care of training and validation of our model
    """

    def __init__(
        self,
        model,
        criterion,
        optimizer,
        experiment_folder_path,
        tb_path=None,
        num_epochs=150,
        lr=5e-4,
        device=torch.device("cpu"),
        scheduler=None,
        epoch_to_load=None
    ):
        self.net = model
        self.criterion = criterion
        self.experiment_folder_path = experiment_folder_path
        self.num_epochs = num_epochs
        self.lr = lr
        self.device = device
        self.tb_path = tb_path

        self.best_loss = float("inf")
        self.best_score = -float("inf")
        self.phases = ["train", "val"]
        self.logger = Log(experiment_folder_path)

        params_to_update = []
        for name, param in model.named_parameters():
            if param.requires_grad:
                params_to_update.append(param)

        self.optimizer = optimizer(params_to_update, lr)

        if scheduler is None:
            self.scheduler = ReduceLROnPlateau(self.optimizer, factor=0.5, patience=20, mode="min", verbose=True)

        self.epoch_to_load = epoch_to_load if epoch_to_load is not None else -1
        self.net = self.net.to(self.device)
        cudnn.benchmark = True  # For more optimization of execution speed

        self.losses = {phase: [] for phase in self.phases}

    def forward(self, images, targets):
        images = images.to(self.device)
        targets = targets.to(self.device)
        outputs = self.net(images)
        loss = self.criterion(outputs, targets)
        return loss, outputs

    def iterate(self, epoch, phase, dataloader):
        self.logger.epoch(epoch, phase)
        self.net.train(phase == "train")
        running_loss = 0.0
        running_acc = 0.0
        running_num = 0
        total_batches = len(dataloader)

        self.optimizer.zero_grad()
        for itr, batch in enumerate(dataloader):
            _, __, images, targets = batch
            with torch.set_grad_enabled(phase == "train"):  # if phase == "train" then gradients are computed
                loss, outputs = self.forward(images, targets)
                outputs = F.softmax(outputs, dim=1)

                if phase == "train":
                    loss.backward()
                    self.optimizer.step()
                    self.optimizer.zero_grad()

            running_loss += loss.item()
            outputs = outputs.detach().cpu()
            outputs = torch.max(outputs, 1)[1]
            running_acc += (targets == outputs).float().sum().item()
            running_num += list(outputs.size())[0]

        epoch_loss = running_loss / total_batches
        epoch_acc = running_acc / running_num

        self.logger.metrics(epoch_loss, epoch_acc)

        return epoch_loss, epoch_acc

    def run(self, dataloader, testloader):
        best_epoch = 0
        writer = SummaryWriter(self.tb_path)

        self.logger.log(f"experiment_params: lr = {self.lr}, optimizer - {type(self.optimizer).__name__}")

        # for densenet log
        # self.logger.log(f'DENSENET_PARAMS: growth_rate = {self.net.growth_rate}, drop_rate = {self.net.drop_rate}')

        for epoch in range(self.epoch_to_load + 1, self.num_epochs):
            loss, acc = self.iterate(epoch, "train", dataloader)
            val_loss, val_acc = self.iterate(epoch, "val", testloader)

            writer.add_scalar("Loss/train", loss, epoch)
            writer.add_scalar("Score/train", acc, epoch)
            writer.add_scalar("Loss/val", val_loss, epoch)
            writer.add_scalar("Score/val", val_acc, epoch)

            state = {
                "epoch": epoch,
                "best_loss": self.best_loss,
                "best_score": self.best_score,
                "state_dict": self.net.state_dict(),
                "optimizer": self.optimizer.state_dict(),
            }

            if self.scheduler is not None:
                self.scheduler.step(val_loss)

            if val_acc > self.best_score:
                state["best_loss"] = self.best_loss = val_loss
                state["best_score"] = self.best_score = val_acc
                best_epoch = epoch
                self.logger.save(state, epoch)
            self.logger.log("")
        writer.close()
        return best_epoch
