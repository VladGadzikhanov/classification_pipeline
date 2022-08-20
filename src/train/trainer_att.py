import torch
import torch.backends.cudnn as cudnn
from torch.nn import functional as F
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.tensorboard import SummaryWriter

from .logger import Log


class TrainerAttention:
    """
    This class takes care of training and validation of our model
    """

    def __init__(
        self,
        model,
        criterion,
        optimizer,
        model_path,
        data_path=None,
        tb_path=None,
        num_epochs=150,
        lr=5e-4,
        gpu=1,
        scheduler=None,
        num_workers=6,
    ):
        self.model_path = model_path
        self.num_workers = num_workers
        self.lr = lr
        self.num_epochs = num_epochs
        self.best_loss = float("inf")
        self.best_score = -float("inf")
        self.phases = ["train", "val"]
        self.net = model
        self.criterion = criterion
        self.data_path = data_path
        self.logger = Log(model_path)
        self.tb_path = tb_path

        if gpu is None:
            self.device = torch.device("cpu")
        else:
            if gpu == -1:
                self.device = torch.device("cuda")
                self.net = torch.nn.DataParallel(model)
            else:
                self.device = torch.device(f"cuda:{gpu}")

        params_to_update = []
        for name, param in model.named_parameters():
            if param.requires_grad:
                params_to_update.append(param)

        self.optimizer = optimizer(params_to_update, lr)

        if scheduler is None:
            self.scheduler = ReduceLROnPlateau(self.optimizer, factor=0.5, patience=20, mode="min", verbose=True)

        self.net = self.net.to(self.device)
        cudnn.benchmark = True

        self.losses = {phase: [] for phase in self.phases}

    def forward(self, images, targets):
        images = images.to(self.device)
        targets = targets.to(self.device)
        outputs, attn_weights = self.net(images)

        loss = self.criterion(outputs, targets)
        return loss, outputs, attn_weights

    def iterate(self, epoch, phase, dataloader):
        self.logger.epoch(epoch, phase)
        self.net.train(phase == "train")
        running_loss = 0.0
        running_acc = 0.0
        running_num = 0
        total_batches = len(dataloader)

        self.optimizer.zero_grad()
        for itr, batch in enumerate(dataloader):
            _, _, images, targets = batch
            loss, outputs, _ = self.forward(images, targets)
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

        lr = -1
        for param_group in self.optimizer.param_groups:
            lr = param_group["lr"]
        self.logger.metrics(epoch_loss, epoch_acc, lr, epoch, phase)

        return epoch_loss, epoch_acc

    def run(self, dataloader, testloader):
        best_epoch = 0
        writer = SummaryWriter(self.tb_path)
        if self.data_path is not None:
            self.logger.log(f"DATA PATH - {self.data_path}")
        # self.optimizer).__name__}') self.logger.log(f'MODEL_PARAMS: width_mult = {self.net.width_mult},
        # use_attention = {self.net.use_attention}, ' f'grayscale = {self.net.grayscale}')
        for epoch in range(self.num_epochs):
            loss, acc = self.iterate(epoch, "train", dataloader)
            val_loss, val_acc = self.iterate(epoch, "val", testloader)

            writer.add_scalar(f"Loss/train", loss, epoch)
            writer.add_scalar(f"Score/train", acc, epoch)
            writer.add_scalar(f"Loss/val", val_loss, epoch)
            writer.add_scalar(f"Score/val", val_acc, epoch)

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
