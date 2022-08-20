import torch
import torch.onnx
from torch.utils.data import DataLoader


def load_model(model_path, model, device):
    state = torch.load(model_path, map_location=device)
    model.load_state_dict(state["state_dict"])
    model = model.to(device)
    model = model.eval()
    return model


def prepare_model_for_eval(model, device):
    model = model.to(device)
    model = model.eval()
    return model


def build_data_loader(data_df, Collection, batch_size=1, num_workers=4):
    data_c = Collection(data_df, "test")
    data_loader = DataLoader(data_c, batch_size=batch_size, num_workers=num_workers, pin_memory=False, shuffle=False)
    return data_loader


def set_device(gpu: str = None):
    if gpu is None:
        device = torch.device("cpu")
    else:
        if gpu == -1:
            device = torch.device("cuda")
        else:
            device = torch.device(f"cuda:{gpu}")
    return device


def freeze_layers(Model, num_layers, names=None):
    """
    Function that allow to freeze Model weights.
    Args:
        Model (torch model): Model for weights freezing
        num_layers (int): number of layers for weights freezing (from the end of NN)
        names (List[str]): names of layers to be freezed.
    """
    modules = list(Model.named_modules())[::-1]
    
    for layer_num in range(num_layers):
        for param in modules[layer_num][1].parameters():
            param.requires_grad = False
    
    if names is not None:
        for module_name, module in modules:
            if module_name in names:
                for param in module.parameters():
                    param.requires_grad = False
                
    return Model