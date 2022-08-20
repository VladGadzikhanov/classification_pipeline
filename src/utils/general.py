import json

import numpy as np
import torch.onnx
import yaml


def drop_onnx(model, model_path, input_tensor_example, output_numpy):
    """
    Save PyTorch model to ONNX format
    :param model: pytorch model, instance of nn.Module
    :param model_path: file path for saving
    :param input_tensor_example: torch.Tensor input example
    :param output_numpy: np.array output example
    :return: None
    """
    # save output example
    np.save(model_path.replace(".onnx", "_output.npy"), output_numpy)

    if input_tensor_example is not None:
        input_tensor_example = input_tensor_example.to("cpu")

    torch.onnx.export(
        model.to("cpu"),  # model being run
        input_tensor_example,  # model input (or a tuple for multiple inputs)
        f"{model_path}",  # where to save the model (can be a file or file-like object)
        export_params=True,  # store the trained parameter weights inside the model file
        opset_version=9,  # the ONNX version to export the model to
        do_constant_folding=True,  # whether to execute constant folding for optimization
        input_names=["input"],  # the model's input names
        output_names=["output"],  # the model's output names
        dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}},  # variable lenght axes
    )
    print(f"Model is successfully saved to ONNX: {model_path}.")


def save_json(value, path):
    """Dump a json-serializable object to a json file."""
    with open(path, "w") as f:
        json.dump(value, f)


def load_json(path):
    """Load the contents of a json file."""
    with open(path, "r") as f:
        return json.load(f)


def load_yaml(path):
    """Load the contents of a yaml file."""
    with open(path, "r") as f:
        return yaml.safe_load(f)


def save_yaml(value, path):
    """Load the contents of a yaml file."""
    with open(path, "w") as f:
        return yaml.safe_dump(value, f)
