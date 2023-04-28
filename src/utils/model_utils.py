import torch
import numpy as np

from utils.utils import load_image

def quick_predict(model, image, device=None, data_format="CHW"):
    '''
    Return model output for image(s) at image_path
    Arg(s):
        model : torch.nn.Module
            model to predict
        image_path : str or list[str] or np.array or torch.tensor
            image(s) to predict for
        image : np.array or None
            If no image_path is given, check if image is passed in directly
        device : torch.device
            device that the model is located on
        data_format : str
            channel format for images
    '''
    # Load image
    # Support strings
    if type(image) == str:
        image = load_image(image, data_format=data_format)
        # Expand to 1 x C x H x W and convert to tensor
        image = np.expand_dims(image, axis=0)
        image = torch.from_numpy(image)
    # Support lists of strings
    elif type(image) == list:
        image_list = image
        image = []
        for path in image_list:
            image.append(load_image(path, data_format=data_format))

        image = np.stack(image, axis=0)
        image = torch.from_numpy(image)
    # Support np.arrays
    elif isinstance(image, np.ndarray):
        if len(image.shape) == 3:
            image = np.expand_dims(image, axis=0)
        # Fix dimension ordering
        if data_format == 'CHW' and image.shape[3] == 3:
            image = np.transpose(image, (0, 3, 1, 2))
        elif data_format == 'HWC' and image.shape[1] == 3:
            image = np.transpose(image, (0, 2, 3, 1))
        image = torch.from_numpy(image)
    # Support torch.tensors
    elif torch.is_tensor(image):
        if len(image.shape) == 3:
            image = torch.unsqueeze(image, dim=0)
        # Check dimension ordering
        if data_format == 'CHW' and image.shape[3] == 3:
            image = torch.transpose(image, (0, 3, 1, 2))
        elif data_format == 'HWC' and image.shape[1] == 3:
            image = torch.transpose(image, (0, 2, 3, 1))
    else:
        raise ValueError("Unsupported type {} for image".format(type(image)))

    # Convert from double -> float and switch to device
    image = image.type(torch.FloatTensor).to(device)

    # Pass through model
    with torch.no_grad():
        logits = model(image)
    return logits

def prepare_device(n_gpu_use):
    """
    setup GPU device if available. get gpu device indices which are used for DataParallel
    """
    n_gpu = torch.cuda.device_count()
    if n_gpu_use > 0 and n_gpu == 0:
        print("Warning: There\'s no GPU available on this machine,"
              "training will be performed on CPU.")
        n_gpu_use = 0
    if n_gpu_use > n_gpu:
        print(f"Warning: The number of GPU\'s configured to use is {n_gpu_use}, but only {n_gpu} are "
              "available on this machine.")
        n_gpu_use = n_gpu
    device = torch.device('cuda:0' if n_gpu_use > 0 else 'cpu')
    list_ids = list(range(n_gpu_use))
    return device, list_ids