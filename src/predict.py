import torch
import os, sys
from tqdm import tqdm

# sys.path.insert(0, 'src')
from utils.utils import ensure_dir
import model.metric as module_metric

def predict(data_loader,
            model,
            metric_fns,
            device,
            loss_fn=None,
            output_save_path=None,
            log_save_path=None):
    '''
    Run the model on the data_loader, calculate metrics, and log

    Arg(s):
        data_loader : torch Dataloader
            data to test on
        model : torch.nn.Module
            model to run
        loss_fn : module
            loss function
        metric_fns : list[model.metric modules]
            list of metric functions
        device : torch.device
            GPU device
        output_save_path : str or None
            if not None, save model_outputs to save_path
        log_save_path : str or None
            if not None, save metrics to save_path

    Returns :
        log : dict{} of metrics
    '''

    # Hold data for calculating metrics
    outputs = []
    targets = []

    # Ensure model is in eval mode
    if model.training:
        model.eval()

    with torch.no_grad():
        for idx, item in enumerate(tqdm(data_loader)):
            if len(item) == 3:
                data, target, path = item
            else:
                data, target = item
            data, target = data.to(device), target.to(device)
            output = model(data)

            # Store outputs and targets
            outputs.append(output)
            targets.append(target)

    # Concatenate predictions and targets
    outputs = torch.cat(outputs, dim=0)
    targets = torch.cat(targets, dim=0)

    # Calculate loss
    if loss_fn is not None:
        loss = loss_fn(outputs, targets).item()
        log = {'loss': loss}
    
    n_samples = len(data_loader.sampler)

    # Calculate predictions based on argmax
    predictions = torch.argmax(outputs, dim=1)
    # Targets might be soft labels, if so use argmax to obtain top-1 label
    if len(targets.shape) == 2:
        targets = torch.argmax(targets, dim=1)
    assert targets.shape == predictions.shape

    # Move predictions and target to cpu and convert to numpy to calculate metrics
    predictions = predictions.cpu().numpy()
    targets = targets.cpu().numpy()

    # Calculate metrics
    log = module_metric.compute_metrics(
        metric_fns=metric_fns,
        prediction=predictions,
        target=targets)

    if output_save_path is not None:
        ensure_dir(os.path.dirname(output_save_path))
        outputs_predictions = save_outputs_predictions(
            outputs=outputs,
            save_path=output_save_path
        )
    else:
        outputs_predictions = save_outputs_predictions(
            outputs=outputs,
            save_path=None
        )

    if log_save_path is not None:
        ensure_dir(os.path.dirname(log_save_path))
        torch.save(log, log_save_path)

    return_data = {
        'metrics': log,
        'outputs_predictions': outputs_predictions
    }
    return return_data

def save_outputs_predictions(outputs, save_path=None):
    '''
    Given torch.tensor of outputs, calculate probabilities and predictions and save
    
    Arg(s):
        outputs : torch.tensor
            Model outputs
        save_path : str 
            Path to save outputs to
    '''
    # Ensure directory exists
    if save_path is not None:
        ensure_dir(os.path.dirname(save_path))

    # Calcluate probabilities and predictions
    probabilities = torch.softmax(outputs, dim=1)
    predictions = torch.argmax(probabilities, dim=1)

    # Verify shape and values
    assert probabilities.shape == outputs.shape
    assert (torch.abs(torch.sum(probabilities, dim=1) - 1) < 1e-3).all()
    assert predictions.shape[0] == outputs.shape[0]

    # Move to CPU and convert to numpy (if they aren't already)
    try: 
        outputs = outputs.cpu().numpy()
    except:
        pass
    try:
        probabilities = probabilities.cpu().numpy()
    except:
        pass
    try:
        predictions = predictions.cpu().numpy()
    except:
        pass
    

    save_data = {
        'test': {
            'outputs': outputs,
            'probabilities': probabilities,
            'predictions': predictions
        }
    }
    
    if save_path is not None:
        torch.save(save_data, save_path)
    return save_data

def restore_and_test(model,
                    config,
                    trial_dir,
                    model_restore_path,
                    val_dataloader,
                    metric_fns,
                    device,
                    loss_fn):

    output_save_path = os.path.join(trial_dir, "outputs_predictions.pth")
    log_save_path = os.path.join(trial_dir, "val_metrics.pth")

    model.restore_model(model_restore_path)

    validation_data = predict(
        data_loader=val_dataloader,
        model=model,
        metric_fns=metric_fns,
        device=device,
        loss_fn=loss_fn,
        output_save_path=output_save_path,
        log_save_path=log_save_path)

    return validation_data