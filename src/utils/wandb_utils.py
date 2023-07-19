import wandb

def log(log, epoch=None, split='val'):
    '''
    Log scalars in log in wandb

    Arg(s):
        epoch : int
            epoch number
        log : dict
            dictionary of metrics to log
        split : str
            train or val

    '''
    assert split in ['train', 'val']
    wandb_log = {}
    if epoch is not None:
        wandb_log['epoch'] = epoch
    for key, val in log.items():
        # Do not log anything that is not a scalar (str, list, array)
        try:
            _ = len(val)
        except:
            # Ensure split is in the key
            if split not in key:
                key = '{}/{}'.format(split, key)
            wandb_log[key] = val
    wandb.log(wandb_log)