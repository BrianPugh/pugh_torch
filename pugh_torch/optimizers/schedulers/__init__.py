import torch

scheduler_lookup = {
    "lambdalr": torch.optim.lr_scheduler.LambdaLR,
    "multiplicativelr": torch.optim.lr_scheduler.MultiplicativeLR,
    "steplr": torch.optim.lr_scheduler.StepLR,
    "multisteplr": torch.optim.lr_scheduler.MultiStepLR,
    "exponentiallr": torch.optim.lr_scheduler.ExponentialLR,
    "cosineannealinglr": torch.optim.lr_scheduler.CosineAnnealingLR,
    "reducelronplateau": torch.optim.lr_scheduler.ReduceLROnPlateau,
    "cycliclr": torch.optim.lr_scheduler.CyclicLR,
    "onecyclelr": torch.optim.lr_scheduler.OneCycleLR,
    "cosineannealingwarmrestarts": torch.optim.lr_scheduler.CosineAnnealingWarmRestarts,
}


def get_scheduler(s):
    s = s.lower()
    return scheduler_lookup[s]


get = get_scheduler
