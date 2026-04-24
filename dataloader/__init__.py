from .dataset_M3DFEL import DFEWDataset
import torch


def create_dataloader(args, mode):
    """create dataloader according to args and training/testing mode

    Args:
        args
        mode: String("train" or "test")

    Returns:
        dataloader
    """
    mode_mes = ['Frame', '.jpg']
    poisoned_mode = args.poisoned_mode
    target_label = 2
    if args.dataset == 'CREMA-D':
        mode_mes = ['Frames', '.png']
        target_label = 4
    elif args.dataset == 'MAFW':
        mode_mes = ['Frame', '.png']
        target_label = 4
    if poisoned_mode == 'Benign':
        poisoned_mode = mode_mes[0]
        args.poison_ratio = 0
    dataset = DFEWDataset(args, mode, mode_mes, target_label, poisoned_mode)

    dataloader = None

    # return train_dataset or test_dataset according to the mode
    if mode == "train":
        dataloader = torch.utils.data.DataLoader(dataset,
                                                 batch_size=args.batch_size,
                                                 shuffle=True,
                                                 num_workers=args.workers,
                                                 pin_memory=True,
                                                 drop_last=True)
    elif mode == "test":
        dataloader = torch.utils.data.DataLoader(dataset,
                                                 batch_size=args.batch_size,
                                                 shuffle=False,
                                                 num_workers=args.workers,
                                                 pin_memory=True,
                                                 drop_last=True)
    return dataloader
