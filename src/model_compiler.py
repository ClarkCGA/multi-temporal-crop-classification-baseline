import os
from pathlib import Path
from datetime import datetime, timedelta
import torch
from torch import optim
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import _LRScheduler
from torch.nn import init


def get_optimizer(optimizer, params, lr, momentum):
    """
    Get an instance of the specified optimizer with the given parameters.

    Parameters:
        optimizer (str): The name of the optimizer. Options: 
                              "sgd", "nesterov", "adam", "amsgrad".
        params (iterable): The parameters to optimize.
        lr (float): The learning rate.
        momentum (float): The momentum factor for optimizers that support it.

    Returns:
        torch.optim.Optimizer: An instance of the specified optimizer with the 
        given parameters.
    """
    optimizer = optimizer.lower()

    if optimizer == "sgd":
        return torch.optim.SGD(params, lr, momentum=momentum)
    elif optimizer == "nesterov":
        return torch.optim.SGD(params, lr, momentum=momentum, nesterov=True)
    elif optimizer == "adam":
        return torch.optim.Adam(params, lr)
    elif optimizer == 'amsgrad':
        return torch.optim.Adam(params, lr, amsgrad=True)
    else:
        raise ValueError(f"{optimizer} currently not supported, please choose a valid optimizer")


def init_weights(model, init_type="normal", gain=0.02):
    """Initialize the network weights using various initialization methods.

    Args:
        model (torch.nn.Module): The initialized model.
        init_type (str): The initialization type. Supported initialization methods: 
                         "normal", "xavier", "kaiming", "orthogonal"
                         Default is "normal" for random initialization
                         using a normal distribution.
        gain (float): The scaling factor for the initialized weights.
    """
    class_name = model.__class__.__name__
    if hasattr(model, "weight") and (class_name.find("Conv") != -1 or 
                                     class_name.find("Linear") != -1):
        if init_type == "normal":
            init.normal_(model.weight.data, 0.0, gain)
        elif init_type == "xavier":
            init.xavier_normal_(model.weight.data, gain=gain)
        elif init_type == "kaiming":
            init.kaiming_normal_(model.weight.data, a=0, mode="fan_out")
        elif init_type == "orthogonal":
            init.orthogonal_(model.weight.data, gain=gain)
        else:
            raise NotImplementedError(f"initialization method {init_type} is not implemented.")

    if hasattr(model, "bias") and model.bias is not None:
        init.constant_(model.bias.data, 0.0)

    if class_name.find("BatchNorm2d") != -1:
        init.normal_(model.weight.data, 1.0, gain)
        init.constant_(model.bias.data, 0.0)

    print(f"initialize network with {init_type}.")


class PolynomialLR(_LRScheduler):
    """Polynomial learning rate decay until the step reaches the max_decay_steps.

    Args:
        optimizer (Optimizer): Wrapped optimizer.
        max_decay_steps (int): The maximum number of steps after which the learning 
                               rate stops decreasing.
        min_learning_rate (float): The minimum value of the learning rate. 
                                   Learning rate decay stops at this value.
        power (float): The power of the polynomial.
    """

    def __init__(self, optimizer, max_decay_steps, min_learning_rate=1e-5, power=1.0):

        if max_decay_steps <= 1.:
            raise ValueError('max_decay_steps should be greater than 1.')

        self.max_decay_steps = max_decay_steps
        self.min_learning_rate = min_learning_rate
        self.power = power
        self.last_step = 0

        super().__init__(optimizer)

    def get_lr(self):
        if self.last_step > self.max_decay_steps:
            return [self.min_learning_rate for _ in self.base_lrs]

        return [(base_lr - self.min_learning_rate) *
                ((1 - self.last_step / self.max_decay_steps) ** self.power) +
                self.min_learning_rate for base_lr in self.base_lrs]

    def step(self, step=None):

        if step is None:
            step = self.last_step + 1
        self.last_step = step if step != 0 else 1

        if self.last_step <= self.max_decay_steps:
            decay_lrs = [(base_lr - self.min_learning_rate) *
                         ((1 - self.last_step / self.max_decay_steps) ** self.power) +
                         self.min_learning_rate for base_lr in self.base_lrs]

            for param_group, lr in zip(self.optimizer.param_groups, decay_lrs):
                param_group['lr'] = lr


class ModelCompiler:

    def __init__(self, model, working_dir, out_dir, num_classes, inch, gpu_devices=[0],
                 model_init_type="kaiming", params_init=None, freeze_params=None):
        r"""
        Train the model.

        Arguments:
            model (ordered Dict) -- initialized model either vanilla or pre-trained depending on
                                    the argument 'params_init'.
            working_dir (str) -- General Directory to store output from any experiment.
            out_dir (str) -- specific output directory for the current experiment.
            num_classes (int) -- number of output classes based on the classification scheme.
            inch (int) -- number of input channels.
            gpu_devices (list) -- list of GPU indices to use for parallelism if multiple GPUs are available.
                                  Default is set to index 0 for a single GPU.
            model_init_type -- (str) model initialization choice if it's not pre-trained.
            params_init --(str or None) Path to the saved model parameters. If set to 'None', a vanilla model will
                          be initialized.
            freeze_params (list) -- list of integers that show the index of layers in a pre-trained
                                    model (on the source domain) that we want to freeze for fine-tuning
                                    the model on the target domain used in the model-based transfer learning.
        """

        self.working_dir = working_dir
        self.out_dir = out_dir

        self.num_classes = num_classes
        self.inch = inch
        self.gpu_devices = gpu_devices
        self.use_sync_bn = use_sync_bn
        self.model_init_type = model_init_type
        self.params_init = params_init
        self.checkpoint_dirpath = None

        self.model = model
        self.model_name = self.model.__class__.__name__

        if self.params_init:
            self.load_params(self.params_init, freeze_params)

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        if self.device.type == "cuda":
            print("----------GPU available----------")
            if self.gpu_devices:
                torch.cuda.set_device(self.gpu_devices[0])
                self.model = torch.nn.DataParallel(self.model, device_ids=self.gpu_devices)
        else:
            print('----------No GPU available, using CPU instead----------')
            self.model = self.model.to(device)


        if params_init is None:
            init_weights(self.model, self.model_init_type, gain=0.01)

        num_params = sum([p.numel() for p in self.model.parameters() if p.requires_grad])
        print("total number of trainable parameters: {:2.1f}M".format(num_params / 1000000))

        if self.params_init:
            print("---------- Pre-trained model compiled successfully ----------")
        else:
            print("---------- Vanilla Model compiled successfully ----------")

    def load_params(self, dir_params, freeze_params):
        """
        Load parameters from a file and update the model's state dictionary.

        Args:
            dir_params (str): Directory path to the parameters file.
            freeze_params (list): List of indices corresponding to the model's parameters that should be frozen.

        Returns:
            None
        """

        # inparams = torch.load(self.params_init, map_location='cuda:0')
        inparams = torch.load(self.params_init)

        model_dict = self.model.state_dict()

        if "module" in list(inparams.keys())[0]:
            inparams_filter = {k[7:]: v.cpu() for k, v in inparams.items() if k[7:] in model_dict}
        else:
            inparams_filter = {k: v.cpu() for k, v in inparams.items() if k in model_dict}

        model_dict.update(inparams_filter)

        # load new state dict
        self.model.load_state_dict(model_dict)

        if freeze_params:
            for i, p in enumerate(self.model.parameters()):
                if i in freeze_params:
                    p.requires_grad = False

    def fit(self, trainDataset, valDataset, epochs, optimizer_name, lr_init, 
            lr_policy, criterion, momentum=None, resume=False, resume_epoch=None, **kwargs):
        """
        Train the model on the provided datasets.

        Args:
            trainDataset: The loaded training dataset.
            valDataset: The loaded validation dataset.
            epochs (int): The number of epochs to train.
            optimizer_name (str): The name of the optimizer to use.
            lr_init (float): The initial learning rate.
            lr_policy (str): The learning rate policy.
            criterion: The loss criterion.
            momentum (float, optional): The momentum factor for the optimizer (default: None).
            resume (bool, optional): Whether to resume training from a checkpoint (default: False).
            resume_epoch (int, optional): The epoch from which to resume training (default: None).
            **kwargs: Additional arguments specific to certain learning rate policies.

        Returns:
            None
        """

        # Set the folder to save results.
        working_dir = self.working_dir
        out_dir = self.out_dir
        model_dir = "{}/{}/{}_ep{}".format(working_dir, out_dir, self.model_name, epochs)

        if not os.path.exists(Path(working_dir) / out_dir / model_dir):
            os.makedirs(Path(working_dir) / out_dir / model_dir)

        self.checkpoint_dirpath = Path(working_dir) / out_dir / model_dir / "chkpt"
        if not os.path.exists(self.checkpoint_dirpath):
            os.makedirs(self.checkpoint_dirpath)

        os.chdir(Path(working_dir) / out_dir / model_dir)

        print("-------------------------- Start training --------------------------")
        start = datetime.now()

        writer = SummaryWriter('../')
        lr = lr_init

        optimizer = get_optimizer(optimizer_name,
                                  filter(lambda p: p.requires_grad, self.model.parameters()),
                                  lr,
                                  momentum)

        # Initialize the learning rate scheduler
        if lr_policy == "StepLR":
            step_size = kwargs.get("step_size", 3)
            gamma = kwargs.get("gamma", 0.98)
            scheduler = optim.lr_scheduler.StepLR(optimizer,
                                                  step_size=step_size,
                                                  gamma=gamma)

        elif lr_policy == "MultiStepLR":
            milestones = kwargs.get("milestones", [5, 10, 20, 35, 50, 70, 90])
            gamma = kwargs.get("gamma", 0.5)
            scheduler = optim.lr_scheduler.MultiStepLR(optimizer,
                                                       milestones=milestones,
                                                       gamma=gamma)

        elif lr_policy == "ReduceLROnPlateau":
            mode = kwargs.get("mode", "min")
            factor = kwargs.get("factor", 0.8)
            patience = kwargs.get("patience", 3)
            threshold = kwargs.get("threshold", 0.0001)
            threshold_mode = kwargs.get("threshold_mode", "rel")
            min_lr = kwargs.get("min_lr", 3e-6)
            verbose = kwargs.get("verbose", True)
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer,
                                                             mode=mode,
                                                             factor=factor,
                                                             patience=patience,
                                                             threshold=threshold,
                                                             threshold_mode=threshold_mode,
                                                             min_lr=min_lr,
                                                             verbose=verbose)

        elif lr_policy == "PolynomialLR":
            max_decay_steps = kwargs.get("max_decay_steps", 75)
            min_learning_rate = kwargs.get("min_learning_rate", 1e-5)
            power = kwargs.get("power", 0.8)
            scheduler = PolynomialLR(optimizer,
                                     max_decay_steps=max_decay_steps,
                                     min_learning_rate=min_learning_rate,
                                     power=power)

        elif lr_policy == "CyclicLR":
            base_lr = kwargs.get("base_lr", 3e-5)
            max_lr = kwargs.get("max_lr", 0.01)
            step_size_up = kwargs.get("step_size_up", 1100)
            mode = kwargs.get("mode", "triangular")
            scheduler = optim.lr_scheduler.CyclicLR(optimizer,
                                                    base_lr=base_lr,
                                                    max_lr=max_lr,
                                                    step_size_up=step_size_up,
                                                    mode=mode)

        else:
            scheduler = None

        # Resume the model from the specified checkpoint in the config file.
        train_loss = []
        val_loss = []

        if resume:
            model_state_file = os.path.join(self.checkpoint_dirpath, "{}_checkpoint.pth.tar".format(resume_epoch))
            if os.path.isfile(model_state_file):
                checkpoint = torch.load(model_state_file)
                resume_epoch = checkpoint["epoch"]
                scheduler.load_state_dict(checkpoint["scheduler"])
                self.model.load_state_dict(checkpoint["state_dict"])
                optimizer.load_state_dict(checkpoint["optimizer"])
                train_loss = checkpoint["train loss"]
                val_loss = checkpoint["Evaluation loss"]

        # epoch iteration
        if resume:
            iterable = range(resume_epoch, epochs)
        else:
            iterable = range(epochs)

        for t in iterable:

            print("Epoch [{}/{}]".format(t + 1, epochs))

            start_epoch = datetime.now()

            train_one_epoch(trainDataset, self.model, criterion, optimizer, 
                            scheduler, lr_policy, device=self.device, 
                            train_loss=train_loss)
            validate_one_epoch(valDataset, self.model, criterion, device=self.device, 
                               val_loss=val_loss)

            # Update the scheduler
            if lr_policy in ["StepLR", "MultiStepLR"]:
                scheduler.step()
                print("LR: {}".format(scheduler.get_last_lr()))

            if lr_policy == "ReduceLROnPlateau":
                scheduler.step(val_loss[t])

            if lr_policy == "PolynomialLR":
                scheduler.step(t)
                print("LR: {}".format(optimizer.param_groups[0]['lr']))

            print('time:', (datetime.now() - start_epoch).seconds)

            # Adjust logger to resume status and save checkpoints in defined intervals.
            checkpoint_interval = 20

            writer.add_scalars("Loss",
                               {"train loss": train_loss[t],
                                "Evaluation loss": val_loss[t]},
                               t + 1)

            if (t + 1) % checkpoint_interval == 0:
                torch.save({"epoch": t + 1,
                            "state_dict": self.model.state_dict() if len(self.gpu_devices) > 1 else \
                                self.model.module.state_dict(),
                            "scheduler": scheduler.state_dict(),
                            "optimizer": optimizer.state_dict(),
                            "train loss": train_loss,
                            "Evaluation loss": val_loss},
                           os.path.join(self.checkpoint_dirpath, f"{t + 1}_checkpoint.pth.tar"))

        writer.close()

        duration_in_sec = (datetime.now() - start).seconds
        duration_format = str(timedelta(seconds=duration_in_sec))
        print(f"----------- Training finished in {duration_format} -----------")

    def accuracy_evaluation(self, evalDataset, filename):
        """
        Evaluate the accuracy of the model on the provided evaluation dataset.

        Args:
            evalDataset (DataLoader): The evaluation dataset to evaluate the model on.
            filename (str): The filename to save the evaluation results in the output CSV.
    """

        if not os.path.exists(Path(self.working_dir) / self.out_dir):
            os.makedirs(Path(self.working_dir) / self.out_dir)

        os.chdir(Path(self.working_dir) / self.out_dir)

        print("---------------- Start evaluation ----------------")

        start = datetime.now()

        do_accuracy_evaluation(evalDataset, self.model, filename, self.gpu)

        duration_in_sec = (datetime.now() - start).seconds
        print(
            f"---------------- Evaluation finished in {duration_in_sec}s ----------------")


    def save(self, save_object="params"):
        """
        Save model parameters or the entire model to disk.

        Args:
            save_object (str): Specifies whether to save "params" or "model". 
            Defaults to "params".
        """

        if save_object == "params":
            if len(self.gpu_devices) > 1:
                torch.save(self.model.module.state_dict(),
                           os.path.join(self.checkpoint_dirpath, "{}_final_state.pth".format(self.model_name)))
            else:
                torch.save(self.model.state_dict(),
                           os.path.join(self.checkpoint_dirpath, "{}_final_state.pth".format(self.model_name)))

            print("--------------------- Model parameters is saved to disk ---------------------")

        elif save_object == "model":
            torch.save(self.model,
                       os.path.join(self.checkpoint_dirpath, "{}_final_state.pth".format(self.model_name)))

        else:
            raise ValueError("Improper object type.")
