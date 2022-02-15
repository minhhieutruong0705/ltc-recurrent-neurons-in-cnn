import torch
import torch.optim as optim
import os

from models import CRNet
from models import CRNet_3FC
from models import CRNetNCP_YRNN
from models import BCEDiceLossWithLogistic
from utils import CovidTrainer
from utils import CovidValidator
from covid_facade import get_transformers, get_data_loaders
from train_facade import init_weights, log_to_file, save_checkpoint, load_checkpoint


if __name__ == '__main__':
    # record files
    checkpoints_dir = "../covid_crnet3fc_checkpoints"
    checkpoint_name = "covid_crnet3fc_checkpoint.pth.tar"
    train_log_file = os.path.join(checkpoints_dir, "covid_log.txt")

    # create folders
    os.makedirs(checkpoints_dir, exist_ok=True)
    os.makedirs(os.path.join(checkpoints_dir, "stats"), exist_ok=True)

    # image params
    img_dim = 256
    img_crop_dim = 224
    lung_mask_incor = False

    # train params
    epochs = 360
    batch_size = 64
    learning_rate = 1e-4
    scheduler_period = 10
    in_channels = 3
    if lung_mask_incor:
        in_channels = 4

    # models
    # model = CRNet(in_channels=in_channels).cuda()
    model = CRNet_3FC(in_channels=in_channels).cuda()
    # model = CRNetNCP_YRNN(in_channels=in_channels).cuda()
    print(model)

    # augmentation params
    random_crop_scale = 0.8
    rotation_limit = 15
    blur_kernel_range = (3, 7)
    mean_norm = [0.0, 0.0, 0.0]
    std_norm = [1.0, 1.0, 1.0]
    max_pixel_value = 255.0
    contrast_factor = brightness_factor = 0.2

    # loss params
    bce_reduction = 'mean'

    # device
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print("[INFO] Using " + device + " for training ...")

    # path
    covid_dir = "../datasets/Dataset_PNG/COVID"
    non_covid_dir = "../datasets/Dataset_PNG/NONCOVID"
    train_covid_file = "../datasets/Dataset_PNG/covid_train.txt"
    train_non_covid_file = "../datasets/Dataset_PNG/normal_train.txt"
    val_covid_file = "../datasets/Dataset_PNG/covid_validation.txt"
    val_non_covid_file = "../datasets/Dataset_PNG/normal_validation.txt"
    test_covid_file = "../datasets/Dataset_PNG/covid_test.txt"
    test_non_covid_file = "../datasets/Dataset_PNG/normal_test.txt"

    # augmentation
    train_transformer, val_transformer = get_transformers(
        img_dim=img_dim, img_crop_dim=img_crop_dim,
        random_crop_scale=random_crop_scale,
        rotation_limit=rotation_limit,
        blur_kernel_range=blur_kernel_range,
        contrast_factor=contrast_factor,
        brightness_factor=brightness_factor, mean_norm=mean_norm,
        std_norm=std_norm, max_pixel_value=max_pixel_value
    )

    # data loaders
    train_loader, val_loader, test_loader = get_data_loaders(
        covid_dir=covid_dir,
        non_covid_dir=non_covid_dir,
        list_train_covid=train_covid_file,
        list_train_non_covid=train_non_covid_file,
        list_val_covid=val_covid_file,
        list_val_non_covid=val_non_covid_file,
        list_test_covid=test_covid_file,
        list_test_non_covid=test_non_covid_file,
        batch_size=batch_size,
        train_transformer=train_transformer,
        val_transformer=val_transformer,
        lung_mask_incor=lung_mask_incor
    )

    # init
    model.apply(init_weights)
    loss_function = BCEDiceLossWithLogistic(reduction=bce_reduction).cuda()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer, T_max=scheduler_period)
    scaler = torch.cuda.amp.GradScaler()

    # trainer
    covid_trainer = CovidTrainer(
        model=model,
        train_loader=train_loader,
        loss_function=loss_function,
        optimizer=optimizer,
        scaler=scaler,
        device=device
    )
    print("[INFO] Trainer loaded!")

    # validator
    covid_validator = CovidValidator(
        model=model,
        val_loader=val_loader,
        loss_function=loss_function,
        device=device
    )

    # train & eval
    for i in range(epochs):
        # train
        loss, accuracy, f1, dice, precision, recall, tp, tn, fp, fn = covid_trainer.train()
        log_to_file(train_log_file, "TRAIN", i, loss, accuracy, f1, dice, precision, recall, tp, tn, fp, fn)
        # eval
        loss, accuracy, f1, dice, precision, recall, tp, tn, fp, fn = covid_validator.eval()
        log_to_file(train_log_file, "EVAL", i, loss, accuracy, f1, dice, precision, recall, tp, tn, fp, fn)
        # save checkpoint
        checkpoint = {
            "epoch": i,
            "state_dict": model.state_dict(),
            "optimizer": optimizer.state_dict(),
        }
        save_checkpoint(checkpoint, os.path.join(checkpoints_dir, checkpoint_name), checkpoint_index=i)

    # tester
    covid_tester = CovidValidator(
        model=model,
        val_loader=test_loader,
        loss_function=loss_function,
        device=device,
        is_test=True
    )

    # test
    checkpoint_basename = checkpoint_name.split('.', maxsplit=1)
    for i in range(epochs):
        checkpoint_path = os.path.join(checkpoints_dir, f"{checkpoint_basename[0]}{i}.{checkpoint_basename[1]}")
        load_checkpoint(
            checkpoint=torch.load(checkpoint_path),
            model=model,
            optimizer=optimizer
        )
        loss, accuracy, f1, dice, precision, recall, tp, tn, fp, fn = covid_tester.eval()
        log_to_file(train_log_file, "TEST", i, loss, accuracy, f1, dice, precision, recall, tp, tn, fp, fn)
