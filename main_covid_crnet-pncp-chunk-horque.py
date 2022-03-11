import torch
import torch.nn as nn
import torch.optim as optim
import torchinfo
import os

from models import CRNetNCP_ChunkPRNN
from models import BCEDiceLossWithLogistic
from utils_covid import CovidTrainer
from utils_covid import CovidValidator
from facade_covid import get_transformers, get_data_loaders
from facade_train import init_weights, log_to_file, save_checkpoint, load_checkpoint

if __name__ == '__main__':
    training_name = "covid_crnet-pncp-chunk-horque"
    shuffler_version = 1

    # image params
    img_dim = 256
    img_crop_dim = 224
    lung_mask_incor = False

    # train params
    epochs = 175
    batch_size = 64
    learning_rate = 1e-4
    scheduler_period = 10
    in_channels = 3 if not lung_mask_incor else 4

    # models
    bi_directional = False
    seq_horizontal = True
    seq_zigzag = False
    model = CRNetNCP_ChunkPRNN(
        in_channels=in_channels,
        bi_directional=bi_directional,
        seq_horizontal=seq_horizontal,
        seq_zigzag=seq_zigzag
    ).cuda()
    print(model)
    model_summary = torchinfo.summary(
        model=model,
        input_size=(batch_size, in_channels, img_crop_dim, img_crop_dim)
    )

    # checkpoint files
    checkpoints_dir = f"../{training_name}_{shuffler_version}_checkpoints"
    checkpoint_name = f"{training_name}.pth.tar"

    # record files
    record_dir = f"records/covid_{shuffler_version}/{training_name}_{shuffler_version}"
    train_log_file = os.path.join(record_dir, f"{training_name}_{shuffler_version}_log.txt")
    model_summary_file = os.path.join(record_dir, f"{training_name}_model-summary.txt")

    # create folders
    os.makedirs(checkpoints_dir, exist_ok=True)
    os.makedirs(record_dir, exist_ok=True)

    # write model specification to a file
    with open(model_summary_file, 'w') as f:
        f.write(str(model))
        f.write('\n\n')
        f.write(str(model_summary))

    # augmentation params
    random_crop_scale = 0.8
    rotation_limit = 15
    blur_kernel_range = (3, 7)
    mean_norm = [0.0, 0.0, 0.0]
    std_norm = [1.0, 1.0, 1.0]
    max_pixel_value = 255.0
    contrast_factor = brightness_factor = 0.2

    # loss params
    loss_reduction = 'mean'

    # device
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print("[INFO] Using " + device + " for training ...")

    # path for images
    covid_dir = "../datasets/Dataset_Covid/COVID"
    non_covid_dir = "../datasets/Dataset_Covid/NONCOVID"

    # path for train, validation, and test sets
    train_covid_file = f"records/covid_{shuffler_version}/covid_train_{shuffler_version}.txt"
    train_non_covid_file = f"records/covid_{shuffler_version}/normal_train_{shuffler_version}.txt"
    val_covid_file = f"records/covid_{shuffler_version}/covid_val_{shuffler_version}.txt"
    val_non_covid_file = f"records/covid_{shuffler_version}/normal_val_{shuffler_version}.txt"
    test_covid_file = f"records/covid_{shuffler_version}/covid_test_{shuffler_version}.txt"
    test_non_covid_file = f"records/covid_{shuffler_version}/normal_test_{shuffler_version}.txt"

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
    loss_function = BCEDiceLossWithLogistic(reduction=loss_reduction).cuda()
    # loss_function = nn.MSELoss(reduction=loss_reduction).cuda()
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
    best_score = -1
    for i in range(epochs):
        print(f"[INFO] {i + 1}/{epochs} epochs")
        # train
        loss, accuracy, f1, dice, precision, recall, tp, tn, fp, fn = covid_trainer.train()
        log_to_file(train_log_file, "TRAIN", i, loss, accuracy, f1, dice, precision, recall, tp, tn, fp, fn)
        # eval
        loss, accuracy, f1, dice, precision, recall, tp, tn, fp, fn = covid_validator.eval()
        log_to_file(train_log_file, "VALID", i, loss, accuracy, f1, dice, precision, recall, tp, tn, fp, fn)
        # set checkpoint
        checkpoint = {
            "epoch": i,
            "state_dict": model.state_dict(),
            "optimizer": optimizer.state_dict(),
        }
        # save last checkpoint
        save_checkpoint(
            state=checkpoint,
            checkpoint_file=os.path.join(checkpoints_dir, checkpoint_name.replace(".pth.tar", "_last.pth.tar"))
        )
        # save best checkpoint
        score = accuracy * 0.2 + f1 * 0.3 + dice * 0.5
        if score > best_score:
            best_score = score
            print(f"[INFO] New best scored obtained: {best_score:.2f}")
            save_checkpoint(
                state=checkpoint,
                checkpoint_file=os.path.join(checkpoints_dir, checkpoint_name.replace(".pth.tar", "_best.pth.tar"))
            )

    # tester
    covid_tester = CovidValidator(
        model=model,
        val_loader=test_loader,
        loss_function=loss_function,
        device=device,
        is_test=True
    )

    # test
    checkpoint_file = os.path.join(checkpoints_dir, checkpoint_name.replace(".pth.tar", "_best.pth.tar"))
    best_epoch = load_checkpoint(checkpoint_file=checkpoint_file, model=model, optimizer=optimizer)
    loss, accuracy, f1, dice, precision, recall, tp, tn, fp, fn = covid_tester.eval()
    log_to_file(train_log_file, "TEST", best_epoch, loss, accuracy, f1, dice, precision, recall, tp, tn, fp, fn)
