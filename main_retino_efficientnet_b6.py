import torch
import torch.nn as nn
import torch.optim as optim
import torchinfo
import os

from models import EfficientNet_B6
from models import WeightedCCELoss
from utils_retino import DiabeticRetinopathyTrainer
from utils_retino import DiabeticRetinopathyValidator
from utils_retino import DiabeticRetinopathyTester
from facade_retino import get_transformers, get_data_loaders
from facade_train import init_weights, log_to_file, save_checkpoint, load_checkpoint, draw_confusion_matrix

if __name__ == '__main__':
    training_name = "retino_efficientnet-b6"
    shuffler_version = 1

    # image params
    img_dim = 528
    img_crop_dim = 528

    # train params
    epochs = 500
    batch_size = 64
    data_load_workers = 6
    learning_rate = 1e-4
    scheduler_period = 10
    in_channels = 3
    class_names = ["No DR", "Mild", "Moderate", "Severe", "Proliferative DR"]
    classes = len(class_names)

    # models
    pretrain = True
    model = EfficientNet_B6(classes=classes, pretrain=pretrain).cuda()
    print(model)
    model_summary = torchinfo.summary(
        model=model,
        input_size=(batch_size, in_channels, img_crop_dim, img_crop_dim)
    )

    # checkpoint files
    checkpoints_dir = f"../{training_name}_{shuffler_version}_checkpoints"
    checkpoint_name = f"{training_name}.pth.tar"

    # record files
    record_dir = f"records/retino_{shuffler_version}/{training_name}_{shuffler_version}"
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
    mean_norm = [0.485, 0.456, 0.406]
    std_norm = [0.229, 0.224, 0.225]
    max_pixel_value = 255.0
    contrast_factor = brightness_factor = 0.2

    # loss params
    loss_reduction = 'mean'

    # device
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print("[INFO] Using " + device + " for training ...")

    # path for images
    train_image_dir = "../datasets/Dataset_DiabeticRetinopathy/train_small"
    test_image_dir = "../datasets/Dataset_DiabeticRetinopathy/test_small"

    # path for train, validation, and test sets
    train_retino_file = f"records/retino_{shuffler_version}/retino_train_{shuffler_version}.csv"
    val_retino_file = f"records/retino_{shuffler_version}/retino_val_{shuffler_version}.csv"
    test_retino_file = f"records/retino_{shuffler_version}/retino_test_{shuffler_version}.csv"

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
    train_loader, val_loader, test_loader, cce_class_weight = get_data_loaders(
        train_dir=train_image_dir,
        test_dir=test_image_dir,
        list_train=train_retino_file,
        list_val=val_retino_file,
        list_test=test_retino_file,
        train_transformer=train_transformer,
        val_transformer=val_transformer,
        batch_size=batch_size,
        data_load_workers=data_load_workers
    )
    print("[INFO] Loss weighted:", cce_class_weight)

    # init
    model.apply(init_weights)
    loss_function = WeightedCCELoss(
        class_weights=torch.tensor(cce_class_weight, device=device),
        reduction=loss_reduction
    )
    # loss_function = WeightedCCEFocalTverskyLossWithSoftmax(
    #     class_weights=torch.tensor(cce_class_weight, device=device),
    #     reduction=loss_reduction,
    #     device=device
    # ).cuda()
    # loss_function = nn.MSELoss(reduction=loss_reduction).cuda()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer, T_max=scheduler_period)
    scaler = torch.cuda.amp.GradScaler()

    # trainer
    retino_trainer = DiabeticRetinopathyTrainer(
        model=model,
        train_loader=train_loader,
        loss_function=loss_function,
        optimizer=optimizer,
        scaler=scaler,
        device=device
    )
    print("[INFO] Trainer loaded!")

    # validator
    retino_validator = DiabeticRetinopathyValidator(
        model=model,
        val_loader=val_loader,
        loss_function=loss_function,
        device=device
    )

    # train & eval
    best_score = -1
    for i in range(epochs):
        print(f"\n[INFO] {i + 1}/{epochs} epochs")
        # train
        loss, accuracy, f1, dice, precision, recall, tp, tn, fp, fn = retino_trainer.train()
        log_to_file(train_log_file, "TRAIN", i, loss, accuracy, f1, dice, precision, recall, tp, tn, fp, fn)
        # eval
        loss, accuracy, f1, dice, precision, recall, tp, tn, fp, fn = retino_validator.eval()
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
        score = accuracy * 1.0 + f1 * 0.0 + dice * 0.0
        if score > best_score:
            best_score = score
            print(f"[INFO] New best scored obtained: {best_score:.2f}")
            save_checkpoint(
                state=checkpoint,
                checkpoint_file=os.path.join(checkpoints_dir, checkpoint_name.replace(".pth.tar", "_best.pth.tar"))
            )

    # tester
    retino_tester = DiabeticRetinopathyTester(
        model=model,
        val_loader=test_loader,
        loss_function=loss_function,
        device=device,
    )

    # test
    checkpoint_file = os.path.join(checkpoints_dir, checkpoint_name.replace(".pth.tar", "_best.pth.tar"))
    best_epoch = load_checkpoint(checkpoint_file=checkpoint_file, model=model, optimizer=optimizer)
    loss, accuracy, f1, dice, precision, recall, tp, tn, fp, fn, confusion_matrix = retino_tester.test()
    log_to_file(train_log_file, "TEST", best_epoch, loss, accuracy, f1, dice, precision, recall, tp, tn, fp, fn)
    draw_confusion_matrix(
        matrix=confusion_matrix,
        class_names=class_names,
        fig_name=f"{training_name}_{shuffler_version}_cfm",
        save_dir=record_dir,
        normalize=True
    )
