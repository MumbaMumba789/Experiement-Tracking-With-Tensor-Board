import torch
import torch.nn as nn
from torchvision import models
from torch.utils.tensorboard import SummaryWriter
from going_modular.data_setup import create_dataloaders
from going_modular.model_builder import create_model, freeze_base_model
from going_modular.engine import train_step, test_step
from going_modular.utils import save_model


# Configuration
BATCH_SIZE = 32
EPOCHS = 5
LEARNING_RATE = 0.001
MODEL_NAME = "resnet18"

device = "cuda" if torch.cuda.is_available() else "cpu"

train_dir = "data/pizza_steak_sushi/train"
test_dir = "data/pizza_steak_sushi/test"

weights = models.ResNet18_Weights.DEFAULT
transform = weights.transforms()

train_dataloader, test_dataloader, class_names = create_dataloaders(
    train_dir=train_dir,
    test_dir=test_dir,
    transform=transform,
    batch_size=BATCH_SIZE
)

print("Classes:", class_names)

model = create_model(
    model_name=MODEL_NAME,
    num_classes=len(class_names),
    pretrained=True
)

model = freeze_base_model(model)

model = model.to(device)


loss_fn = nn.CrossEntropyLoss()

optimizer = torch.optim.Adam(
    filter(lambda p: p.requires_grad, model.parameters()),
    lr=LEARNING_RATE
)


for epoch in range(EPOCHS):

    train_loss = train_step(
        model,
        train_dataloader,
        loss_fn,
        optimizer,
        device
    )

    test_loss, test_accuracy = test_step(
        model,
        test_dataloader,
        loss_fn,
        device
    )

    print(
        f"Epoch: {epoch + 1} | "
        f"Train Loss: {train_loss:.4f} | "
        f"Test Loss: {test_loss:.4f} | "
        f"Accuracy: {test_accuracy:.2%}"
    )

save_model(model, MODEL_NAME)

writer = SummaryWriter(
    log_dir="runs/resnet18_experient_1"
)

writer.add_scalar("Loss/train", train_loss, epoch)
writer.add_scalar("Loss/test", test_loss, epoch)
writer.add_scalar("Accuracy/test", test_accuracy, epoch)

writer.add_hparams(
    {
        "model": MODEL_NAME,
        "learning_rate": LEARNING_RATE,
        "batch_size": BATCH_SIZE,
        "epochs": EPOCHS
    },

    {
        "hparam/final_test_loss": test_loss,
        "hparam/final_accuracy": test_accuracy
    }
)

writer.close()