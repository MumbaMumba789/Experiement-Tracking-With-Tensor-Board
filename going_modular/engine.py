import torch


def train_step(model, dataloader, loss_fn, optimizer, device):

    model.train()

    train_loss = 0

    for X, y in dataloader:

        X, y = X.to(device), y.to(device)

        y_pred = model(X)

        loss = loss_fn(y_pred, y)

        train_loss += loss.item()

        optimizer.zero_grad()

        loss.backward()

        optimizer.step()

    train_loss = train_loss / len(dataloader)

    return train_loss


def test_step(model, dataloader, loss_fn, device):

    model.eval()

    test_loss = 0
    correct = 0

    with torch.inference_mode():

        for X, y in dataloader:

            X, y = X.to(device), y.to(device)

            y_pred = model(X)

            loss = loss_fn(y_pred, y)

            test_loss += loss.item()

            predicted_classes = torch.argmax(y_pred, dim=1)

            correct += (predicted_classes == y).sum().item()

    test_loss = test_loss / len(dataloader)

    accuracy = correct / len(dataloader.dataset)

    return test_loss, accuracy