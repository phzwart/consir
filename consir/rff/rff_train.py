import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import dask


def train_rffcerberus(
    model,
    train_loader: DataLoader,
    test_loader: DataLoader,
    num_epochs: int,
    optimizer,
    criterion,
    quantile=[0.10, 0.50, 0.90],
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
    scheduler=None,
    display_results=None,
    show=100
):
    """
    Function to train the RFFCerberus model.

    Args:
        model (nn.Module): The model to be trained (RFFCerberus).
        train_loader (DataLoader): DataLoader for training data.
        test_loader (DataLoader): DataLoader for test/validation data.
        num_epochs (int): Number of training epochs.
        optimizer (torch.optim.Optimizer): Optimizer for training.
        criterion (function): Loss function, such as pinball_loss.
        quantile (float): Quantile level for pinball loss.
        device (str): Device to run the training on (e.g., 'cpu', 'cuda'). Default is 'cuda' if available.
        scheduler (torch.optim.lr_scheduler, optional): Learning rate scheduler. Default is None.
        display_results (function, optional): Function to display results after each epoch. Default is None.

    Returns:
        None
    """
    model.to(device)
    model.train()

    for epoch in range(num_epochs):
        running_loss = 0.0

        for i, (inputs, targets) in enumerate(train_loader):
            # Use Dask's `delayed` feature to parallelize the data transfer if needed
            inputs, targets = dask.delayed(inputs.to)(device), dask.delayed(targets.to)(device)

            # Ensure values are computed in a compatible way
            inputs, targets = dask.compute(inputs, targets)

            # Zero the parameter gradients
            optimizer.zero_grad()

            # Forward pass
            lower, median, upper = model(inputs)

            # Compute the pinball loss
            loss_lower = criterion(median, targets, quantile[0])
            loss_median = criterion(median, targets, quantile[1])
            loss_upper = criterion(median, targets, quantile[2])
            loss = loss_lower + loss_median + loss_upper

            # Backward pass
            loss.backward()

            # Optimizer step
            optimizer.step()

            # Accumulate loss for logging
            running_loss += loss.item()

        # Step scheduler if available
        if scheduler:
            scheduler.step()

        # Average loss over training set
        avg_loss = running_loss / len(train_loader)
        if epoch%show==0:
            print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}")

        # Evaluate on test/validation set
        if test_loader:
            model.eval()
            with torch.no_grad():
                test_loss = 0.0
                for inputs, targets in test_loader:
                    inputs, targets = dask.delayed(inputs.to)(device), dask.delayed(targets.to)(device)
                    inputs, targets = dask.compute(inputs, targets)

                    lower, median, upper = model(inputs)
                    loss_lower = criterion(median, targets, quantile[0])
                    loss_median = criterion(median, targets, quantile[1])
                    loss_upper = criterion(median, targets, quantile[2])


                    test_loss += loss_lower.item() + loss_median.item() + loss_upper.item()

                avg_test_loss = test_loss / len(test_loader)
                if epoch%show==0:
                    print(f"Test Loss: {avg_test_loss:.4f}")

            model.train()

        # Display results if function is provided
        if display_results:
            display_results(model, epoch, avg_loss, avg_test_loss if test_loader else None)

    print("Training complete.")
