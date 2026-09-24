import torch
import torch.nn as nn
import torch.optim as optim
from dataset import CardioDataset
from torch.utils.data import DataLoader, random_split


class MLP(nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.net(x)


dataset = CardioDataset("data/cardio_train.csv")
generator = torch.Generator().manual_seed(42)
train_set, val_set, test_set = random_split(dataset, [0.8, 0.1, 0.1], generator=generator)

train_loader = DataLoader(train_set, batch_size=64, shuffle=True)
val_loader = DataLoader(val_set, batch_size=64, shuffle=False)
test_loader = DataLoader(test_set, batch_size=64, shuffle=False)

batch = next(iter(train_loader))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = MLP(input_size=batch['features'].shape[1], hidden_size=128).to(device)

criterion = nn.BCELoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)

l1_lambda = 1e-4
l2_lambda = 1e-3

for epoch in range(10):
    model.train()
    running_loss = 0.0
    running_correct = 0
    running_total = 0
    for batch in train_loader:
        inputs, targets = batch["features"].to(device), batch["labels"].to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        base_loss = criterion(outputs, targets)

        l1_penalty = sum(p.abs().sum() for p in model.parameters())
        l2_penalty = sum((p ** 2).sum() for p in model.parameters())

        loss = base_loss + l1_lambda * l1_penalty + l2_lambda * l2_penalty

        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        preds = (outputs > 0.5).float()
        running_correct += (preds == targets).sum().item()
        running_total += targets.size(0)

    print(f"Epoch {epoch+1:02d} | loss={running_loss / len(train_loader):.4f} | acc={running_correct/running_total:.4f}")
