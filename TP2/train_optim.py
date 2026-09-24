import torch
import torch.nn as nn
import torch.optim as optim
from dataset import CardioDataset
from torch.utils.data import DataLoader, random_split
from torch.utils.tensorboard import SummaryWriter


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


# Chargement des données (une seule fois, réutilisé par toutes les expériences)
dataset = CardioDataset("data/cardio_train.csv")
generator = torch.Generator().manual_seed(42)
train_set, val_set, test_set = random_split(dataset, [0.8, 0.1, 0.1], generator=generator)

train_loader = DataLoader(train_set, batch_size=64, shuffle=True)
val_loader = DataLoader(val_set, batch_size=64, shuffle=False)
test_loader = DataLoader(test_set, batch_size=64, shuffle=False)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

input_size = next(iter(train_loader))["features"].shape[1]


def train_model(opt_name, learning_rate=0.001, epochs=30):
    model = MLP(input_size=input_size, hidden_size=128).to(device)
    criterion = nn.BCELoss()

    # Choix de l'optimiseur
    if opt_name == "SGD":
        optimizer = optim.SGD(model.parameters(), lr=learning_rate)
    elif opt_name == "Momentum":
        optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)
    elif opt_name == "RMSprop":
        optimizer = optim.RMSprop(model.parameters(), lr=learning_rate)
    elif opt_name == "Adam":
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    writer = SummaryWriter(f'runs/cardio_{opt_name}_lr{learning_rate}')

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for batch in train_loader:
            inputs, targets = batch["features"].to(device), batch["labels"].to(device)
            optimizer.zero_grad()
            loss = criterion(model(inputs), targets)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        # Logging de la loss moyenne par epoch
        writer.add_scalar('Training Loss', running_loss / len(train_loader), epoch)
        print(f"[{opt_name}] Epoch {epoch+1:02d} | loss={running_loss / len(train_loader):.4f}")

    writer.close()
    return model


# Lancement des expériences
trained_models = {}
for opt in ["SGD", "Momentum", "RMSprop", "Adam"]:
    print(f"\n=== Entraînement avec {opt} ===")
    trained_models[opt] = train_model(opt, learning_rate=0.001)
