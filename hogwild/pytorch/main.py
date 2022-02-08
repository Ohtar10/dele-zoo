import torch
import torch.nn as nn
import torch.optim as optim
import torch.multiprocessing as mp
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset, DistributedSampler
from torchvision.datasets import CIFAR10
from tqdm import tqdm


class Model(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 10, kernel_size=5),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(10, 20, kernel_size=5),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(500, 10),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(10, 10),
            nn.LogSoftmax(dim=1)
        )

    def forward(self, x) -> torch.Tensor:
        return self.net(x)


def build_dataset(train: bool = True) -> Dataset:
    return CIFAR10(
        "data",
        train=train,
        download=True,
        transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
    )

def train(model: nn.Module, data_loader: DataLoader) -> None:
    optimizer = optim.Adam(model.parameters())
    criterion = nn.NLLLoss()

    for data, labels in tqdm(data_loader):
        if torch.cuda.is_available():
                data.to("cuda")                
        optimizer.zero_grad()
        loss = criterion(model(data), labels)
        loss.backward()
        optimizer.step()


def test(model: nn.Module, data_loader: DataLoader) -> None:
    print("Validating results against the test set...")
    model.eval()
    total, correct = .0, .0
    with torch.no_grad():
        for data, labels in data_loader:
            if torch.cuda.is_available():
                data.to("cuda")                
            output = model(data)
            _, predicted = torch.max(output.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print(f"Test Accuracy: {100 * correct / total}")

def run() -> None:
    num_processes = 4
    model = Model()
    if torch.cuda.is_available():
        model.to("cuda")

    model.share_memory()

    trainset = build_dataset()
    testset = build_dataset(False)
    
    processes = []
    for rank in range(num_processes):
        data_loader = DataLoader(
            dataset=trainset,
            sampler=DistributedSampler(
                dataset=trainset,
                num_replicas=num_processes,
                rank=rank
            ),
            batch_size=32
        )

        p = mp.Process(target=train, args=(model, data_loader))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()

    test(model, DataLoader(
        dataset=testset,
        batch_size=1000
    ))


if __name__ == "__main__":
    run()

