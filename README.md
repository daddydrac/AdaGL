# AdaGL Optimization Algorthim for PyTorch
AdaGL is an advanced deep learning optimizer that integrates fractional-order calculus with adaptive learning techniques. Leveraging the Grünwald–Letnikov (G–L) fractional-order derivative, AdaGL captures both long-term gradient trends and short-term variations, enhancing optimization across diverse tasks. It dynamically adjusts learning rates using a step size control coefficient, enabling faster convergence and better generalization. Designed to overcome limitations of traditional optimizers like Adam and SGD, AdaGL excels in avoiding local minima and saddle points while targeting flat minima for robust performance. Tested on tasks like image classification, graph analysis, and language modeling, it achieves higher accuracy and efficiency across deep learning domains.

<strong>The math formula for the AdaGL Algorthim was adapted for PyTorch from this published paper <em>[here](https://link.springer.com/article/10.1007/s11063-024-11571-7)</em>.</strong> 
<br/><em>*Any addtions or corrections are welcome in the form of PR's.</em>

-----------------------------------------

| Use Case                          | Generalization Improvement Over Adam         | Speed-Up (if indicated)                  |
|-----------------------------------|---------------------------------------------|------------------------------------------|
| Image Classification (ResNet34)  | +1.04%                                      | Faster convergence (qualitative).        |
| Image Classification (DenseNet121)| +1.13%                                     | Faster convergence (qualitative).        |
| Node Classification (GCN)        | +0.25% (Pubmed); ~0% (Cora)                | Comparable training times.               |
| Graph Classification (GCN)       | ~0.8–2.5%                                  | Faster convergence (qualitative).        |
| Image Generation (WGAN)          | ~27.56% (FID), ~2.5% (IS)                  | Faster generation quality.               |
| Language Modeling (LSTM)         | +4.45% (1-layer), +5.15% (3-layer)         | Faster convergence to lower PPL.         |

-------------------------------------------

### Usage Example for Image Classification (CNNs)
```
import torch
import torch.nn as nn
import torch.optim as optim

model = torch.nn.Sequential(
    nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1),
    nn.ReLU(),
    nn.MaxPool2d(2),
    nn.Flatten(),
    nn.Linear(16 * 16 * 16, 10)  # Assuming CIFAR-10 with 32x32 images
)
criterion = nn.CrossEntropyLoss()
optimizer = AdaGL(model.parameters(), lr=0.001, alpha=1.5)

# Training loop
for epoch in range(100):
    for inputs, targets in dataloader:  # Assuming a CIFAR-10 dataloader
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

```

### Usage example for node classification (GCNs)
```
import torch
from torch_geometric.nn import GCNConv
from torch_geometric.data import DataLoader

class GCN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, out_channels)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index).relu()
        x = self.conv2(x, edge_index)
        return x

model = GCN(in_channels=34, hidden_channels=16, out_channels=4)  # Example for Cora dataset
criterion = nn.CrossEntropyLoss()
optimizer = AdaGL(model.parameters(), lr=0.01, alpha=1.5)

# Training loop
for epoch in range(100):
    for data in dataloader:  # Assuming a PyG DataLoader
        optimizer.zero_grad()
        outputs = model(data.x, data.edge_index)
        loss = criterion(outputs[data.train_mask], data.y[data.train_mask])
        loss.backward()
        optimizer.step()
```

### Image Generation (WGAN)
```
import torch
import torch.nn as nn

class Generator(nn.Module):
    def __init__(self, latent_dim):
        super(Generator, self).__init__()
        self.fc = nn.Linear(latent_dim, 128)

    def forward(self, z):
        return self.fc(z)

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.fc = nn.Linear(128, 1)

    def forward(self, x):
        return self.fc(x)

G = Generator(latent_dim=100)
D = Discriminator()
criterion = nn.BCELoss()  # Or Wasserstein Loss approximation
optimizer_G = AdaGL(G.parameters(), lr=0.0002, alpha=1.5)
optimizer_D = AdaGL(D.parameters(), lr=0.0002, alpha=1.5)

# Training loop
for epoch in range(100):
    for real_samples in dataloader:
        # Update Discriminator
        optimizer_D.zero_grad()
        z = torch.randn(real_samples.size(0), 100)
        fake_samples = G(z).detach()
        loss_D = criterion(D(real_samples), torch.ones_like(D(real_samples))) + \
                 criterion(D(fake_samples), torch.zeros_like(D(fake_samples)))
        loss_D.backward()
        optimizer_D.step()

        # Update Generator
        optimizer_G.zero_grad()
        z = torch.randn(real_samples.size(0), 100)
        fake_samples = G(z)
        loss_G = criterion(D(fake_samples), torch.ones_like(D(fake_samples)))
        loss_G.backward()
        optimizer_G.step()

```

### Language Modeling (LTSM)
```
import torch
import torch.nn as nn

class LSTMModel(nn.Module):
    def __init__(self, vocab_size, embed_size, hidden_size, num_layers):
        super(LSTMModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, vocab_size)

    def forward(self, x, h):
        x = self.embedding(x)
        out, h = self.lstm(x, h)
        out = self.fc(out)
        return out, h

model = LSTMModel(vocab_size=5000, embed_size=300, hidden_size=512, num_layers=2)
criterion = nn.CrossEntropyLoss()
optimizer = AdaGL(model.parameters(), lr=0.001, alpha=1.5)

# Training loop
h = None
for epoch in range(100):
    for inputs, targets in dataloader:  # Assuming tokenized text data
        optimizer.zero_grad()
        outputs, h = model(inputs, h)
        h = tuple([state.detach() for state in h])  # Detach hidden states
        loss = criterion(outputs.view(-1, 5000), targets.view(-1))  # Reshape for NLL
        loss.backward()
        optimizer.step()
```

### LLM fine-tuning example
 - Domain Adaptation: Fine-tune a general-purpose LLM for a specific domain like healthcare or legal.
 - Instruction Tuning: Fine-tune a model like GPT or LLaMA for instruction-following tasks.
```
from transformers import GPT2LMHeadModel, GPT2Tokenizer, AdamW

model = GPT2LMHeadModel.from_pretrained("gpt2")
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token
criterion = torch.nn.CrossEntropyLoss()

# Use AdaGL instead of AdamW
optimizer = AdaGL(model.parameters(), lr=5e-5, alpha=1.5)

# Dataset and dataloader setup
dataloader = create_dataloader()  # Custom function for loading data

# Fine-tuning loop
model.train()
for epoch in range(3):  # Fine-tune for 3 epochs
    for batch in dataloader:
        inputs = tokenizer(batch['text'], return_tensors="pt", padding=True, truncation=True)
        outputs = model(**inputs, labels=inputs["input_ids"])
        loss = outputs.loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
```

### GNN fine-tuning example
 - Node Classification: Fine-tune on datasets with new node types or attributes.
 - Graph-Level Tasks: Adapt pre-trained GNNs for chemical or molecular property prediction tasks.
```
import torch
from torch_geometric.nn import GCNConv
from torch_geometric.data import DataLoader

class GCN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, out_channels)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index).relu()
        x = self.conv2(x, edge_index)
        return x

model = GCN(in_channels=34, hidden_channels=16, out_channels=2)  # Example for graph classification
criterion = torch.nn.CrossEntropyLoss()
optimizer = AdaGL(model.parameters(), lr=0.01, alpha=1.5)

# Fine-tuning loop
for epoch in range(50):  # Shorter fine-tuning schedule
    for data in dataloader:  # Assuming a PyG DataLoader
        optimizer.zero_grad()
        outputs = model(data.x, data.edge_index)
        loss = criterion(outputs[data.train_mask], data.y[data.train_mask])
        loss.backward()
        optimizer.step()
```

### Fine-tuning computer vision models
 - Pre-trained Models: Use models like ResNet, DenseNet, or Vision Transformers.
 - Loss Functions: Typically, CrossEntropyLoss for classification tasks or MeanSquaredError for regression tasks.
 - AdaGL Benefits: Its adaptive learning rates and global-local gradient tracking improve performance and avoid overfitting.
```
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision import datasets, models
from torch.utils.data import DataLoader

# Define data augmentation and preprocessing
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Load datasets
train_dataset = datasets.ImageFolder(root="path_to_train_data", transform=transform)
val_dataset = datasets.ImageFolder(root="path_to_val_data", transform=transform)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

# Load pre-trained ResNet model
model = models.resnet50(pretrained=True)

# Modify the final fully connected layer to match the new number of classes
num_classes = len(train_dataset.classes)
model.fc = nn.Linear(model.fc.in_features, num_classes)

# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = AdaGL(model.parameters(), lr=1e-3, alpha=1.5)

# Fine-tuning loop
num_epochs = 10
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {total_loss / len(train_loader):.4f}")

    # Validation loop
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = outputs.max(1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)
    
    print(f"Validation Accuracy: {100 * correct / total:.2f}%")
```

### Fine-tuning Vision Transformer (ViT)
```
from transformers import ViTForImageClassification, ViTFeatureExtractor
from torch.optim import AdamW

# Load pre-trained Vision Transformer and feature extractor
feature_extractor = ViTFeatureExtractor.from_pretrained("google/vit-base-patch16-224-in21k")
model = ViTForImageClassification.from_pretrained("google/vit-base-patch16-224-in21k", num_labels=num_classes)

# Define optimizer and loss function
optimizer = AdaGL(model.parameters(), lr=5e-5, alpha=1.5)
criterion = nn.CrossEntropyLoss()

# Transform images using the feature extractor
def transform_fn(batch):
    return feature_extractor(images=batch["image"], return_tensors="pt")

# Fine-tuning loop (similar structure to ResNet example above)
for epoch in range(num_epochs):
    model.train()
    for batch in train_loader:
        inputs = transform_fn(batch)
        labels = batch["labels"]

        optimizer.zero_grad()
        outputs = model(**inputs, labels=labels)
        loss = outputs.loss
        loss.backward()
        optimizer.step()

    # Perform validation as shown in the ResNet example

```

### Fine-Tuning a SQuAD Model
```
import torch
from transformers import BertForQuestionAnswering, BertTokenizer, AdamW
from torch.utils.data import DataLoader, Dataset
from transformers import SquadV2Processor

# Load pre-trained model and tokenizer
model = BertForQuestionAnswering.from_pretrained("bert-base-uncased")
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

# Define the optimizer (Replace AdamW with AdaGL if implemented)
class AdaGL(torch.optim.Optimizer):
    def __init__(self, params, lr=0.001, alpha=1.5):
        defaults = {'lr': lr, 'alpha': alpha}
        super(AdaGL, self).__init__(params, defaults)

    def step(self, closure=None):
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data
                # AdaGL logic for parameter updates
                # (Implement Grünwald–Letnikov fractional derivative-based updates here)
                p.data.add_(-group['lr'], grad)

optimizer = AdaGL(model.parameters(), lr=0.001, alpha=1.5)

# Load and preprocess SQuAD dataset
processor = SquadV2Processor()
train_examples = processor.get_train_examples("./data")
tokenized_data = []

for example in train_examples[:100]:  # Limit examples for demo
    inputs = tokenizer(
        example.question_text,
        example.context_text,
        max_length=384,
        padding="max_length",
        truncation=True,
        return_tensors="pt"
    )
    tokenized_data.append({
        "input_ids": inputs["input_ids"].squeeze(),
        "attention_mask": inputs["attention_mask"].squeeze(),
        "start_positions": torch.tensor(example.start_position_character),
        "end_positions": torch.tensor(example.end_position_character),
    })

class SquadDataset(Dataset):
    def __init__(self, tokenized_data):
        self.data = tokenized_data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

train_loader = DataLoader(SquadDataset(tokenized_data), batch_size=8, shuffle=True)

# Define the training loop
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

for epoch in range(3):  # Training for 3 epochs
    model.train()
    total_loss = 0
    for batch in train_loader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        start_positions = batch["start_positions"].to(device)
        end_positions = batch["end_positions"].to(device)

        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            start_positions=start_positions,
            end_positions=end_positions
        )
        loss = outputs.loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(f"Epoch {epoch + 1}, Loss: {total_loss / len(train_loader)}")

print("Fine-tuning complete!")

```

--------------------------

## Summary of Algorthim
Natural Language Processing (NLP), Graph Neural Networks (GNNs), and Computer Vision (CV) drive innovation across industries by enabling machines to understand text, analyze relational data, and interpret visual content. NLP powers applications like chatbots, language translation, content generation, and healthcare insights. GNNs excel in analyzing interconnected data for tasks such as recommendation systems, drug discovery, fraud detection, and logistics optimization. Computer Vision enables breakthroughs in medical imaging, autonomous vehicles, security, retail, and environmental monitoring. Together, these technologies create synergistic solutions, like multimodal AI systems for autonomous driving or knowledge graphs for healthcare, transforming how we process and use information in a multi-modality format.

------------------------------

## Cite
This implementation uses the AdaGL optimizer, proposed by S. Chen, C. Zhang, and H. Mu in their paper:
"An Adaptive Learning Rate Deep Learning Optimizer Using Long-and Short-Term Gradient Information."
For more information, contact the authors at zclun@bucea.edu.cn, 2102520021007@stu.bucea.edu.cn, hbmu@bjtu.edu.cn.

Source:
https://link.springer.com/article/10.1007/s11063-024-11571-7

