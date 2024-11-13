import torch
import torch.nn as nn
import torchvision.transforms as T
from torch.utils.data import DataLoader
from torch.optim import Adam
from torchvision.datasets.mnist import MNIST
import numpy as np
import matplotlib.pyplot as plt
import torch.ao.quantization as quantization

class PatchEmbedding(nn.Module):
    def __init__(self, d_model, img_size, patch_size, n_channels):
        super().__init__()
        self.d_model = d_model               #Dimension of the model
        self.img_size = img_size             #Image size
        self.patch_size = patch_size         #Patch size
        self.n_channels = n_channels         #Number of channels

        self.linear_project = nn.Conv2d(self.n_channels, self.d_model, kernel_size=self.patch_size, stride=self.patch_size)

    def forward(self, x):
        x = self.linear_project(x)
        x = x.flatten(2)
        x = x.transpose(-2, -1)
        return x
    
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_seq_length):
        super().__init__()

        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model))   # Classification token

        pe = torch.zeros(max_seq_length, d_model)                   # Positional encoding

        for pos in range(max_seq_length):
            for i in range(d_model):
                if i%2 == 0:
                    pe[pos][i] = np.sin(pos/(10000 ** (i/d_model)))
                else:
                    pe[pos][i] = np.cos(pos/(10000 ** ((i-1)/d_model)))

        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        tokens_batch = self.cls_token.expand(x.size()[0], -1, -1)
        x = torch.cat((tokens_batch, x), dim=1)
        seq_length = x.size(1)  # New sequence length after adding the CLS token
        pe = self.pe[:, :seq_length, :] 
        x = x + pe
        return x
        
class AttentionHead(nn.Module):
    def __init__(self, d_model, head_size):
        super().__init__()

        self.head_size = head_size

        self.query = nn.Linear(d_model, head_size)
        self.key = nn.Linear(d_model, head_size)
        self.value = nn.Linear(d_model, head_size)
    
    def forward(self, x):
        Q = self.query(x)
        K = self.key(x)
        V = self.value(x)
        
        attention = Q @ K.transpose(-2, -1)
        attention = attention / (self.head_size ** 0.5)
        attention = torch.softmax(attention, dim=-1)
        attention = attention @ V

        return attention
    
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_heads):
        super().__init__()
        self.head_size = d_model // n_heads
        self.W_o = nn.Linear(d_model, d_model)
        self.heads = nn.ModuleList([AttentionHead(d_model, self.head_size) for _ in range(n_heads)])
    
    def forward(self, x):
        out = torch.cat([head(x) for head in self.heads], dim=-1)
        out = self.W_o(out)
        return out

class TransformerEncoder(nn.Module):
    def __init__(self, d_model, n_heads, r_mlp = 4):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads

        #Sub-layer1 - Normalization
        self.ln1 = nn.LayerNorm(d_model)

        #Sub-layer2 - Multi-head attention
        self.mha = MultiHeadAttention(d_model, n_heads)

        #Sub-layer3 - Normalization
        self.ln2 = nn.LayerNorm(d_model)

        #MLP
        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_model * r_mlp),
            nn.GELU(),
            nn.Linear(d_model * r_mlp, d_model) 
        )

    def forward(self, x):
        #Residual connection after sub-layer1
        out = x+ self.mha(self.ln1(x))

        #Residual connection after sub-layer2
        out = out + self.mlp(self.ln2(out))

        return out

class VisionTransformer(nn.Module):
    def __init__ (self, d_model, n_classes, img_size, patch_size, n_channels, n_heads, n_layers):
        super().__init__()

        assert img_size[0] % patch_size[0] == 0 and img_size[1] % patch_size[1] == 0, 'Image dimensions must be divisible by the patch size dimensions'
        assert d_model % n_heads == 0, 'd_model must be divisible by n_heads'

        self.d_model = d_model
        self.n_classes = n_classes
        self.img_size = img_size
        self.patch_size = patch_size
        self.n_channels = n_channels
        self.n_heads = n_heads

        self.n_patches = (self.img_size[0] * self.img_size[1]) // (self.patch_size[0] // self.patch_size[1])
        self.max_seq_length = self.n_patches + 1

        self.patch_embedding = PatchEmbedding(self.d_model, self.img_size, self.patch_size, self.n_channels)
        self.positional_encoding = PositionalEncoding(self.d_model, self.max_seq_length)
        self.transformer_encoder = nn.Sequential(*[TransformerEncoder(self.d_model, self.n_heads) for _ in range(n_layers)])
        
        # Classification mlp
        self.classifier = nn.Sequential(
            nn.Linear(self.d_model, self.n_classes),
            nn.Softmax(dim=-1)
        )

    def forward(self, images):
        x = self.patch_embedding(images)
        x = self.positional_encoding(x)
        x = self.transformer_encoder(x)
        x = self.classifier(x[:, 0])
        return x

class VisionTransformerQAT(nn.Module):
    def __init__ (self, d_model, n_classes, img_size, patch_size, n_channels, n_heads, n_layers):
        super().__init__()
        self.quant = quantization.QuantStub()
        self.vit = VisionTransformer(d_model, n_classes, img_size, patch_size, n_channels, n_heads, n_layers)
        self.dequant = quantization.DeQuantStub()

    def forward(self, x):
        x = self.quant(x)
        x = self.vit(x)
        x = self.dequant(x)
        return x

d_model = 9
n_classes = 10
img_size = (32,32)
patch_size = (16,16)
n_channels = 1
n_heads = 3
n_layers = 3
batch_size = 128
epochs = 25
alpha = 0.005  

model_qat = VisionTransformerQAT(d_model, n_classes, img_size, patch_size, n_channels, n_heads, n_layers)
model_qat.qconfig = quantization.get_default_qat_qconfig('fbgemm')

quantization.prepare_qat(model_qat, inplace=True)

print("Model prepared for QAT:")
print(model_qat)

transform = T.Compose([
  T.Resize(img_size),
  T.ToTensor()
])

train_set = MNIST(
  root=r"C:\Users\arun2\Downloads\datasets", train=True, download=True, transform=transform
)
test_set = MNIST(
  root=r"C:\Users\arun2\Downloads\datasets", train=False, download=True, transform=transform
)

train_loader = DataLoader(train_set, shuffle=True, batch_size=batch_size)
test_loader = DataLoader(test_set, shuffle=False, batch_size=batch_size)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_qat.to(device)
print("Using device: ", device, f"({torch.cuda.get_device_name(device)})" if torch.cuda.is_available() else "")

transformer = VisionTransformer(d_model, n_classes, img_size, patch_size, n_channels, n_heads, n_layers).to(device)

optimizer = Adam(transformer.parameters(), lr=alpha)
criterion = nn.CrossEntropyLoss()

epoch_losses = []

for epoch in range(epochs):
    model_qat.train()
    training_loss = 0.0
    for i, data in enumerate(train_loader, 0):
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()

        outputs = model_qat(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        training_loss += loss.item()


    avg_epoch_loss = training_loss / len(train_loader)
    epoch_losses.append(avg_epoch_loss)
    print(f'Epoch {epoch + 1}/{epochs} loss: {training_loss  / len(train_loader) :.3f}')

# Plot the training loss
plt.plot(range(1, epochs + 1), epoch_losses, marker='o')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training Loss per Epoch')
plt.grid(True)
plt.show()

model_qat.eval()
quantized_model = quantization.convert(model_qat, inplace=False)
print("Model converted to quantized version for inference.")

correct = 0
total = 0

with torch.no_grad():
  for data in test_loader:
    images, labels = data
    images, labels = images.to(device), labels.to(device)

    outputs = transformer(images)

    _, predicted = torch.max(outputs.data, 1)
    total += labels.size(0)
    correct += (predicted == labels).sum().item()
  print(f'\nModel Accuracy: {100 * correct // total} %')
    