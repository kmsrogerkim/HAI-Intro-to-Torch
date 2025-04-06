import torch
import torch.nn as nn
from torchvision import transforms
import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
latent_dim = 100

class Generator(nn.Module):
  def __init__(self):
    super(Generator, self).__init__()

    # Define the Generator's architecture
    self.model = nn.Sequential(
        nn.Linear(latent_dim, 256),
        nn.LeakyReLU(0.2),
        nn.Linear(256, 512),
        nn.LeakyReLU(0.2),
        nn.Linear(512, 1024),
        nn.LeakyReLU(0.2),
        nn.Linear(1024, 28*28),
        nn.Tanh()
    )

  def forward(self, x):
    return self.model(x)

try:
    state_dict = torch.load("./generator.pth")
except:
   state_dict = torch.load("./local/generator.pth")
generator = Generator()
generator.load_state_dict(state_dict)
generator.to(device)

z = torch.randn(16, latent_dim).to(device)
fake_images = generator(z)
fake_images = fake_images.view(fake_images.size(0), 1, 28, 28)
fake_images = (fake_images + 1) / 2  # Rescale images to [0, 1]

# Plotting
fig, axes = plt.subplots(1, 16, figsize=(15, 15))
for ax, img in zip(axes.flatten(), fake_images):
    ax.axis('off')
    ax.set_adjustable('box')
    img = transforms.ToPILImage()(img.cpu().squeeze())
    ax.imshow(img, cmap='gray')
try:
    fig.savefig("./local/generated_images.png", bbox_inches='tight')
except:
    fig.savefig("./generated_images.png", bbox_inches='tight')