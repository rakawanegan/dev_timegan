import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

from src.model import Generator, Discriminator, Autoencoder


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 1. 秩序を持ったデータの生成 (正弦波にノイズを加えたデータ)
def generate_sine_data(seq_len, num_samples):
    t = np.linspace(0, 100, seq_len)
    data = []
    for _ in range(num_samples):
        amplitude = np.random.uniform(0.8, 1.2)
        frequency = np.random.uniform(0.8, 1.2)
        noise = np.random.normal(0, 0.05, seq_len)
        sine_wave = amplitude * np.sin(frequency * t) + noise
        data.append(sine_wave)
    return np.array(data)


seq_len = 50
num_samples = 1000
data = generate_sine_data(seq_len, num_samples)
data = torch.tensor(data, dtype=torch.float32).to(device)

latent_dim = 6  # 潜在空間の次元
hidden_dim = 24  # 隠れ層のサイズ
num_layers = 3  # LSTMの層数

latent_dim = 6
hidden_dim = 24
seq_len = 50

generator = Generator(latent_dim=latent_dim, hidden_dim=hidden_dim, seq_len=seq_len).to(device)
discriminator = Discriminator(seq_len=seq_len, hidden_dim=hidden_dim).to(device)
autoencoder = Autoencoder(seq_len=seq_len, hidden_dim=hidden_dim, latent_dim=latent_dim).to(device)

g_optimizer = optim.Adam(generator.parameters(), lr=0.0001)
d_optimizer = optim.Adam(discriminator.parameters(), lr=0.0001)
ae_optimizer = optim.Adam(autoencoder.parameters(), lr=0.0001)
criterion = nn.BCELoss()
mse_loss = nn.MSELoss()

epochs = 1000
batch_size = 128

for epoch in range(epochs):
    idx = np.random.randint(0, data.size(0), batch_size)
    real_data = data[idx].to(device)
    
    z = torch.randn(batch_size, latent_dim).to(device)
    gen_data = generator(z)

    d_real = discriminator(real_data)
    d_fake = discriminator(gen_data.detach())
    d_loss_real = criterion(d_real, torch.ones_like(d_real))
    d_loss_fake = criterion(d_fake, torch.zeros_like(d_fake))
    d_loss = (d_loss_real + d_loss_fake) / 2

    d_optimizer.zero_grad()
    d_loss.backward()
    d_optimizer.step()

    d_fake = discriminator(gen_data)
    g_loss = criterion(d_fake, torch.ones_like(d_fake))

    g_optimizer.zero_grad()
    g_loss.backward()
    g_optimizer.step()

    reconstructed_data = autoencoder(real_data)
    ae_loss = mse_loss(reconstructed_data, real_data)

    ae_optimizer.zero_grad()
    ae_loss.backward()
    ae_optimizer.step()

    if epoch % 100 == 0:
        print(f"Epoch [{epoch}/{epochs}], D Loss: {d_loss.item():.4f}, G Loss: {g_loss.item():.4f}, AE Loss: {ae_loss.item():.4f}")

with torch.no_grad():
    z = torch.randn(10, latent_dim).to(device)
    generated_data = generator(z).cpu().numpy()

for i in range(3):
    plt.plot(generated_data[i], label=f'Generated Sample {i+1}')
plt.title("Generated Sine Waves with Noise")
plt.xlabel("Time Step")
plt.ylabel("Value")
plt.legend()
plt.show()
