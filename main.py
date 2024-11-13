import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from scipy import signal
from scipy.integrate import solve_ivp

from src.model import Generator, Discriminator, Autoencoder


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# データ生成の関数
def generate_data_with_missing(seq_len, num_samples, func, noise_level, func_params={}, missing_ratio=0.1):
    data = list()
    mask = list()

    for _ in range(num_samples):
        # 関数とパラメータでデータを生成
        values = func(seq_len)
        
        # ノイズの追加
        noise = np.random.normal(0, noise_level, seq_len)
        noisy_values = values + noise
        
        # 欠損の追加
        missing_length = int(missing_ratio * seq_len)
        start_index = np.random.randint(0, seq_len - missing_length + 1)
        noisy_values[start_index:start_index + missing_length] = np.nan
        
        # マスク生成
        mask_wave = (~np.isnan(noisy_values)).astype(int)
        data.append(noisy_values)
        mask.append(mask_wave)

    return np.array(data), np.array(mask)

# 正弦波関数の生成関数
def sine_wave(seq_len, amplitude=1.0, frequency=1.0):
    t = np.linspace(0, 100, seq_len)
    return amplitude * np.sin(frequency * t)

# 三角波の生成関数
def triangle_wave(seq_len, amplitude=1.0, frequency=1.0):
    t = np.linspace(0, 1, seq_len)
    return amplitude * signal.sawtooth(2 * np.pi * frequency * t, 0.5)

# ランダムウォークの生成関数
def random_walk(seq_len, start=0, step_std=1.0):
    steps = np.random.normal(0, step_std, seq_len)
    return np.cumsum(steps) + start

# 減衰正弦波の生成関数
def damped_sine_wave(seq_len, amplitude=1.0, frequency=1.0, damping_factor=0.1):
    t = np.linspace(0, 10, seq_len)
    return amplitude * np.exp(-damping_factor * t) * np.sin(2 * np.pi * frequency * t)

# ローレンツアトラクタの生成関数
def lorenz_attractor(seq_len, sigma=10, beta=8/3, rho=28):
    def lorenz(t, state):
        x, y, z = state
        dxdt = sigma * (y - x)
        dydt = x * (rho - z) - y
        dzdt = x * y - beta * z
        return [dxdt, dydt, dzdt]

    t_span = [0, 1]  # 時間範囲
    t_eval = np.linspace(t_span[0], t_span[1], seq_len)
    initial_state = [1.0, 1.0, 1.0]
    solution = solve_ivp(lorenz, t_span, initial_state, t_eval=t_eval)
    
    return solution.y[0]

# 欠損データの可視化関数
def plot_missing_data(data, mask, completed_data=None, title="Data with Missing Values"):
    plt.figure(figsize=(10, 6))
    
    # 元データのプロット（薄い青）
    plt.plot(data, label="Original Data", color="lightblue", linestyle="-")
    
    # 観測データのプロット（オレンジ色）
    # observed_data = np.where(mask == 1, data, np.nan)
    # plt.plot(observed_data, 'o', label="Observed Data", color='orange', markersize=3)
    
    # 欠損箇所のプロット（赤色の×）
    missing_points = np.where(mask == 0, 0, np.nan)
    plt.plot(missing_points, 's', label="Missing Points", color='red')
    
    # 補完データがあれば、補完後のデータをプロット（緑色の線）
    if completed_data is not None:
        plt.plot(completed_data, label="Completed Data", color="green", linestyle="-")
    
    plt.title(title)
    plt.xlabel("Time Step")
    plt.ylabel("Value")
    plt.legend()
    plt.grid(True)
    plt.savefig("outputs/missing_data.png")
    plt.close()

# 2. 欠損値補完のトレーニング
def train_missing_completion(
        generator, discriminator, autoencoder, data, mask, epochs=10000, batch_size=128, latent_dim=10, per_iter=1000
    ):
    g_optimizer = optim.Adam(generator.parameters(), lr=0.0001)
    d_optimizer = optim.Adam(discriminator.parameters(), lr=0.0001)
    ae_optimizer = optim.Adam(autoencoder.parameters(), lr=0.0001)
    criterion = nn.BCELoss()
    mse_loss = nn.MSELoss()

    data_filled = torch.tensor(data, dtype=torch.float32).to(device)
    mask_tensor = torch.tensor(mask, dtype=torch.float32).to(device)
    data_filled[torch.isnan(data_filled)] = 0

    for epoch in range(epochs + 1):
        idx = np.random.randint(0, data_filled.size(0), batch_size)
        real_data = data_filled[idx].to(device)
        real_mask = mask_tensor[idx].to(device)

        # 生成器のデータ生成
        z = torch.randn(batch_size, latent_dim).to(device)
        gen_data = generator(z)

        # 識別器のトレーニング
        d_real = discriminator(real_data)
        d_fake = discriminator(gen_data.detach())
        d_loss_real = criterion(d_real, torch.ones_like(d_real))
        d_loss_fake = criterion(d_fake, torch.zeros_like(d_fake))
        d_loss = (d_loss_real + d_loss_fake) / 2

        d_optimizer.zero_grad()
        d_loss.backward()
        d_optimizer.step()

        # 生成器のトレーニング
        d_fake = discriminator(gen_data)
        g_loss = criterion(d_fake, torch.ones_like(d_fake))

        g_optimizer.zero_grad()
        g_loss.backward()
        g_optimizer.step()

        # オートエンコーダのトレーニング
        reconstructed_data = autoencoder(real_data)
        ae_loss = mse_loss(reconstructed_data * real_mask, real_data * real_mask)

        ae_optimizer.zero_grad()
        ae_loss.backward()
        ae_optimizer.step()

        # ログと生成データの可視化
        if epoch % per_iter == 0:
            print(f"Epoch [{epoch}/{epochs}], D Loss: {d_loss.item():.4f}, G Loss: {g_loss.item():.4f}, AE Loss: {ae_loss.item():.4f}")

            # 可視化用のデータ準備
            incomplete_data = data[idx[0]]
            incomplete_mask = real_mask[0].cpu().numpy()
            completed_data = autoencoder(real_data[0].unsqueeze(0)).detach().cpu().numpy()[0]

            # 欠損データと補完結果のプロット
            plot_missing_data(
                incomplete_data,
                incomplete_mask,
                completed_data,
                title=f"Epoch {epoch} - Data with Missing Values"
            )

    return generator, discriminator, autoencoder

# 実行例
def main():
    seq_len = 500
    num_samples = 1000
    latent_dim = 10
    hidden_dim = 64
    epochs = 1000
    batch_size = 128
    per_iter = epochs // 10
    missing_ratio = 0.1
    noise_level = 0.01
    simulation_wave = [
        # sine_wave,
        # triangle_wave,
        # random_walk,
        # damped_sine_wave,
        lorenz_attractor,
    ][0]

    # データ生成と欠損マスク生成
    data, mask = generate_data_with_missing(
        seq_len, num_samples, simulation_wave, noise_level,
        missing_ratio=missing_ratio,
    )
    original_data = data.copy()

    generator = Generator(latent_dim=latent_dim, hidden_dim=hidden_dim, seq_len=seq_len).to(device)
    discriminator = Discriminator(seq_len=seq_len, hidden_dim=hidden_dim).to(device)
    autoencoder = Autoencoder(seq_len=seq_len, hidden_dim=hidden_dim, latent_dim=latent_dim).to(device)

    generator, discriminator, autoencoder = train_missing_completion(
        generator,
        discriminator,
        autoencoder,
        original_data,
        mask,
        epochs=epochs,
        batch_size=batch_size,
        latent_dim=latent_dim,
        per_iter=per_iter
    )

    torch.save(
        autoencoder.state_dict(),
        "outputs/autoencoder.pth",
    )

if __name__ == "__main__":
    main()
