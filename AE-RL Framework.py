import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torch.distributions import Normal
import numpy as np
import pandas as pd
import os
import torch.quantization as quantization
import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# Step 1: Simulate ML-KEM Key Generation Traces
def simulate_key_gen(n, q, sigma, security_level):
    
    n = max(2, n)  # Avoid log2(1) or less
    ntt_time = n * np.log2(n) * 0.01  # ms, arbitrary scaling
    sampling_time = sigma * n * 0.005
    total_latency = ntt_time + sampling_time
    bit_security = min(128 + 64*(security_level-1), np.log2(q) * n / sigma)
    trace = np.random.normal(total_latency, sigma, 100)  # 100-dim synthetic trace
    return trace, total_latency, bit_security

# Generate dataset for each security level
def generate_traces(num_traces=100):  # Reduced for faster execution
    
    levels = [1, 3, 5]
    params = [(256, 3329, 2.0), (512, 7681, 3.0), (768, 12289, 4.0)]  # NIST-inspired
    all_traces, all_latencies, all_securities = [], [], []
    for level, (n, q, sigma) in zip(levels, params):
        for _ in range(num_traces):
            trace, lat, sec = simulate_key_gen(n, q, sigma, level)
            all_traces.append(trace)
            all_latencies.append(lat)
            all_securities.append(sec)
    return np.array(all_traces), np.array(all_latencies), np.array(all_securities)

traces, latencies, securities = generate_traces()
print(f"Generated {len(traces)} traces. Baseline avg latency: {np.mean(latencies):.2f} ms")

# Step 2: Autoencoder for Trace Compression (100D -> 8D latent)
class Autoencoder(nn.Module):
    
    def __init__(self, input_dim=100, latent_dim=8):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 64), nn.ReLU(),
            nn.Linear(64, 32), nn.ReLU(),
            nn.Linear(32, latent_dim)
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 32), nn.ReLU(),
            nn.Linear(32, 64), nn.ReLU(),
            nn.Linear(64, input_dim)
        )

    def forward(self, x):
        latent = self.encoder(x)
        recon = self.decoder(latent)
        return recon, latent

# Train Autoencoder
def train_autoencoder(traces, epochs=5, batch_size=32):  # Reduced epochs for speed
    
    dataset = TensorDataset(torch.tensor(traces, dtype=torch.float32))
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    model = Autoencoder()
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    for epoch in range(epochs):
        for data in loader:
            inputs = data[0]
            recon, _ = model(inputs)
            loss = criterion(recon, inputs)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item():.4f}")
    
    return model

ae_model = train_autoencoder(traces)

# Precompute latent representations
def precompute_latents(ae_model, traces):
    
    ae_model.eval()
    with torch.no_grad():
        latents = ae_model.encoder(torch.tensor(traces, dtype=torch.float32)).numpy()
    return latents

latents = precompute_latents(ae_model, traces)

# Step 3: Simple REINFORCE Policy Gradient
class Policy(nn.Module):
    
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.fc1 = nn.Linear(state_dim, 64)
        self.fc2 = nn.Linear(64, 32)
        self.mu = nn.Linear(32, action_dim)
        self.log_std = nn.Linear(32, action_dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        mu = self.mu(x)
        log_std = self.log_std(x)
        std = torch.exp(log_std)
        return mu, std

# Parameters
state_dim = 8
action_dim = 3
lr = 0.0003
epochs = 10
batch_size = 32
min_action = torch.tensor([128., 2000., 1.0])
max_action = torch.tensor([1024., 16384., 5.0])

# Initialize policy
policy = Policy(state_dim, action_dim)
optimizer = optim.Adam(policy.parameters(), lr=lr)

baseline_lat = np.mean(latencies)
baseline_sec = np.mean(securities)

last_reward = 0

# Training loop
for epoch in range(epochs):
    
    states = []
    log_probs = []
    rewards = []
    for _ in range(batch_size):
        state_idx = np.random.randint(0, len(latents))
        state = torch.tensor(latents[state_idx], dtype=torch.float32)
        mu, std = policy(state)
        dist = Normal(mu, std)
        action = dist.sample()
        action = torch.clamp(action, min_action, max_action)
        log_prob = dist.log_prob(action).sum()
        
        n, q, sigma = action.detach().numpy()
        _, lat, sec = simulate_key_gen(n, q, sigma, 3)
        reward = -lat / baseline_lat
        if sec < baseline_sec * 0.95:
            reward -= 100
        
        states.append(state)
        log_probs.append(log_prob)
        rewards.append(reward)
    
    # Normalize rewards
    rewards_np = np.array(rewards)
    last_reward = np.mean(rewards_np)
    rewards_norm = (rewards_np - np.mean(rewards_np)) / (np.std(rewards_np) + 1e-8)
    rewards = torch.tensor(rewards_norm, dtype=torch.float32)
    
    # Loss
    log_probs = torch.stack(log_probs)
    loss = - (log_probs * rewards).mean()
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item():.4f}, Avg Reward: {last_reward:.4f}")

# Quantize policy and AE
policy.eval()
quantized_policy = quantization.quantize_dynamic(policy, {nn.Linear}, dtype=torch.qint8)
torch.save(quantized_policy.state_dict(), 'policy_quantized.pth')
policy_size = os.path.getsize('policy_quantized.pth') / 1024

ae_model.eval()
quantized_ae = quantization.quantize_dynamic(ae_model, {nn.Linear}, dtype=torch.qint8)
torch.save(quantized_ae.state_dict(), 'ae_quantized.pth')
ae_size = os.path.getsize('ae_quantized.pth') / 1024

print(f"Quantized Policy size: {policy_size:.2f} KB, AE {ae_size:.2f} KB (Total: {policy_size + ae_size:.2f} KB)")

# Step 4: Evaluate and Log Metrics
def evaluate(policy, num_episodes=100):
    
    opt_latencies, opt_securities = [], []
    for _ in range(num_episodes):
        state_idx = np.random.randint(0, len(latents))
        state = torch.tensor(latents[state_idx], dtype=torch.float32)
        mu, std = policy(state)
        action = mu  # Use mean for evaluation
        action = torch.clamp(action, min_action, max_action)
        n, q, sigma = action.detach().numpy()
        _, lat, sec = simulate_key_gen(n, q, sigma, 3)
        opt_latencies.append(lat)
        opt_securities.append(sec)
    
    baseline_lat = np.mean(latencies)
    opt_lat = np.mean(opt_latencies)
    reduction = (baseline_lat - opt_lat) / baseline_lat * 100
    avg_sec = np.mean(opt_securities)
    
    metrics = {
        'Baseline Latency (ms)': baseline_lat,
        'Optimized Latency (ms)': opt_lat,
        'Latency Reduction (%)': reduction,
        'Avg Security Score (bits)': avg_sec,
        'Model Size (KB)': policy_size + ae_size
    }
    print("Evaluation Metrics:", metrics)
    
    # Log to CSV
    df = pd.DataFrame([metrics])
    df['Last Reward'] = [last_reward]  # Last mean reward
    df.to_csv('metrics.csv', index=False)
    print("Metrics logged to metrics.csv")

evaluate(policy)
