# Install dependencies if needed (run once in Colab)
!pip install torch torchvision stable-baselines3 gymnasium numpy pandas

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv
import pandas as pd
import os
import torch.quantization as quantization
import warnings

# Suppress deprecation warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# Step 1: Simulate ML-KEM Key Generation Traces
# Simplified simulation: Traces are 100-dim vectors (e.g., NTT cycles, sampling times)
# Params: n (dim), q (modulus), sigma (Gaussian std)
# Latency ~ O(n log n) for NTT + sampling cost
def simulate_key_gen(n, q, sigma, security_level):
    # Simulated compute: NTT (FFT-like) + Gaussian sampling
    ntt_time = n * np.log2(n) * 0.01  # ms, arbitrary scaling
    sampling_time = sigma * n * 0.005
    total_latency = ntt_time + sampling_time
    # Security: Simplified bit-security estimate (NIST baselines: 128,192,256 for levels 1,3,5)
    bit_security = min(128 + 64*(security_level-1), np.log2(q) * n / sigma)
    # Trace: High-dim vector (e.g., per-poly coeffs timings)
    trace = np.random.normal(total_latency, sigma, 100)  # 100-dim synthetic trace
    return trace, total_latency, bit_security

# Generate dataset for each security level
def generate_traces(num_traces=1000):
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
        super(Autoencoder, self).__init__()
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

    def __getstate__(self):
        return self.state_dict()

    def __setstate__(self, state):
        self.__init__()
        self.load_state_dict(state)

# Train Autoencoder (unquantized)
def train_autoencoder(traces, epochs=10, batch_size=32):
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

# Step 3: Gymnasium Environment for PPO
class KeyGenEnv(gym.Env):
    def __init__(self, ae_model, baseline_latencies, baseline_securities):
        super(KeyGenEnv, self).__init__()
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(8,), dtype=np.float32)
        self.action_space = spaces.Box(low=np.array([128, 2000, 1.0], dtype=np.float32), high=np.array([1024, 16384, 5.0], dtype=np.float32), shape=(3,))
        self.ae_model = ae_model
        self.baseline_lat = np.mean(baseline_latencies)
        self.baseline_sec = np.mean(baseline_securities)
        self.current_trace = None

    def reset(self, *, seed=None, options=None):
        if seed is not None:
            np.random.seed(seed)
        self.current_trace = traces[np.random.randint(0, len(traces))]
        with torch.no_grad():
            latent = self.ae_model.encoder(torch.tensor(self.current_trace, dtype=torch.float32).unsqueeze(0)).detach()
        obs = latent.squeeze().numpy().astype(np.float32)
        info = {}
        return obs, info

    def step(self, action):
        n, q, sigma = action
        _, lat, sec = simulate_key_gen(int(n), int(q), sigma, security_level=3)  # Avg level for sim
        reward = -lat / self.baseline_lat  # Penalize latency
        if sec < self.baseline_sec * 0.95:  # Enforce security (5% tolerance)
            reward -= 100  # Heavy penalty
        terminated = False  # Natural end (e.g., goal reached)
        truncated = True   # Time limit or external end
        info = {}
        next_obs, _ = self.reset()
        return next_obs, reward, terminated, truncated, info

# Train PPO on CPU
def train_ppo(ae_model, total_timesteps=10000):
    env_fn = lambda: KeyGenEnv(ae_model, latencies, securities)
    vec_env = SubprocVecEnv([env_fn])
    model = PPO("MlpPolicy", vec_env, verbose=1, device='cpu')
    model.learn(total_timesteps=total_timesteps)
    
    # Quantize policy after training
    policy = model.policy
    policy.eval()
    quantized_policy = quantization.quantize_dynamic(policy, {nn.Linear}, dtype=torch.qint8)
    torch.save(quantized_policy.state_dict(), 'ppo_quantized.pth')
    policy_size = os.path.getsize('ppo_quantized.pth') / 1024
    
    # Quantize AE after training
    ae_model.eval()
    quantized_ae = quantization.quantize_dynamic(ae_model, {nn.Linear}, dtype=torch.qint8)
    torch.save(quantized_ae.state_dict(), 'ae_quantized.pth')
    ae_size = os.path.getsize('ae_quantized.pth') / 1024
    print(f"Quantized PPO model size: {policy_size:.2f} KB (Total framework: {policy_size + ae_size:.2f} KB)")
    
    # Log reward history (access via logger)
    return model, model.logger.get_log_dict()['train/ep_rew_mean'][-1] if 'train/ep_rew_mean' in model.logger.get_log_dict() else None

ppo_model, last_reward = train_ppo(ae_model)

# Step 4: Evaluate and Log Metrics
def evaluate(ppo_model, ae_model, num_episodes=100):
    env = KeyGenEnv(ae_model, latencies, securities)
    opt_latencies, opt_securities = [], []
    for _ in range(num_episodes):
        obs, _ = env.reset()
        action, _ = ppo_model.predict(obs)
        _, lat, sec = simulate_key_gen(*action, security_level=3)
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
        'Model Size (KB)': os.path.getsize('ae_quantized.pth')/1024 + os.path.getsize('ppo_quantized.pth')/1024
    }
    print("Evaluation Metrics:", metrics)
    
    # Log to CSV
    df = pd.DataFrame([metrics])
    df['Last Reward'] = [last_reward]  # Last mean reward
    df.to_csv('metrics.csv', index=False)
    print("Metrics logged to metrics.csv")

evaluate(ppo_model, ae_model)
