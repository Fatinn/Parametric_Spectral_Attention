import torch
import torch.nn as nn
import torch.fft
import numpy as np
import os
import matplotlib.pyplot as plt

class RadarConfig:
    START_FREQ = 77.0         
    FREQ_SLOPE = 29.982       
    IDLE_TIME = 40            
    RAMP_END_TIME = 60        
    ADC_START_TIME = 6        
    SAMPLE_RATE = 6000        
    
    NUM_SAMPLES = 512          
    NUM_RX = 4                
    
    NUM_CHIRPS = 160          
    NUM_FRAMES = 40            
    
    FILE_PATH = "adc_data.bin"

def read_dca1000_bin(file_path, config):
    if not os.path.exists(file_path):
        print(f"[Warning] File {file_path} not found. Generating random noise.")
        raw_data = np.random.randn(config.NUM_FRAMES, config.NUM_CHIRPS, config.NUM_RX, config.NUM_SAMPLES) + \
                   1j * np.random.randn(config.NUM_FRAMES, config.NUM_CHIRPS, config.NUM_RX, config.NUM_SAMPLES)
        return raw_data.astype(np.complex64)

    adc_data = np.fromfile(file_path, dtype=np.int16)
    adc_data = adc_data.reshape(-1, 2)
    complex_data = adc_data[:, 0] + 1j * adc_data[:, 1]
    
    expected_points = config.NUM_FRAMES * config.NUM_CHIRPS * config.NUM_RX * config.NUM_SAMPLES

    if complex_data.size != expected_points:
        if complex_data.size > expected_points:
            complex_data = complex_data[:expected_points]
        else:
            complex_data = np.pad(complex_data, (0, expected_points - complex_data.size))

    # Output: [Frames, Chirps, RX, Samples]
    raw_cube = complex_data.reshape(config.NUM_FRAMES, config.NUM_CHIRPS, config.NUM_RX, config.NUM_SAMPLES)
    return raw_cube

class ParametricSpectralAttention(nn.Module):
    def __init__(self, num_samples, num_chirps, init_window='hamming'):
        super(ParametricSpectralAttention, self).__init__()
        
        self.num_samples = num_samples
        self.num_chirps = num_chirps

        # Range Dimension (Amplitude & Phase)
        if init_window == 'hamming':
            init_w_r = torch.hamming_window(num_samples)
        else:
            init_w_r = torch.ones(num_samples)
        self.w_r_amp = nn.Parameter(init_w_r)
        self.w_r_phase = nn.Parameter(torch.zeros(num_samples)) 

        # Doppler Dimension (Amplitude & Phase)
        if init_window == 'hamming':
            init_w_d = torch.hamming_window(num_chirps)
        else:
            init_w_d = torch.ones(num_chirps)
        self.w_d_amp = nn.Parameter(init_w_d)
        self.w_d_phase = nn.Parameter(torch.zeros(num_chirps))

        # Learnable Log-Modulus Parameter
        self.log_epsilon = nn.Parameter(torch.tensor(1e-5))

        # Instance Normalization
        self.norm = nn.InstanceNorm2d(num_features=1, affine=True)

    def get_complex_weight(self, amp, phase):
        # Enforce positive amplitude for stability
        A = torch.abs(amp) 
        return torch.polar(A, phase)

    def forward(self, x):
        # Input:  [Batch, Frames, Chirps, RX, Samples] (Complex)
        # Output: [Batch, Frames, RX, Doppler, Range] (Real, Normalized)
        
        # 1. Range Processing (Complex Attention)
        W_range = self.get_complex_weight(self.w_r_amp, self.w_r_phase)
        x_range_att = x * W_range.view(1, 1, 1, 1, -1)
        X_range = torch.fft.fft(x_range_att, dim=-1)

        # 2. Doppler Processing (Complex Attention)
        W_doppler = self.get_complex_weight(self.w_d_amp, self.w_d_phase)
        W_doppler_expanded = W_doppler.view(1, 1, -1, 1, 1)
        X_doppler_att = X_range * W_doppler_expanded
        
        X_doppler = torch.fft.fft(X_doppler_att, dim=2)
        X_doppler = torch.fft.fftshift(X_doppler, dim=2)

        # 3. Magnitude & Adaptive Log
        X_mag = torch.abs(X_doppler)
        eps = torch.abs(self.log_epsilon) + 1e-9 
        M_RD = 20 * torch.log10(X_mag + eps)

        # 4. Normalization
        # Permute to [Batch, Frames, RX, Doppler, Range]
        M_RD = M_RD.permute(0, 1, 3, 2, 4) 
        
        B, T, R, D, S = M_RD.shape
        # Flatten to apply InstanceNorm per RD-map
        M_view = M_RD.reshape(-1, 1, D, S) 
        M_norm = self.norm(M_view)
        
        out = M_norm.reshape(B, T, R, D, S)
        return out

if __name__ == "__main__":
    cfg = RadarConfig()
    print(f"--- PSA-Mamba Integration Test ---")
    
    # Simulate loading data
    raw_data_numpy = read_dca1000_bin(cfg.FILE_PATH, cfg)
    raw_tensor = torch.from_numpy(raw_data_numpy).unsqueeze(0) # Add Batch Dim
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    raw_tensor = raw_tensor.to(device)
    
    # Initialize PSA Module
    psa_module = ParametricSpectralAttention(cfg.NUM_SAMPLES, cfg.NUM_CHIRPS).to(device)
    
    with torch.no_grad():
        out_tensor = psa_module(raw_tensor)
    
    print(f"[Success] Output Shape: {out_tensor.shape}")
    print(f"Mean: {out_tensor.mean().item():.3f}, Std: {out_tensor.std().item():.3f}")

    # Visualization
    if device.type == 'cuda':
        vis_data = out_tensor[0, 0, 0, :, :].cpu().numpy()
        w_r_mag = torch.abs(psa_module.w_r_amp).detach().cpu().numpy()
        w_r_phase = psa_module.w_r_phase.detach().cpu().numpy()
    else:
        vis_data = out_tensor[0, 0, 0, :, :].numpy()
        w_r_mag = torch.abs(psa_module.w_r_amp).detach().numpy()
        w_r_phase = psa_module.w_r_phase.detach().numpy()

    plt.figure(figsize=(12, 4))
    plt.subplot(1, 3, 1)
    plt.imshow(vis_data, aspect='auto', cmap='jet', origin='lower')
    plt.title("PSA Output")
    plt.subplot(1, 3, 2)
    plt.plot(w_r_mag, label='Amp')
    plt.title("Learned Amplitude")
    plt.subplot(1, 3, 3)
    plt.plot(w_r_phase, color='red')
    plt.title("Learned Phase")
    plt.tight_layout()
    plt.show()
