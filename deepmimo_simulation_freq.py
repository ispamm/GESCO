# deepmimo_simulation.py
# ------------------------------------------------------------
# Trace-driven channel corruption for one-hot semantic maps
# using DeepMIMO-v3     (frequency-domain model, no ISI)

import torch, random, numpy as np, DeepMIMOv3 as dm

# ---------- OFDM PARAMETERS (match your model) --------------
N_FFT   = 256                 # sub-carriers
CP_LEN  = 32                  # cyclic prefix
QPSK_MAP = torch.tensor([ 1+1j,  1-1j, -1+1j, -1-1j],
                        dtype=torch.cfloat) / np.sqrt(2)

# ---------- 1. DeepMIMO dataset -----------------------------
p = dm.default_params()
p['scenario']      = 'O1_60'      # urban canyon @ 60 GHz
p['active_BS']     = [1]            # one BS
p['active_user_first'] = 1
p['active_user_last']  = 2000
p['num_paths']     = 15           # strongest taps
p['enable_BS2BS']  = 0
deepmimo_dataset = dm.generate_data(p)   # list of BS dicts

# ---------- 2. Helper: random measured FRF (256,)
def random_Hf(device='cpu'):
    """Draw one random user & return the BS-to-UE freq-resp (N_FFT,)"""
    bs = random.choice(deepmimo_dataset)               # dict for one BS
    ch_list = bs['user']['channel']                    # list over UEs
    ue   = random.randrange(len(ch_list))
    # shape: (N_rx, N_tx, N_sub)  – take scalar 1×1 link, all sub-carriers
    Hf_np = ch_list[ue][0, 0, :]
    return torch.as_tensor(Hf_np, dtype=torch.cfloat, device=device)

# ---------- 3. Modulation helpers ----------------------------
def qpsk_mod(bits):                                    # bits (B, K)
    b = bits.to(torch.int64).reshape(bits.size(0), -1, 2)
    sym_idx = (b[...,0] << 1) | b[...,1]               # 2b -> {0…3}
    return QPSK_MAP.to(bits.device)[sym_idx]           # (B, Ns) complex

def ofdm_pack(sym):                                    # (B, Ns)
    """Insert CP & serialize to waveform"""
    sym = sym.reshape(sym.size(0), -1, N_FFT)          # (..., 256)
    time = torch.fft.ifft(sym, n=N_FFT, norm='ortho')
    cp   = time[..., -CP_LEN:]
    return torch.cat([cp, time], dim=-1).reshape(sym.size(0), -1)

# ---------- 4. Main corruption function ----------------------
def deepmimo_corrupt(one_hot, EbN0_dB=10.0):
    """
    one_hot: (B,C,H,W) float 0/1
    returns : noisy maps with identical shape / dtype
    """
    B, C, H, W = one_hot.shape
    bits = one_hot.flatten(1).to(torch.uint8)          # (B, CHW)
    tx_sym = qpsk_mod(bits)                            # QPSK

    # --- build OFDM waveform (with CP) ------------------------
    tx_wave = ofdm_pack(tx_sym)                        # (B, T)
    # reshape back into symbols & drop CP for freq-domain view
    tx_sym_f = torch.fft.fft(
        tx_wave.view(B, -1, N_FFT+CP_LEN)[..., CP_LEN:],
        n=N_FFT, norm='ortho')                         # (B, S, 256)

    # --- apply measured channel per batch --------------------
    H_f = torch.stack([random_Hf(tx_wave.device) for _ in range(B)])  # (B,256)
    rx_f = tx_sym_f * H_f.unsqueeze(1)                 # broadcast

    # --- AWGN (per carrier, complex) -------------------------
    Es = rx_f.abs().pow(2).mean()                      # symbol energy
    N0 = Es / (2 * 10**(EbN0_dB/10))
    noise = (torch.randn_like(rx_f)+1j*torch.randn_like(rx_f)) \
            * torch.sqrt(N0/2)
    rx_f += noise

    # --- zero-forcing equaliser ------------------------------
    sym_hat = rx_f / (H_f.unsqueeze(1) + 1e-7)

    # --- hard QPSK decision -> bits --------------------------
    bits_hat0 = (sym_hat.real < 0).to(torch.uint8)
    bits_hat1 = (sym_hat.imag < 0).to(torch.uint8)
    bits_hat = torch.stack([bits_hat0, bits_hat1], dim=-1) \
                     .reshape(B, -1)

    # --- reshape back to maps (float for diffusion model) ----
    maps_hat = bits_hat.view(B, C, H, W).to(torch.float32)
    return maps_hat
