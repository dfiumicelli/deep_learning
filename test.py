import torch

# Verifica la versione di PyTorch
print(f"Versione di PyTorch: {torch.__version__}")

# Verifica se CUDA (GPU) è disponibile
cuda_disponibile = torch.cuda.is_available()
print(f"GPU/CUDA disponibile: {cuda_disponibile}")

if cuda_disponibile:
    print(f"Nome della GPU: {torch.cuda.get_device_name(0)}")

# Test rapido di creazione ed esecuzione di un tensore
x = torch.rand(3, 3)
print("\nTensore di test (3x3):")
print(x)

# Sposta il tensore sulla GPU se disponibile
device = torch.device("cuda" if cuda_disponibile else "cpu")
x_device = x.to(device)
print(f"\nTensore creato con successo su: {x_device.device}")