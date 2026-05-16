import cupy as cp

nocc = 2
nvir = 3
nao = 5

x = cp.random.rand(nocc, nvir)
xpy = x.T # shape (nvir, nocc)
orbv = cp.random.rand(nao, nvir)

try:
    P_I = orbv @ xpy.T @ xpy @ orbv.T
    print("Success")
except Exception as e:
    print(f"Error: {e}")
