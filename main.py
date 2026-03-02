import cmath
import time

def iterative_fft(x):
    N = len(x)
    
    # 1. BIT-REVERSAL PERMUTATION
    # Möblerar om indata så att de ligger i den ordning som 
    # "butterfly"-operationerna kräver.
    j = 0
    for i in range(1, N):
        bit = N >> 1
        while j & bit:
            j ^= bit
            bit >>= 1
        j ^= bit
        if i < j:
            x[i], x[j] = x[j], x[i]

    # 2. BUTTERFLY BERÄKNINGAR
    # Vi börjar med små grupper (2 punkter) och slår ihop dem till 
    # större och större grupper (4, 8, 16...)
    length = 2
    while length <= N:
        ang = -2j * cmath.pi / length
        w_m = cmath.exp(ang) # Enhetsrot (twiddle factor)
        
        for i in range(0, N, length):
            w = 1 # Startvärde för rotationen
            for k in range(i, i + length // 2):
                # Här sker själva "Butterfly"-matematiken
                u = x[k]
                t = w * x[k + length // 2]
                
                x[k] = u + t
                x[k + length // 2] = u - t
                
                w *= w_m # Rotera inför nästa par
        length <<= 1 # Dubbla gruppstorleken (samma som length *= 2)
        
    return x

# --- TESTKÖRNING ---
# Skapa en testsignal (N måste vara en potens av 2!)
N_test = 1024 
test_signal = [complex(cmath.sin(2 * cmath.pi * i / 32)) for i in range(N_test)]

start_time = time.perf_counter_ns()
resultat = iterative_fft(test_signal)
end_time = time.perf_counter_ns()

print(f"FFT klar på {(end_time - start_time) / 1e6:.3f} ms")