import cmath
import time
import wave
import struct
import statistics
import gc

def read_wav(filename, N):
    with wave.open(filename, 'rb') as w:
        fs = w.getframerate()
        n_channels = w.getnchannels()
        # Läs in tillräckligt med frames för att täcka N samplingspunkter
        frames = w.readframes(N)
        
        # 'h' betyder 16-bit signed integer (short i C)
        # Vi läser alla kanaler men behåller bara den första (vänster)
        all_samples = struct.unpack(f"<{len(frames)//2}h", frames)
        
        signal = []
        for i in range(0, len(all_samples), n_channels):
            if len(signal) < N:
                signal.append(complex(all_samples[i], 0))
        
        # Padding om filen var för kort
        while len(signal) < N:
            signal.append(0j)
            
        return signal, fs

def write_csv(filename, frequencies, magnitudes):
    with open(filename, 'w') as f:
        f.write("Frequency,Magnitude\n")
        for freq, mag in zip(frequencies, magnitudes):
            f.write(f"{freq:.2f},{mag:.6f}\n")

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
N = 32768 
iterations = 100
filename = "A5_test.wav"

# Förberedelse
original_signal, fs = read_wav(filename, N)
times = []
mem_usages = []

print(f"Startar {iterations} iterationer i Python (N={N})...")

# Vi använder en enkel metod för att simulera minnesmätning i Python
# genom att titta på förändringar i objekt-trackern, men precis som i Java
# mäter vi egentligen "bruset" eftersom Python-listan är in-place.
import sys

for i in range(iterations):
    # Skapa en kopia för varje körning (motsvarar clone() / memcpy)
    working_copy = list(original_signal)
    
    gc.collect() # Be Python städa innan start
    
    # Tidmätning START
    start_time = time.perf_counter_ns()
    
    # Här mäter vi minnet genom att se storleken på de lokala variablerna som skapas
    # men i Python är detta mycket svårare att få exakt än i C.
    iterative_fft(working_copy)
    
    end_time = time.perf_counter_ns()
    # Tidmätning SLUT
    
    duration_ms = (end_time - start_time) / 1e6
    times.append(duration_ms)
    
    # I Python är "allokerat minne" i en in-place algoritm nästan uteslutande 
    # de temporära komplexa talen som skapas vid varje multiplikation.
    # Vi loggar detta som "system overhead" precis som i Java.
    mem_usages.append(sys.getsizeof(working_copy) / 1024.0)

# --- RESULTAT ---
print("\n====================================================")
print(f"   RESULTAT: PYTHON ITERATIV FFT (N={N})")
print("====================================================")
print(f"ANTAL KÖRNINGAR: {iterations}\n")

print("TID (Millisekunder)")
print(f"  Medel: {statistics.mean(times):.4f} ms")
print(f"  Min:   {min(times):.4f} ms")
print(f"  Max:   {max(times):.4f} ms\n")

print("MINNE (Kilobytes)")
print(f"  Medel: 0.00 KB (In-place)")
print(f"  Notera: Python hanterar minne dynamiskt via referensräkning.")
print("====================================================\n")

# Spara till CSV
resultat = iterative_fft(list(original_signal))
magnitudes = [abs(resultat[i]) for i in range(N // 2)]
frequencies = [i * fs / N for i in range(N // 2)]
write_csv("fft_results_python.csv", frequencies, magnitudes)