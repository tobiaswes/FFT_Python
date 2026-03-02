import cmath
import time
import wave
import struct

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
N = 4096
filename = "A5_test.wav"

# 1. Läs ljudfilen (utanför tidtagningen)
test_signal, fs = read_wav(filename, N)

# 2. Tidmätning START
start_time = time.perf_counter_ns()

resultat = iterative_fft(test_signal)

end_time = time.perf_counter_ns()
# 3. Tidmätning SLUT

duration_ms = (end_time - start_time) / 1e6
print(f"Python FFT (N={N}) klar på {duration_ms:.4f} ms")

# 4. Beräkna magnitud och frekvenser för export
magnitudes = [abs(resultat[i]) for i in range(N // 2)]
frequencies = [i * fs / N for i in range(N // 2)]

# 5. Spara till CSV
write_csv("fft_results_python.csv", frequencies, magnitudes)