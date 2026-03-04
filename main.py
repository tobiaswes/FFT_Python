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

def recursive_fft(x):
    N = len(x)
    
    # Basfall: Om listan bara har ett element
    if N <= 1:
        return x

    # 1. Dela upp i jämna och udda index
    # Python-slicing skapar nya listor här:
    even = recursive_fft(x[0::2])
    odd = recursive_fft(x[1::2])

    # 2. Slå ihop resultaten (Butterfly)
    # Vi skapar en tom lista för resultatet av denna nivå
    combined = [0] * N
    for k in range(N // 2):
        # Twiddle factor
        ang = -2j * cmath.pi * k / N
        w = cmath.exp(ang)
        
        t = w * odd[k]
        
        combined[k] = even[k] + t
        combined[k + N // 2] = even[k] - t
        
    return combined

# --- TESTKÖRNING ---
N = 4096
filename = "A5_test.wav"

# 1. Läs ljudfilen (utanför tidtagningen)
test_signal, fs = read_wav(filename, N)

# 2. Tidmätning START
start_time = time.perf_counter_ns()

resultat = recursive_fft(test_signal)

end_time = time.perf_counter_ns()
# 3. Tidmätning SLUT

duration_ms = (end_time - start_time) / 1e6
print(f"Python FFT (N={N}) klar på {duration_ms:.4f} ms")

# 4. Beräkna magnitud och frekvenser för export
magnitudes = [abs(resultat[i]) for i in range(N // 2)]
frequencies = [i * fs / N for i in range(N // 2)]

# 5. Spara till CSV
write_csv("recursive_fft_results_python.csv", frequencies, magnitudes)