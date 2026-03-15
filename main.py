import cmath
import time
import wave
import struct
import statistics
import gc
import sys

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
    global current_iteration_memory
    N = len(x)
    
    # Basfall: Om listan bara har ett element
    if N <= 1:
        return x
    
    # --- MINNESMÄTNING ---
    # Slicing x[0::2] skapar en ny lista. Vi mäter storleken på de nya listorna
    # plus den 'combined' lista som skapas längre ner.
    even_list = x[0::2]
    odd_list = x[1::2]
    current_iteration_memory += sys.getsizeof(even_list)
    current_iteration_memory += sys.getsizeof(odd_list)

    # 1. Rekursiva anrop
    even = recursive_fft(even_list)
    odd = recursive_fft(odd_list)

    # 2. Slå ihop resultaten (Butterfly)
    # Vi skapar en tom lista för resultatet av denna nivå
    combined = [0] * N
    current_iteration_memory += sys.getsizeof(combined)

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
iterations = 100
filename = "A5_test.wav"

# 1. Läs ljudfilen (utanför tidtagningen)
test_signal, fs = read_wav(filename, N)

times = []
mem_usages = []

print(f"Startar {iterations} iterationer av Rekursiv FFT i Python (N={N})...")

for i in range(iterations):

    current_iteration_memory = 0
    gc.collect() # Rensa skräp innan mätning
    
    # 2. Tidmätning START
    start_time = time.perf_counter_ns()
    
    resultat = recursive_fft(test_signal)
    
    end_time = time.perf_counter_ns()
    # 3. Tidmätning SLUT

    duration_ms = (end_time - start_time) / 1e6
    times.append(duration_ms)
    mem_usages.append(current_iteration_memory)

# --- RESULTAT ---
print("\n====================================================")
print(f"   RESULTAT: PYTHON REKURSIV FFT (N={N})")
print("====================================================")
print(f"ANTAL KÖRNINGAR: {iterations}\n")

print("TID (Millisekunder)")
print(f"  Medel: {statistics.mean(times):.4f} ms")
print(f"  Min:   {min(times):.4f} ms")
print(f"  Max:   {max(times):.4f} ms\n")

print("MINNE (Kilobytes allokerat via list-skapande)")
print(f"  Medel: {statistics.mean(mem_usages) / 1024.0:.2f} KB")
print(f"  Min:   {min(mem_usages) / 1024.0:.2f} KB")
print(f"  Max:   {max(mem_usages) / 1024.0:.2f} KB")
print("====================================================")

# Spara till CSV
# 4. Beräkna magnitud och frekvenser för export
magnitudes = [abs(resultat[i]) for i in range(N // 2)]
frequencies = [i * fs / N for i in range(N // 2)]
write_csv("recursive_fft_results_python.csv", frequencies, magnitudes)