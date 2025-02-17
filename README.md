This repo contains the code to implement the TV-SGM 




## Unit of PSD

1. `SGM` class output PSD in abs magnitude (20*log10(PSD) to dB)
2. When using CNN to approximate the SGM, I trained on PSD in dB
3. The real data PSD is in squared magnitude (10*log10(PSD) to dB)
    - I infer this based on code, but not 100% sure 

Anyway, since I standardize PSD via subtracting mean and dividing std across freqs, abs or squared magnitude does not matter

## Others
1. All the input freq in `SGM` class is in Hz
