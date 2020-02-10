#  BPF bank analysis Spectrogram  

## feature  

- BPF's target response is 2nd harmonic level less than -70dB
- Mel-frequency division
- Half-wave rectification until a few KHz signal or DC with ripple signal
- Down sampling to decrease temporal resolution
- N-th root compression

## description  

- BPF_analysis1.py BPF bank analysis Spectrogram main program
- BPF4.py IIR Band Pass Filter, process twice. Target response is 2nd harmonic level less than -70dB
- Compressor1.py compressor by power(input,1/3.5), 3.5 root compression 
- ema1.py Exponential Moving Average with Half-wave rectification, and smoothing via lpf
- iir1.py iir butterworth filter
- mel.py mel-frequency equal shifted list


## analysis of Tube Amplifier Distortion (PC simulation)  

### Input signal is 1KHz -10dB sin wave  

FFT spectrum of 1KHz-10dB_44100Hz_400ms-TwoTube_stereo.wav. There are 2nd harmonic(2KHz) and 3rd harmonic(3KHz).  
![figure_input1](doc/spectrum_FFT4096Hanning_TwoTube_1KHz_wav.png)  
  
BPF_analysis1.py output. Dark white line in the mid area shows 2nd harmonic(2KHz). Bright white line shows 1KHz.  
Start wave pattern is filter transient response.  
![figure_input2](doc/BPF_analysis1_outputFigure_TwoTube_1KHz_wav.png)  

### Input signal is mix of 400Hz -10dB sin wave and 1KHz -10dB sin wave  

FFT spectrum of Mix_400Hz1KHz-10dB_44100Hz_400msec_TwoTube_mono.wav.  There are 600Hz, 1400Hz, etc other than 800Hz, 2KHz, 1600Hz and 3000Hz.   
![figure_input3](doc/spectrum_FFT4096Hanning_TwoTube_400Hz1KHz_MIX_wav.png)  

BPF_analysis1.py output. Dark white lines show 600Hz, 1400Hz, and 2KHz.  
![figure_input4](doc/BPF_analysis1_outputFigure_TwoTube_400Hz1KHz_MIX_wav.png)  


