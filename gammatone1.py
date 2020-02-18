#coding:utf-8

#
# a class of gammatone (gammachirp) FIR filter
# filtering uses scipy overlap add convolution

import sys
import numpy as np
from matplotlib import pyplot as plt
from scipy import signal   # scipy > 1.14.1


# Check version
#  Python 3.6.4 on win32 (Windows 10)
#  numpy 1.14.0 
#  matplotlib  2.1.1
#  scipy 1.4.1


class Class_Gammatone(object):
    def __init__(self, fc=1000, sampling_rate=48000, gammachirp=False):
        # initalize
        self.sr= sampling_rate
        self.fc= fc # center frequency by unit is [Hz]
        self.ERB_width, self.ERB_rate = self.ERB_N( self.fc )
        print ('fc, ERB_width', self.fc, self.ERB_width)
        self.N= 4  #  4th order
        self.a= 1.0
        self.b= 1.019
        self.phi= 0.0
        self.c= 1.0  # only use for gammachirp
        self.gammachirp= gammachirp
        # self.bz   FIR coefficient
        # self.nth  FIR length
        _,_= self.get_gt()
    
    def ERB_N(self, fin):
        # Equivalent Rectangular Bandwidth
        return 24.7 * ( 4.37 * fin / 1000. + 1.0), 21.4 * np.log10( 4.37 * fin / 1000. + 1.0)
        
    def gt(self, t0): # gammatone
        return self.a * np.power( t0, self.N-1) * np.exp( -2.0 * np.pi * self.b * self.ERB_width * t0) * np.cos( 2.0 * np.pi * self.fc * t0 + self.phi)
        
    def gt_chirp(self, t0): # gammachirp
        return self.a * np.power( t0, self.N-1) * np.exp( -2.0 * np.pi * self.b * self.ERB_width * t0) * np.cos( 2.0 * np.pi * self.fc * t0 +  self.c * np.log(t0) + self.phi)

    def get_gt(self,nsec=0.1, neffective=1E-12):
        # compute some duration
        tlist= np.arange(0.0, (1000.0 / self.fc) * nsec, 1.0/self.sr)
        if self.gammachirp :
            tlist= tlist[1:]
            print ('warning: t is from 1/sampling rate, due to log(0.0) is error')
            self.gt_nsec= self.gt_chirp( tlist)
        else:
            self.gt_nsec= self.gt( tlist)
        
        self.a= 1.0 / np.sum( np.abs(self.gt_nsec))  # fc gain is not 0dB(1.0) !  a few small.
        self.gt_nsec = self.a * self.gt_nsec
        
        index0= np.where( np.abs( np.diff(self.gt_nsec)) <  neffective)
        self.nth= index0[0][0]
        print ('duration', self.nth)
        self.bz= self.gt_nsec[0:self.nth]  # get only effective duration
        
        return self.bz, self.nth

    def filtering(self, xin):
        # filtering process via overlap add convolution, using scipy > 1.4.1
        return signal.oaconvolve(xin, self.bz, axes=0)[0: - self.nth]

    def gt_show(self,):
        # show gammatone (gammachirp) waveform
        plt.xlabel('step (1/sampling rate)')
        plt.ylabel('value')
        if self.gammachirp:
            plt.title('gammachirp')
        else:
            plt.title('gammatone')
        plt.plot( self.gt_nsec)
        plt.grid()
        plt.axis('tight')
        plt.show()
    
    def f_show(self, worN=1024, log_scale= True):
        # show frequency response, using scipy
        wlist, fres = signal.freqz(self.bz, a=1, worN=worN)
        
        fig = plt.figure()
        ax1 = fig.add_subplot(111)
        flist = wlist / ((2.0 * np.pi) / self.sr)
        plt.title('frequency response')
        ax1 = fig.add_subplot(111)
        
        if log_scale :
            plt.semilogx(flist, 20 * np.log10(abs(fres)), 'b')  # plt.plot(flist, 20 * np.log10(abs(fres)), 'b')
        else:
            plt.plot(flist, 20 * np.log10(abs(fres)), 'b')
        plt.ylabel('Amplitude [dB]', color='b')
        plt.xlabel('Frequency [Hz]')
        
        ax2 = ax1.twinx()
        angles = np.unwrap(np.angle(fres))
        angles = angles / ((2.0 * np.pi) / 360.0)
        
        if log_scale:
            plt.semilogx(flist, angles, 'g')  # plt.plot(flist, angles, 'g')
        else:
            plt.plot(flist, angles, 'g') 
        plt.ylabel('Angle(deg)', color='g')
        plt.grid()
        plt.axis('tight')
        plt.show()
        
    def wav_show(self,y1,y2=None, y3=None):
    	# draw wavform
        plt.figure()
        plt.subplot(311)
        plt.xlabel('time step')
        plt.ylabel('amplitude')
        tlist= np.arange( len(y1) ) * (1 /self.sr)
        plt.plot( tlist, y1)
        
        if y2 is not None:
            plt.subplot(312)
            plt.xlabel('time step')
            plt.ylabel('amplitude')
            tlist= np.arange( len(y2) ) * (1 /self.sr)
            plt.plot( tlist, y2)
        
        if y3 is not None:
            plt.subplot(313)
            plt.xlabel('time step')
            plt.ylabel('amplitude')
            tlist= np.arange( len(y3) ) * (1 /self.sr)
            plt.plot( tlist, y3)
        
        plt.grid()
        plt.axis('tight')
        plt.show()

if __name__ == '__main__':
    
    from scipy import signal
    from scipy.io.wavfile import read as wavread
    # instance
    gt=Class_Gammatone(fc=2000, sampling_rate=44100, gammachirp=False)
    
    # show gammatone waveform
    gt.gt_show()
    # draw frequecny response, using scipy
    gt.f_show()
    
    
    # load a sample wav to filter test
    #path0='wav/1KHz-10dB_44100Hz_400msec.wav'
    path0='wav/1KHz-10dB_44100Hz_400ms-TwoTube_stereo.wav'
    #path0='wav/Mix_400Hz1KHz8KHz-10dB_44100Hz_400msec_MONO.wav'
    try:
        sr, y = wavread(path0)
    except:
        print ('error: wavread ')
        sys.exit()
    else:
        yg= y / (2 ** 15)
        if yg.ndim == 2:  # if stereo
            yg= np.average(yg, axis=1)
        print ('sampling rate ', sr)
        print ('y.shape', yg.shape)
    # filtering 
    y2=gt.filtering( yg) 
    
    # compare input and filter output
    gt.wav_show(yg,y2)
    

