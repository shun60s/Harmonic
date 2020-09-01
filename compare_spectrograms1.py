#coding:utf-8

#  Compare matching portion of two spectrograms and show its difference
#
#  BPF bank analysis Spectrogram of which feature are
#   BPF's target response is 2nd harmonic level less than -70dB
#   Mel-frequency division
#   Half-wave rectification until a few KHz signal or DC with ripple signal
#   Down sampling to decrease temporal resolution
#   N-th root compression 
#   normalized Gray scale image output

import sys
import argparse
from scipy import signal
from scipy.io.wavfile import read as wavread
from scipy.io.wavfile import write as wavwrite
from matplotlib import pyplot as plt
import matplotlib.patches as patches
from matplotlib.colors import Normalize
import cv2

from mel  import *
from BPF4 import *
from Compressor1 import *


# Check version
#  Python 3.6.4 on win32 (Windows 10)
#  numpy 1.18.4
#  matplotlib  3.3.1
#  scipy 1.4.1
#  opencv-python (3.4.6)

class Class_Analysis1(object):
    def __init__(self, num_band=1024, fmin=40, fmax=8000, sr=44100, Q=40.0, \
        moving_average_factor=50, down_sample_factor=10, \
        power_index=1/3.5):
        # instance
        # (1) mel frequency list
        self.num_band=num_band
        self.fmin=fmin
        self.fmax=fmax
        self.mel=Class_mel(self.num_band, self.fmin, self.fmax)
        # (2) BPF bank
        self.sr= sr
        self.Q= Q
        self.maf= int(moving_average_factor)
        self.dsf= int(down_sample_factor)
        self.BPF_list=[]
        for flist0 in self.mel.flist:
            bpf=Class_BPFtwice(fc=flist0, Q=self.Q, sampling_rate=self.sr, moving_average_factor=self.maf, down_sample_factor=self.dsf)
            self.BPF_list.append(bpf)
        # (3) compress via power function
        self.power_index= power_index
        self.comp1= Class_Compressor1(power_index= self.power_index)
        
    def compute(self, yg):
        # yg should be mono
        self.dwn_len= int(len(yg)/self.dsf)
        self.out1= np.empty( ( self.num_band, self.dwn_len), dtype=np.float32  )
        
        for i, bpf in enumerate( self.BPF_list ):
            print ('\r fc', bpf.fc, end='')
            self.out1[i]=self.comp1(bpf.filtering2( yg, self.dwn_len))
        
        print ('self.out1.shape', self.out1.shape)
        print ('max', np.amax(self.out1), ' min', np.amin(self.out1))
        
        return self.out1
    
    def compute_mapped(self, yg):
        # yg should be mono
        #
        # decompose via BPF, and compose from each BPF output
        #
        self.map_out1= np.empty( ( self.num_band, len(yg)), dtype=np.float32  )
        
        for i, bpf in enumerate( self.BPF_list ):
            print ('\r fc', bpf.fc, end='')
            self.map_out1[i]= bpf.filtering( yg )
        
        print ('self.map_out1.shape', self.map_out1.shape)
        print ('max', np.amax(self.map_out1), ' min', np.amin(self.map_out1))
        
        self.map_sum_out1= np.sum( self.map_out1, axis=0)
        print ('self.map_sum_out1.shape', self.map_sum_out1.shape)
        print ('max', np.amax(self.map_sum_out1), ' min', np.amin(self.map_sum_out1))
        
        return self.map_sum_out1


    def trans_gray(self, indata0 ):
        # in_data0 dimension should be 2 zi-gen
        # convert to single Gray scale
        f= np.clip( indata0, 0.0, None)  # clip to >= 0
        # Normalize to [0, 255]
        f=  f / np.amax(f)  # normalize as max is 1.0
        fig_unit = np.uint8(np.around( f * 255))
        return fig_unit
    
    def conv_gray2RGBgray(self, in_fig ):
        # convert single Gray scale to RGB gray
        rgb_fig= np.zeros( (in_fig.shape[0],in_fig.shape[1], 3) )
        
        for i in range(3):
            rgb_fig[:,:,i] = 255 - in_fig
        
        return rgb_fig
    
    def conv_int255(self, in_fig):
        # matplotllib imshow x format was changed from version 2.x to version 3.x
        if 1:  # matplotlib > 3.x
            return np.array(np.abs(in_fig - 255), np.int)
        else:  # matplotlib = 2.x
            return in_fig
    
    def plot_image(self, yg=None, LabelOn= True, template=None ):
        #
        self.fig_image= self.conv_gray2RGBgray( self.trans_gray(self.out1))
        
        # When template is defined, search match area
        if template is not None:
            self.match_template( template = template )
            self.show_comparison()
            return None
        # 
        if yg is not None:
            fig,  [ax0, ax] = plt.subplots(2, 1)
            ax0.plot(yg)
            ax0.set_xlim(0, len(yg))
        else:
            fig, ax = plt.subplots()
        
        # use for mouse click and draw
        self.fig= fig
        self.ax= ax
        self.x0=-1
        self.y0=-1
        
        if LabelOn:
            ax.set_title('BPF bank analysis Spectrogram')
            ax.set_xlabel('time step [sec]')
            ax.set_ylabel('frequecny [Hz]')
        
        # draw time value
        self.xlen=self.fig_image.shape[1]
        
        if LabelOn:
            slen=self.xlen / ( self.sr/ self.dsf)
            char_slen=str( int(slen*1000) / 1000) # ms
            char_slen2=str( int((slen/2)*1000) / 1000) # ms
            ax.set_xticks([0,int(self.xlen/2)-1, self.xlen-1])
            ax.set_xticklabels(['0', char_slen2, char_slen])
        
        # draw frequecny value
        self.ylen=self.fig_image.shape[0]
        
        if LabelOn:
            flens=[self.fmin, 300, 1000, 3000,  self.fmax]
            # flens=[self.fmin, 300, 600, 1000, 1400, 2000, 3000,  self.fmax] # forMix_400Hz1KHz-10dB_44100Hz_400msec_TwoTube_mono.wav
            yflens,char_flens= self.mel.get_postion( flens)
            ax.set_yticks( yflens )
            ax.set_yticklabels( char_flens)
        
        self.img0= ax.imshow( self.conv_int255(self.fig_image), aspect='auto', origin='lower')
        
        cid =  fig.canvas.mpl_connect('button_press_event', self.onclick)  # mouse
        cid2 = fig.canvas.mpl_connect('key_press_event',   self.onkey)  #  keyboard
        plt.tight_layout()
        plt.show()


    def show_comparison(self,):
        fig,  [ax0, ax1] = plt.subplots(1, 2)
        ax0.imshow( self.conv_int255(self.fig_image_template), origin='lower')
        ax1.imshow( self.conv_int255(self.fig_image_match),    origin='lower')
        ax0.set_title('fig_image_template')
        ax1.set_title('fig_image_match')
        plt.tight_layout()
        #plt.show()
        
        
        diff0 = self.fig_image_template - self.fig_image_match
        """
        diff0p= np.where( diff0 < 0, 0, diff)
        diff0m= np.where( diff0 > 0, 0, diff)
        """
        contrast_adjust=2 # adjust constrast of difference.  2 is tetative value
        diff0 = contrast_adjust * diff0  + 128
        diff0x = diff0[:,:,0] / 255
        diff0xc = np.clip( diff0x, 0., 255.)
        fig,  [ax0, ax1] = plt.subplots(1, 2)
        # seismic and bwr are center value is white 
        ax1.imshow( diff0xc, aspect='auto', origin='lower', cmap=plt.cm.seismic, norm=Normalize(vmin=0, vmax=1))  # plt.cm.bwr, #plt.cm.jet )
        ax0.imshow( self.conv_int255(self.fig_image_match),  aspect='auto',  origin='lower')
        ax1.set_title('diff (negative-blue, positive-red)')
        ax0.set_title('fig_image_match')
        plt.tight_layout()
        plt.show()
        
        
    def show_fig2(self, X0,X1,Y0,Y1, ShowEnable=True, Disp0=False):
        #
        X0b = int( self.fig_image.shape[1] *  (X0 / self.img0.get_extent()[1]) )
        X1b = int( self.fig_image.shape[1] *  (X1 / self.img0.get_extent()[1]) )
        Y0b = int( self.fig_image.shape[0] *  (Y0 / self.img0.get_extent()[3]) )
        Y1b = int( self.fig_image.shape[0] *  (Y1 / self.img0.get_extent()[3]) )
        
        self.fig_image_sub= self.fig_image[Y0b:Y1b,X0b:X1b,:]
        
        if Disp0:
            print ( 'X0b, X1b, Y0b, Y1b', X0b, X1b, Y0b, Y1b)
            #print ( 'fig_image.shape', self.fig_image.shape)
        
        if ShowEnable :
            plt.figure()
            plt.imshow( self.conv_int255(self.fig_image_sub), origin='lower')
            plt.show()
        
    def match_template(self, template=None, Disp0=False):
        image= np.array( self.fig_image, dtype=np.uint8)
        if template is None:
            template= np.array( self.fig_image_sub, dtype=np.uint8)
            self.fig_image_template= self.fig_image_sub.copy()
        else:
            self.fig_image_template= template.copy()
            template= np.array( template, dtype=np.uint8)
        
        result = cv2.matchTemplate(image, template, cv2.TM_CCORR_NORMED)
        self.minVal, self.maxVal, self.minLoc, self.maxLoc = cv2.minMaxLoc(result)
        
        if Disp0:
            print ('minVal, maxVal, minLoc, maxLoc', self.minVal, self.maxVal, self.minLoc, self.maxLoc)
        
        X0c= self.maxLoc[0]
        Y0c= self.maxLoc[1]
        X1c= X0c + template.shape[1]
        Y1c= Y0c + template.shape[0]
        self.fig_image_match= self.fig_image[Y0c:Y1c,X0c:X1c,:]
        
        if Disp0:
           print ( 'X0c, X1c, Y0c, Y1c', X0c, X1c, Y0c, Y1c)
           #print ( 'shapes', template.shape, self.fig_image_match.shape)
    
    def onclick(self,event):
        # mouse control
        if 0:
            print ('event.button=%d,  event.x=%d, event.y=%d, event.xdata=%f, event.ydata=%f'%(event.button, event.x, event.y, event.xdata, event.ydata))
        self.x1= self.x0
        self.y1= self.y0
        self.x0= event.xdata # - self.img0.get_extent()[0])
        self.y0= event.ydata # - self.img0.get_extent()[2])
        
        if self.x1 > 0:
            X0= min([self.x0, self.x1])
            X1= max([self.x0, self.x1])
            Y0= min([self.y0, self.y1])
            Y1= max([self.y0, self.y1])
            r = patches.Rectangle(xy=(X0, Y0), width= (X1-X0), height= (Y1-Y0) , ec='r', fill=False)
            self.ax.add_patch(r)
            self.fig.canvas.draw()
            self.fig.canvas.flush_events()
            [p.remove() for p in reversed(self.ax.patches)]
            # to clear next click
            self.x0=-1
            self.y0=-1
            
            self.show_fig2(X0,X1,Y0,Y1, ShowEnable=False)


        else:  # clear
            self.fig.canvas.draw()
            self.fig.canvas.flush_events()
            
    def onkey(self,event):
        # Key control 
        print('you pressed', event.key, event.xdata, event.ydata)
        
        # quit
        if event.key == 'q':
            plt.close(event.canvas.figure)
        
        # call match_template
        elif event.key == 'm':
            self.match_template()
        
        
        sys.stdout.flush()
    
    
    
def load_wav( path0):
    # return 
    #        yg: wav data (mono) 
    #        sr: sampling rate
    try:
        sr, y = wavread(path0)
    except:
        print ('error: wavread ', path0)
        sys.exit()
    else:
        yg= y / (2 ** 15)
        if yg.ndim == 2:  # if stereo
            yg= np.average(yg, axis=1)
    
    print ('file ', path0)
    print ('sampling rate ', sr)
    print ('length ', len(yg))
    return yg,sr

def save_wav( path0, data, sr=44100):
    #
    print ('file ', path0)
    
    amplitude = np.iinfo(np.int16).max
    max_data = np.amax(np.abs(data))  # normalize, max level is 16bit full bit
    if max_data <  (1.0 / amplitude):
        max_data=1.0
    
    try:
        wavwrite(path0, sr, np.array( (amplitude / max_data) * data , dtype=np.int16))
    except:
        print ('error: wavwrite ', path0)
        sys.exit()



if __name__ == '__main__':
    #
    parser = argparse.ArgumentParser(description='Compare matching portion of two spectrograms and show its difference')
    parser.add_argument('--wav_file_template', '-t', default='wav/sample-for-study-short2.wav', help='template wav file name(16bit)')
    parser.add_argument('--wav_file_matching', '-m', default='wav/sample-for-study-short2-output-rtwdf.wav', help='matching wav file name(16bit)')
    args = parser.parse_args()
    
    path0= args.wav_file_template
    path1= args.wav_file_matching
    
    # load compared two wav files
    yg0,sr0=load_wav( path0)
    yg1,sr1=load_wav( path1)
    
    # instance
    Ana0= Class_Analysis1(num_band=1024, fmin=40, fmax=8000, sr=sr0)
    Ana1= Class_Analysis1(num_band=1024, fmin=40, fmax=8000, sr=sr1)
    
    # process
    yo0= Ana0.compute(yg0)
    yo1= Ana1.compute(yg1)
    
    # draw image
    Ana0.plot_image() #yg=yg)
    Ana1.plot_image(template = Ana0.fig_image_sub)
    
    
