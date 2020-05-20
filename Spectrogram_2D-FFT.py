#coding:utf-8

#   Figure 1: Show Spectrogram and Select Area for 2D FFT by mouse
#   Figure 2: Show selected Area and Show its 2D FFT.  Select Area to inverse 2D FFT by mouse
#   Figure 3: Show selected Area, Show 2D inverse FFT of selected area in 2D FFT, Show same 2D FFT of Figure 2 
#
#  BPF bank analysis Spectrogram of which feature are
#   BPF's target response is 2nd harmonic level less than -70dB
#   Mel-frequency division
#   Half-wave rectification until a few KHz signal or DC with ripple signal
#   Down sampling to decrease temporal resolution
#   N-th root compression 
#   normalized Gray scale image output

import sys
import os
import argparse
from scipy import signal
from scipy.io.wavfile import read as wavread
from scipy.io.wavfile import write as wavwrite
from matplotlib import pyplot as plt
import matplotlib.patches as patches
from matplotlib.colors import Normalize

from mel  import *
from BPF4 import *
from Compressor1 import *


# Check version
#  Python 3.6.4 on win32 (Windows 10)
#  numpy 1.18.4
#  matplotlib  2.1.1
#  scipy 1.4.1


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
        
        #
        self.boxes_previous = None
        self.boxes_previous_score = None
        
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
        self.fig_unit = np.uint8(np.around( f * 255))
        return self.fig_unit
    
    def conv_gray2RGBgray(self, in_fig, flag_opposite=True ):
        # convert single Gray scale to RGB gray
        self.rgb_fig= np.zeros( (in_fig.shape[0],in_fig.shape[1], 3) )
        
        if flag_opposite:
            for i in range(3):
                self.rgb_fig[:,:,i] = 255 - in_fig
        else:
            for i in range(3):
                self.rgb_fig[:,:,i] =in_fig  
        
        return self.rgb_fig
    
    def plot_image(self, yg=None, LabelOn= True):
        #
        self.fig_image= self.conv_gray2RGBgray( self.trans_gray(self.out1))
        
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
        
        self.img0= ax.imshow( self.fig_image, aspect='auto', origin='lower')
        
        cid =  fig.canvas.mpl_connect('button_press_event', self.onclick)  # mouse
        cid2 = fig.canvas.mpl_connect('key_press_event',   self.onkey)  #  keyboard
        plt.tight_layout()
        plt.show()


    def compute_FFT2D(self,):
    	# compute FFT2D of select area by mouse
        #
        image= self.fig_image_sub[:,:,0].copy()  # get only one element of RGB
        self.fimage = np.fft.fft2(image)
        
        # Replace quadrant
        self.fimage_shift =  np.fft.fftshift(self.fimage)
        
        # Power spectrum calculation
        self.mag = 20*np.log(np.abs(self.fimage_shift))
        
        # convert to single Gray scale
        self.mag_norm= np.clip( self.mag, 0.0, None)  # clip to >= 0
        # Normalize
        self.mag_norm=  self.mag_norm / np.amax( self.mag_norm)  # normalize as max is 1.0
        
        
    def trans2(self, fin, XY):
        #
        # put zero except XY area
        X0=XY[0]
        X1=XY[1]
        Y0=XY[2]
        Y1=XY[3]
        fout=np.zeros_like(fin) 
        print ('fin.shape', fin.shape)
        fout[Y0:Y1,X0:X1]=fin[Y0:Y1,X0:X1]
        
        return fout
        
        
    def compute_IFFT2D(self, XY=None):
    	# compute IFFT2D of select area Fig2  by mouse
        #
        self.fimage_shift2 = self.fimage_shift.copy() 
        
        #
        if XY is not None:
            self.fimage_shift2= self.trans2(self.fimage_shift2, XY)
        
        # back to replace quadrant
        self.fimage2 = np.fft.ifftshift( self.fimage_shift2 )
        
        # inverse FFT2D
        self.image2= np.fft.ifft2(self.fimage2)
        
        # complex to real
        self.mag2 = np.abs(self.image2)
        
        # convert to single Gray scale
        self.mag2_norm= self.conv_gray2RGBgray( self.trans_gray(self.mag2), flag_opposite=False)
        #self.mag2_norm= self.trans_gray(self.mag2)
        
    def show_fig2(self, X0,X1,Y0,Y1, ShowEnable=True, Disp0=False):
        # show select area and FFT2D of it
        #
        X0b = int( self.fig_image.shape[1] *  (X0 / self.img0.get_extent()[1]) )
        X1b = int( self.fig_image.shape[1] *  (X1 / self.img0.get_extent()[1]) )
        Y0b = int( self.fig_image.shape[0] *  (Y0 / self.img0.get_extent()[3]) )
        Y1b = int( self.fig_image.shape[0] *  (Y1 / self.img0.get_extent()[3]) )
        
        self.fig_image_sub= self.fig_image[Y0b:Y1b,X0b:X1b,:]
        self.template_pos=[X0b, X1b, Y0b, Y1b]
        
        if Disp0:
            print ( 'X0b, X1b, Y0b, Y1b', X0b, X1b, Y0b, Y1b)
            #print ( 'fig_image.shape', self.fig_image.shape)
        
        if ShowEnable :
            #
            fig2,  [ax1, ax2] = plt.subplots(2, 1)
            self.fig2=fig2
            self.ax2=ax2
            self.x0b=-1
            self.y0b=-1
            ax1.imshow( self.fig_image_sub, origin='lower')
            
            self.compute_FFT2D()
            self.img2= ax2.imshow( self.mag_norm , cmap = 'gray',  origin='lower')
            
            cid =  fig2.canvas.mpl_connect('button_press_event', self.onclick2)  # mouse
            
            plt.show()
    
    def show_fig3(self, X0,X1,Y0,Y1, ShowEnable=True, Disp0=False):
        # show select area, IFFT2D from FFT2D select portion, and  FFT2D of it
        # 
        X0b = int( self.mag_norm.shape[1] *  (X0 / self.img2.get_extent()[1]) )
        X1b = int( self.mag_norm.shape[1] *  (X1 / self.img2.get_extent()[1]) )
        Y0b = int( self.mag_norm.shape[0] *  (Y0 / self.img2.get_extent()[3]) )
        Y1b = int( self.mag_norm.shape[0] *  (Y1 / self.img2.get_extent()[3]) )
        XYb=[X0b, X1b, Y0b, Y1b]
        if Disp0:
            print ( 'X0b, X1b, Y0b, Y1b', X0b, X1b, Y0b, Y1b)
        
        if ShowEnable :
            #
            fig3,  [ax1, ax2, ax3] = plt.subplots(3, 1)
            ax1.imshow( self.fig_image_sub, origin='lower')
            
            self.compute_IFFT2D( XY=XYb )
            ax2.imshow( self.mag2_norm, origin='lower')
            
            ax3.imshow( self.mag_norm, cmap='gray', origin='lower')
            # show selected area in Fig 2
            [X0b,X1b,Y0b,Y1b]=self.fig2_xy
            r3 = patches.Rectangle(xy=(X0b, Y0b), width= (X1b-X0b), height= (Y1b-Y0b) , ec='r', fill=False)
            ax3.add_patch(r3)
            fig3.canvas.draw()
            fig3.canvas.flush_events()
            plt.show()
    
    
    def add_patch1(self,):
        if self.boxes_previous is not None:
            #print (' try to draw self.boxes_previous')
            for box in self.boxes_previous:
                x, y = box[:2]
                w, h = box[2:] - box[:2] + 1
                self.ax.add_patch(patches.Rectangle((x, y), w, h, linewidth=1, edgecolor='green', fill=None))
    
    def onclick(self,event):
        # for Fig1
        # mouse control
        if 0:
            print ('event.button=%d,  event.x=%d, event.y=%d, event.xdata=%f, event.ydata=%f'%(event.button, event.x, event.y, event.xdata, event.ydata))
        self.x1= self.x0
        self.y1= self.y0
        self.x0= event.xdata # - self.img0.get_extent()[0])
        self.y0= event.ydata # - self.img0.get_extent()[2])
        
        # call compute_fft2d1 via mouse right click
        if event.button == 3:
            self.compute_FFT2D()
            self.add_patch1()
            self.fig.canvas.draw()
            self.fig.canvas.flush_events()
            
            # reset mouse position 
            self.x0=-1
            
            return None
        
        if self.x1 > 0:
            X0= min([self.x0, self.x1])
            X1= max([self.x0, self.x1])
            Y0= min([self.y0, self.y1])
            Y1= max([self.y0, self.y1])
            r = patches.Rectangle(xy=(X0, Y0), width= (X1-X0), height= (Y1-Y0) , ec='r', fill=False)
            self.ax.add_patch(r)
            
            # draw previous select as green
            self.add_patch1()
            
            self.fig.canvas.draw()
            self.fig.canvas.flush_events()
            [p.remove() for p in reversed(self.ax.patches)]
            # to clear next click
            self.x0=-1
            self.y0=-1
            
            self.show_fig2(X0,X1,Y0,Y1, ShowEnable=True, Disp0=False)
            
            
        else:  # clear
            pass
    
    
    def onclick2(self,event):
        # For Fig 2
        # mouse control
        if 0:
            print ('event.button=%d,  event.x=%d, event.y=%d, event.xdata=%f, event.ydata=%f'%(event.button, event.x, event.y, event.xdata, event.ydata))
        self.x1b= self.x0b
        self.y1b= self.y0b
        self.x0b= event.xdata # - self.img0.get_extent()[0])
        self.y0b= event.ydata # - self.img0.get_extent()[2])
        
        # call compute_ifft2d1 via mouse right click
        if event.button == 3:
            #self.compute_FFT2D()
            #self.add_patch1()
            self.fig2.canvas.draw()
            self.fig2.canvas.flush_events()
            
            # reset mouse position 
            self.x0b=-1
            
            return None
            
        if self.x1b > 0:
            X0b= min([self.x0b, self.x1b])
            X1b= max([self.x0b, self.x1b])
            Y0b= min([self.y0b, self.y1b])
            Y1b= max([self.y0b, self.y1b])
            self.fig2_xy=[X0b,X1b,Y0b,Y1b]
            r2 = patches.Rectangle(xy=(X0b, Y0b), width= (X1b-X0b), height= (Y1b-Y0b) , ec='r', fill=False)
            self.ax2.add_patch(r2)
            
            
            self.fig2.canvas.draw()
            self.fig2.canvas.flush_events()
            [p.remove() for p in reversed(self.ax2.patches)]
            # to clear next click
            self.x0b=-1
            self.y0b=-1
            
            self.show_fig3(X0b,X1b,Y0b,Y1b, ShowEnable=True, Disp0=True)  # False)
            
        else:  # clear
            pass
    
    def onkey(self,event):
        # Key control 
        print('you pressed', event.key, event.xdata, event.ydata)
        
        # quit
        if event.key == 'q':
            plt.close(event.canvas.figure)
        
        # call compute_FFT2D
        elif event.key == 'm':
            self.compute_FFT2D()
            self.add_patch1()
            self.fig.canvas.draw()
            self.fig.canvas.flush_events()
                    
            # reset mouse position 
            self.x0=-1
        
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

def save_data(path0, values):
    # check if the dir exist, if not, create new dir
    if not os.path.exists(  os.path.dirname(path0)  ):
        os.makedirs( os.path.dirname(path0)  )
        
    np.save(path0, values)
    print ('save to ', path0)


if __name__ == '__main__':
    #
    parser = argparse.ArgumentParser(description='Show Spectrogram, 2D inverse FFT of red Rectangle in 2D FFT, and 2D FFT of the spectrogram')
    parser.add_argument('--wav_file', '-w', default='wav/sample-for-study-short2.wav', help='wav file name(16bit)')
    args = parser.parse_args()
    
    path0= args.wav_file
    
    # load compared two wav files
    yg0,sr0=load_wav( path0)
    
    # instance
    Ana0= Class_Analysis1(num_band=1024, fmin=40, fmax=8000, sr=sr0)
    
    # process
    yo0= Ana0.compute(yg0)
    
    
    # draw image
    # Specify the area and then search similar area
    Ana0.plot_image() #yg=yg)
    
    """
    # save data as npy
    path1= 'datas/test1.npy'
    print ('fig.unit.shape', Ana0.fig_unit.shape)
    save_data( path1, Ana0.fig_unit)
    """