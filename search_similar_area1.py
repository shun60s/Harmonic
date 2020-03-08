#coding:utf-8

#  Search some similar area near the specified area in Spectrogram
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
from nms import *

# Check version
#  Python 3.6.4 on win32 (Windows 10)
#  numpy 1.16.3
#  matplotlib  2.1.1
#  scipy 1.4.1
#  opencv-python (3.4.0.12)

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
        fig_unit = np.uint8(np.around( f * 255))
        return fig_unit
    
    def conv_gray2RGBgray(self, in_fig ):
        # convert single Gray scale to RGB gray
        rgb_fig= np.zeros( (in_fig.shape[0],in_fig.shape[1], 3) )
        
        for i in range(3):
            rgb_fig[:,:,i] = 255 - in_fig
        
        return rgb_fig
    
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
        
        self.img0= ax.imshow( self.fig_image, aspect='auto', origin='lower')
        
        cid =  fig.canvas.mpl_connect('button_press_event', self.onclick)  # mouse
        cid2 = fig.canvas.mpl_connect('key_press_event',   self.onkey)  #  keyboard
        plt.tight_layout()
        plt.show()


    def show_comparison(self,):
        fig,  [ax0, ax1] = plt.subplots(1, 2)
        ax0.imshow( self.fig_image_template, origin='lower')
        ax1.imshow( self.fig_image_match,    origin='lower')
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
        ax0.imshow( self.fig_image_match,  aspect='auto',  origin='lower')
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
        self.template_pos=[X0b, X1b, Y0b, Y1b]
        
        if Disp0:
            print ( 'X0b, X1b, Y0b, Y1b', X0b, X1b, Y0b, Y1b)
            #print ( 'fig_image.shape', self.fig_image.shape)
        
        if ShowEnable :
            plt.figure()
            plt.imshow( self.fig_image_sub, origin='lower')
            plt.show()
        
        
    def vlimit_result(self, positions, delta_x=0.2):  # =0.333):
        # limit to match of time in the range (x-axis)
        w= self.template_pos[1] - self.template_pos[0]
        dw= int(w * delta_x)
        x0bmin = self.template_pos[0] - dw
        x0bmax = self.template_pos[0] + dw
        #print ('self.template_pos', self.template_pos)
        print ('limit to match of time in the range (x-axis) of ', x0bmin, x0bmax)
        limit_pos_index= np.where( (positions[1] >= x0bmin) & (positions[1] <= x0bmax) )
        pos2= np.array(positions)
        vl_positions= [ (pos2[0][limit_pos_index]), (pos2[1][limit_pos_index])]
        
        return vl_positions
        
    def flimit_result(self, positions, delta_y=0.01):  # =0.333):
        # limit to match of frequency  (y-axis) more than ...
        h= self.template_pos[3] - self.template_pos[2]
        dh= int(h * delta_y)
        y0bmin = self.template_pos[2] - dh
        y0bmax = self.template_pos[2] + dh
        #print ('self.template_pos', self.template_pos)
        print ('limit to match of frequency  (y-axis) more than ', y0bmin)
        limit_pos_index= np.where( positions[0] >= y0bmin )
        pos2= np.array(positions)
        fl_positions= [ (pos2[0][limit_pos_index]), (pos2[1][limit_pos_index])]
        
        return fl_positions
        
    def rlimit(self,boxes, score, TOP_number=5):
        # limit to top rank xxx only
        #print('boxes.shape', boxes.shape)  # boxes.shape (1, 4)
        #print('score', score)
        boxes_top=[]
        score_top=[]
        
        for i in range ( min([ boxes.shape[0],TOP_number]) ):
            boxes_top.append( boxes[i])
            score_top.append( score[i])
        
        boxes_top=np.array( boxes_top)
        score_top=np.array( score_top)
        #print('boxes_top.shape', boxes_top.shape)  # boxes.shape (1, 4)
        #print('score_top', score_top)        
        
        if min([ boxes.shape[0],TOP_number]) < boxes.shape[0]:
            print ('limit to top rank only ', TOP_number)
        
        return boxes_top, score_top
    	
        
    def match_template(self, template=None, ratio=0.88, Disp0=False):
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
        
        # save maxVal as fig_image_match
        X0c= self.maxLoc[0]
        Y0c= self.maxLoc[1]
        X1c= X0c + template.shape[1]
        Y1c= Y0c + template.shape[0]
        self.fig_image_match= self.fig_image[Y0c:Y1c,X0c:X1c,:]
        
        if Disp0:
           print ( 'X0c, X1c, Y0c, Y1c', X0c, X1c, Y0c, Y1c)
        
        # overwrite ratio
        #ratio = 0.88
        threshold0 = self.minVal + (self.maxVal - self.minVal) * ratio
        positions = np.where(result >= threshold0)
        
        if 1:  # limit to match of time
            positions= self.vlimit_result( positions)
            
        if 1:  # limit to match of frequency 
            positions= self.flimit_result( positions)
        
        scores = result[tuple(positions)]
        
        # Non Maximum Suppression
        boxes = []
        h, w = template.shape[:2]
        for y, x in zip(*positions):
            boxes.append([x, y, x + w - 1, y + h - 1])
        boxes = np.array(boxes)
        print('boxes.shape', boxes.shape) 
        
        boxes, selected_score = non_max_suppression(boxes, probs=scores, overlapThresh=0.6)
        
        
        # limit to top rank
        boxes, selected_score= self.rlimit( boxes, selected_score)
        
        # stack match result 
        self.boxes_previous = boxes.copy()
        self.boxes_previous_score = selected_score.copy()
        
        
    def add_patch1(self,):
        if self.boxes_previous is not None:
            #print (' try to draw self.boxes_previous')
            for box in self.boxes_previous:
                x, y = box[:2]
                w, h = box[2:] - box[:2] + 1
                self.ax.add_patch(patches.Rectangle((x, y), w, h, linewidth=1, edgecolor='green', fill=None))
        
        
    def onclick(self,event):
        # mouse control
        if 0:
            print ('event.button=%d,  event.x=%d, event.y=%d, event.xdata=%f, event.ydata=%f'%(event.button, event.x, event.y, event.xdata, event.ydata))
        self.x1= self.x0
        self.y1= self.y0
        self.x0= event.xdata # - self.img0.get_extent()[0])
        self.y0= event.ydata # - self.img0.get_extent()[2])
        
        # call match_template via mouse right click
        if event.button == 3:
            self.match_template()
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
            
            self.show_fig2(X0,X1,Y0,Y1, ShowEnable=False)
            
            
        else:  # clear
            pass
            
            
    def onkey(self,event):
        # Key control 
        print('you pressed', event.key, event.xdata, event.ydata)
        
        # quit
        if event.key == 'q':
            plt.close(event.canvas.figure)
        
        # call match_template
        elif event.key == 'm':
            self.match_template()
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



if __name__ == '__main__':
    #
    parser = argparse.ArgumentParser(description='Search some similar area near the specified area in Spectrogram')
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
