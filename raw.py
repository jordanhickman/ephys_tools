import os,glob
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


from jlh_ephys.utils import OE

class Raw:
    def __init__(self, analysis_obj):
        self.mouse = analysis_obj.mouse
        self.date = analysis_obj.date
        self.path = analysis_obj.path
        if analysis_obj.processed:
            self.units = analysis_obj.units

        
    def get_raw(self, probe, band = 'ap'):
    # takes in an open ephys recording object and a probe name and returns the continuous binary object
        recording = OE(self.path)
        if band == 'ap': 
            if str(probe) == 'probeA':
                data = recording.continuous[1]
            elif str(probe) == 'probeB':
                data = recording.continuous[3]
            else:
                data = recording.continuous[5]
            return data

        elif band == 'lfp':
            if str(probe) == 'probeA':
                data = recording.continuous[1]
            elif str(probe) == 'probeB':
                data = recording.continuous[3]
            else:
                data = recording.continuous[5]
            return data
        else: 
            print('You got not bands. Get your paper up.')
            return None
        
    def get_chunk(self, probe, # can use binary_path output directly
              stim_times,
              band = 'ap',
              pre = 100, # time in ms
              post = 500, # time in ms
              chs = np.arange(0,200,1), # channels
              output = 'response' # 'response, 'pre/post', 'all'
              ):
        """
        Takes in a continuous binary object and a list of stimulation times and returns a chunk of the data
        """
        data = self.get_raw(probe = probe, band = band)
        sample_rate = data.metadata['sample_rate']
        pre_samps = int((pre/1000 * sample_rate))
        post_samps = int((post/1000 * sample_rate))
        total_samps = pre_samps + post_samps

        n_chs = len(chs)
        if output == 'response':
            response = np.zeros((np.shape(stim_times)[0],total_samps, len(chs)))
            stim_indices = np.searchsorted(data.timestamps,stim_times)
            for i, stim in enumerate(stim_indices):
                start_index = int(stim - ((pre/1000)*sample_rate))
                end_index = int(stim + ((post/1000)*sample_rate))   
                chunk = data.get_samples(start_sample_index = start_index, end_sample_index = end_index, 
                                    selected_channels = chs)
                chunk = chunk - np.median(chunk, axis = 0 )
                response[i,:,:] = chunk

            return response
        
        elif output == 'pre/post':
            pre_response = np.zeros((np.shape(stim_times)[0],pre_samps, n_chs))
            post_response = np.zeros((np.shape(stim_times)[0],post_samps, n_chs))
            stim_indices = np.searchsorted(data.timestamps,stim_times)
            for i, stim in enumerate(stim_indices):
                start_index = int(stim - ((pre/1000)*sample_rate))
                end_index = int(stim + ((post/1000)*sample_rate))   

                pre_chunk = data.get_samples(start_sample_index = start_index, end_sample_index = stim, 
                                    selected_channels = np.arange(0,n_chs,1))
                post_chunk = data.get_samples(start_sample_index = stim, end_sample_index = end_index, 
                                    selected_channels = np.arange(0,n_chs,1))
                pre_chunk = pre_chunk - np.median(pre_chunk, axis = 0 )
                post_chunk = post_chunk - np.median(post_chunk, axis = 0 )
                pre_response[i,:,:] = pre_chunk
                post_response[i,:,:] = post_chunk
            return pre_response, post_response
        
        elif output == 'all':
            response = np.zeros((np.shape(stim_times)[0],total_samps, len(chs)))
            pre_response = np.zeros((np.shape(stim_times)[0],pre_samps, n_chs))
            post_response = np.zeros((np.shape(stim_times)[0],post_samps, n_chs))
            stim_indices = np.searchsorted(data.timestamps,stim_times)
            
            for i, stim in enumerate(stim_indices):
                
                start_index = int(stim - ((pre/1000)*sample_rate))
                end_index = int(stim + ((post/1000)*sample_rate))   

                pre_chunk = data.get_samples(start_sample_index = start_index, end_sample_index = stim, 
                                    selected_channels = np.arange(0,n_chs,1))
                post_chunk = data.get_samples(start_sample_index = stim, end_sample_index = end_index, 
                                    selected_channels = np.arange(0,n_chs,1))
                
                chunk = data.get_samples(start_sample_index = start_index, end_sample_index = end_index, 
                                    selected_channels = chs)
                pre_chunk = pre_chunk - np.median(pre_chunk, axis = 0 )
                post_chunk = post_chunk - np.median(post_chunk, axis = 0 )
                chunk = chunk - np.median(chunk, axis = 0 ) 
                pre_response[i,:,:] = pre_chunk
                post_response[i,:,:] = post_chunk
                response[i,:,:] = chunk
            return pre_response, post_response, response
        
    def plot_ap(self, probe, stim_times, 
                pre = 4, post = 20, 
                first_ch = 125, last_ch = 175, 
                title = '', 
                spike_overlay = False,
                n_trials = 10, spacing_mult = 350, 
                save = False, savepath = '', format ='png'):
        
        data = self.get_raw(probe = probe,band = 'ap')
        response = self.get_chunk(probe = probe, stim_times = stim_times, 
                                  pre = pre, post = post, 
                                  chs = np.arange(first_ch,last_ch))
        
        
        sample_rate = data.metadata['sample_rate']
        total_samps = int((pre/1000 * sample_rate) + (post/1000 * sample_rate))            
        if spike_overlay == True:
            stim_indices = np.searchsorted(data.timestamps,stim_times)
            condition = (
                (self.units['ch'] >= first_ch) &
                (self.units['ch'] <= last_ch) &
                (self.units['probe'] == probe) &
                (self.units['group'] == 'mua')
            )

            spikes = np.array(self.units.loc[condition, 'spike_times'])
            spike_ch = np.array(self.units.loc[condition, 'ch'])

            spike_dict = {}
            for i, stim in enumerate(stim_indices):
                start_index = int(stim - ((pre/1000)*sample_rate))
                end_index = int(stim + ((post/1000)*sample_rate))  
                window = data.timestamps[start_index:end_index]
                filtered_spikes = [spike_times[(spike_times >= window[0]) & (spike_times <= window[-1])] for spike_times in spikes]  
                spike_dict[i] = filtered_spikes

        ## plotting 
        
        trial_subset = np.linspace(0,len(stim_times)-1, n_trials) #choose random subset of trials to plot 
        trial_subset = trial_subset.astype(int)
    
        #set color maps
        cmap = sns.color_palette("crest",n_colors = n_trials)
        #cmap = sns.cubehelix_palette(n_trials)
        colors = cmap.as_hex()
        if spike_overlay == True:
            cmap2 = sns.color_palette("ch:s=.25,rot=-.25", n_colors = len(spikes))
            colors2 = cmap2.as_hex()
        fig=plt.figure(figsize=(16,24))
        time_window = np.linspace(-pre,post,(total_samps))
        for trial,color in zip(trial_subset,colors):
            for ch in range(0,int((last_ch - first_ch))): 
                plt.plot(time_window,response[trial,:,ch]+ch*spacing_mult,color=color)
        
            if spike_overlay == True:
                for i,ch in enumerate(spike_ch): 

                    if spike_dict[trial][i].size > 0:
                        for spike in spike_dict[trial][i]:
                            spike = spike - stim_times[trial]
                            plt.scatter(spike*1000, (spike/spike) + ((ch-(first_ch)))*spacing_mult, 
                            alpha = 0.5, color = colors2[i], s = 500)
        
        plt.gca().axvline(0,ls='--',color='r')       
        plt.xlabel('time from stimulus onset (ms)')
        plt.ylabel('uV')
        plt.title(title)
        
        if save == True:
            plt.gcf().savefig(savepath,format=format,dpi=600)
        
        return fig