
from dlab.nwbtools import option234_positions,load_unit_data,make_spike_secs

import os,glob
import pandas as pd
import numpy as np

from pynwb import NWBHDF5IO
from pynwb import NWBHDF5IO, NWBFile
from pynwb.file import Subject

from datetime import datetime
from dateutil.tz import tzlocal

import pendulum
import re




class NWB_Tools:        
    
        def __init__(self, analysis_obj):
            self.analysis = analysis_obj
            self.mouse = self.analysis.mouse
            if self.analysis.processed:
                self.trials = self.analysis.trials
                self.units = self.analysis.units
            else: 
                self.stim_df = self.analysis.stim_df
            self.path = self.analysis.path


            
        
        def make_unitsdf(self, probes = ['probeA','probeB','probeC'], depths =  [2000, 2000, 2000], herbs = False, stimuli = None):
            """_summary_

            Args:
                probes (list, optional): _description_. Defaults to ['probeA','probeB','probeC'].
                depths (list, optional): _description_. Defaults to [2000, 2000, 2000].
                herbs (bool, optional): _description_. Defaults to False.

            Returns:
                _type_: _description_
            """
            dfs = []
            for i, probe in enumerate(probes):
                
                probe_path = os.path.join(self.path, 'Record Node 105','Experiment1','recording1',
                            'continuous',f'Neuropix-PXI-104.{probe.capitilize()}-AP' )
                make_spike_secs(probe_path)
                df = load_unit_data(probe_path, probe_depth = depths[i], 
                                            spikes_filename = 'spike_secs.npy',
                                            probe_name = probe.strip('Probe'))
                cluster_info = pd.read_csv(os.path.join(probe_path, 'cluster_info.tsv'),sep ='\t')
                ch = np.array(cluster_info.ch)
                depth = np.array(cluster_info.depth)

                df['ch'] = ch
                df['depth'] = depth
                
                dfs.append(df)
            
            units_df = pd.concat(dfs)
            
            print('Line 440: units_df concatenated')
            
            #generate unique unit IDs
            new_unit_ids = []
            for unitid in units_df.index:
                uu = units_df.iloc[[unitid]]
                new_unit_ids.append("{}{}".format(str(list(uu["probe"])[0]), str(list(uu["unit_id"])[0])))
            units_df["unit_id"] = new_unit_ids

            print('Unique Unit IDs generated')

            if herbs:
                from ccf_3D.tools import herbs_processing as hp
                from ccf_3D.tools.metrics import distance
                stim_df = stimuli
                probe_list, stim = hp.load_herbs(self.mouse)
                probe_IDs = [probe.strip('probe') for probe in probes]
                probes_dist = {}
                stim_coords_ = hp.stim_coords(stim)
                most_used_contact = self.stim_df['contact_negative'].value_counts().idxmax()
                
                for ID, probe in zip(probe_IDs, probe_list):
                    vox = hp.neuropixel_coords(probe)
                    dist = distance(vox, stim_coords_[most_used_contact])
                    probes_dist[ID] = dist
                    mask = units_df['probe'] == ID
                    units_df.loc[mask, 'distance_from_stim'] = units_df.loc[mask, 'ch'].apply(lambda x: dist[x-1] if x-1 < len(dist) else None)
                    channel_structures_full,channel_structures_acronym = hp.get_channel_structures_herbs(probe)
                    units_df.loc[mask, 'channel_structures_full'] = units_df.loc[mask,'ch'].apply(lambda x: channel_structures_full[x-1] if x-1 < len(channel_structures_full) else None)
                    units_df.loc[mask, 'channel_structures_acronym'] = units_df.loc[mask,'ch'].apply(lambda x: channel_structures_acronym[x-1] if x-1 < len(channel_structures_acronym) else None)
            
            return units_df
        
        def prep_stim_df(self):   
            stimuli = self.stim_df
            stimuli['notes'] = ''
            stimuli['start_time'] = stimuli['stim_time']
            stimuli['end_time'] = stimuli['stim_time']+2
            return stimuli
        
        def assemble(self, subject, experimenter = 'jlh',
                    experiment_description = 'Electrical Brain Stimulation',
                    device_name = 'DenmanLab_EphysRig2',
                    lab = 'Denman Lab',
                    institution = 'University of Colorado',
                    keywords = ['neuropixels','mouse','electrical stimulation', 'orthogonal neuropixels']):
        
            nwbfile = NWBFile(self.mouse, 
                self.path, 
                datetime.now(tzlocal()),
                experimenter = experimenter,
                lab = lab,
                keywords = keywords,
                institution = institution,
                subject = subject,
                experiment_description = experiment_description,
                session_id = os.path.basename(self.path))
            return nwbfile
            
        def add_stim_epochs(self, nwbfile, stimuli):
            #add epochs
            nwbfile.add_epoch(stimuli.start_time.values[0],
                            stimuli.start_time.values[-1], 'stimulation_epoch')
            nwbfile.add_trial_column('train_duration', 'train duration (s)')
            nwbfile.add_trial_column('train_period', 'train period (s)')
            nwbfile.add_trial_column('train_quantity', 'train quantity')
            nwbfile.add_trial_column('shape', 'monophasic, biphasic or triphasic')
            nwbfile.add_trial_column('run', 'the run number')
            nwbfile.add_trial_column('pulse_duration', 'usecs')
            nwbfile.add_trial_column('pulse_number', 'event quantity')
            nwbfile.add_trial_column('event_period', 'milliseconds')
            nwbfile.add_trial_column('amplitude', 'amplitude in uA')
            nwbfile.add_trial_column('contacts', 'the stimulation contacts and polarities used on the stim electrode')
            nwbfile.add_trial_column('contact_negative', 'the negative (cathodal) contact for a trial')
            nwbfile.add_trial_column('contact_positive', 'the positive (anodal) contact used') 
            nwbfile.add_trial_column('polarity', 'bipolar or monopolar')
            nwbfile.add_trial_column('notes', 'general notes from recording')


            for i in range(len(stimuli)):    
                nwbfile.add_trial(start_time = stimuli.start_time[i],
                    stop_time = stimuli.end_time[i],
                    #parameter = str(stimuli.parameter[i]),
                    amplitude = stimuli.EventAmp1[i],
                    pulse_duration = stimuli.EventDur1[i],
                    shape = stimuli.EventType[i],
                    polarity = str(stimuli.polarity[i]),
                    run = stimuli.Run[i],
                    pulse_number = stimuli.EventQuantity[i],
                    event_period = stimuli.EventPeriod[i]/1e3,
                    train_duration = stimuli.TrainDur[i]/1e6,
                    train_period = stimuli.TrainPeriod[i]/1e6,
                    train_quantity = stimuli.TrainQuantity[i],
                    contacts = stimuli.comment[i],
                    contact_positive = stimuli.contact_positive[i],
                    contact_negative = stimuli.contact_negative[i],
                    notes = stimuli.notes[i])
                
        def add_electrodes(self, nwbfile, units_df, probes, device_name):
            probe_IDs = [probe.strip('Probe') for probe in probes]

            device = nwbfile.create_device(name = device_name)

            for i, probe in enumerate(probes):
                electrode_name = 'probe'+str(i)
                description = "Neuropixels1.0_"+probe_IDs[i]
                location = "near visual cortex"

                electrode_group = nwbfile.create_electrode_group(electrode_name,
                                                                description=description,
                                                                location=location,
                                                                device=device)
                
                #add channels to each probe
                for ch in range(option234_positions.shape[0]):
                    nwbfile.add_electrode(x=option234_positions[ch,0],y=0.,
                                        z=option234_positions[0,1],
                                        imp=0.0,location='none',
                                        filtering='high pass 300Hz',
                                        group=electrode_group)
            nwbfile.add_unit_column('probe', 'probe ID')
            nwbfile.add_unit_column('unit_id','cluster ID from KS2')
            nwbfile.add_unit_column('group', 'user label of good/mua')
            nwbfile.add_unit_column('depth', 'the depth of this unit from zpos and insertion depth')
            nwbfile.add_unit_column('xpos', 'the x position on probe')
            nwbfile.add_unit_column('zpos', 'the z position on probe')
            nwbfile.add_unit_column('no_spikes', 'total number of spikes across recording')
            nwbfile.add_unit_column('KSlabel', 'Kilosort label')
            nwbfile.add_unit_column('KSamplitude', 'Kilosort amplitude')
            nwbfile.add_unit_column('KScontamination', 'Kilosort ISI contamination')
            nwbfile.add_unit_column('template', 'Kilosort template')
            nwbfile.add_unit_column('ch', 'channel number')


            for i,unit_row in units_df[units_df.group != 'noise'].iterrows():
                nwbfile.add_unit(probe=str(unit_row.probe),
                                id = i,
                                unit_id = unit_row.unit_id,
                                spike_times=unit_row.times,
                                electrodes = np.where(unit_row.waveform_weights > 0)[0],
                                depth = unit_row.depth,
                                xpos= unit_row.xpos,
                                zpos= unit_row.zpos,
                                template= unit_row.template,
                                no_spikes = unit_row.no_spikes,
                                group= str(unit_row.group),
                                KSlabel= str(unit_row.KSlabel),
                                KSamplitude= unit_row.KSamplitude,
                                KScontamination= unit_row.KScontamination,
                                ch = unit_row.ch)
        
        def write_nwb(self, nwbfile):
            with NWBHDF5IO(os.path.join(self.path,f'{self.mouse}.nwb'), 'w') as io:
                io.write(nwbfile)     

        def make_nwb(self, subject, probes = ['probeA','probeB','probeC'], 
                    depths =  [2000, 2000, 2000], 
                    herbs = False,
                    experimenter = 'jlh',
                    experiment_description = 'Electrical Brain Stimulation',
                    device_name = 'DenmanLab_EphysRig2',
                    lab = 'Denman Lab',
                    institution = 'University of Colorado',
                    keywords = ['neuropixels','mouse','electrical stimulation', 'orthogonal neuropixels']):
            
            stimuli = self.prep_stim_df()
            print('Stimuli_df prepped')
            
            print('Beginning: loading units_DF')
            units_df = self.make_unitsdf(probes = probes, depths = depths, herbs = herbs, stimuli= stimuli)
            print('Units_df created')

            nwbfile = self.assemble(subject, experimenter = experimenter,
                    experiment_description =  experiment_description,
                    device_name = device_name,
                    lab = lab,
                    institution = institution,
                    keywords = keywords)
            print('NWBFile Assembled')
            self.add_stim_epochs(nwbfile, stimuli)
            print('Stim Epochs added')
            self.add_electrodes(nwbfile, units_df, probes, device_name)
            print('Electrodes added')
            self.write_nwb(nwbfile)
            print('NWB Written')

        def remake_nwb(self, herbs = True):
            # load nwb
            
            # parse nwb file
            
            # get all relevant information
            
            # remake nwb file
            
            pass
        
        def edit_nwb(self, herbs = True, probes = ['probeA','probeB','probeC'],_return = False):
            nwb_path = glob.glob(os.path.join(self.path,'*.nwb'))[0]
            io = NWBHDF5IO(nwb_path, 'r+')
            nwb = io.read()
            units_df = self.units
            stim_df = self.trials

            
            if herbs == True:
                from ccf_3D.tools import herbs_processing as hp
                from ccf_3D.tools.metrics import distance
                probe_list, stim = hp.load_herbs(self.mouse)
                probe_IDs = [probe.strip('probe') for probe in probes]
                probes_dist = {}
                stim_coords_ = hp.stim_coords(stim)
                most_used_contact = stim_df['contact_negative'].value_counts().idxmax()
                
                for ID, probe in zip(probe_IDs, probe_list):
                    vox = hp.neuropixel_coords(probe)
                    dist = distance(vox, stim_coords_[most_used_contact])
                    probes_dist[ID] = dist
                    mask = units_df['probe'] == ID
                    units_df.loc[mask, 'herbs_distance_from_stim'] = units_df.loc[mask, 'ch'].apply(lambda x: dist[x-1] if x-1 < len(dist) else None)
                    channel_structures_full,channel_structures_acronym = hp.get_channel_structures_herbs(probe)
                    units_df.loc[mask, 'channel_structures_full'] = units_df.loc[mask,'ch'].apply(lambda x: channel_structures_full[x-1] if x-1 < len(channel_structures_full) else None)
                    units_df.loc[mask, 'channel_structures_acronym'] = units_df.loc[mask,'ch'].apply(lambda x: channel_structures_acronym[x-1] if x-1 < len(channel_structures_acronym) else None)
                    

                brain_reg_full = list(units_df['channel_structures_full'])
                brain_reg_acronym = list(units_df['channel_structures_acronym'])  
                distance_from_stim = list(units_df['herbs_distance_from_stim'])
                #add columns to nwb
                nwb.units.add_column(name='brain_reg_full', 
                                    description='brain regions by ch from HERBs data', 
                                    data = brain_reg_full)
                nwb.units.add_column(name='brain_reg_acronym', 
                                    description='brain region acronyms by ch from HERBs data', 
                                    data = brain_reg_acronym)
                nwb.units.add_column(name='herbs_distance_from_stim', 
                                    description='distance from a ch to the most used contact of the stimulating electrode', 
                                    data = distance_from_stim)
                
                io.write(nwb)

            if _return == True:
                return nwb #as a writeable object   