import os, glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle as pkl
import re
from tqdm.notebook import tqdm as tqdm

from open_ephys.analysis import Session

from jlh_ephys.loader import Loader
from jlh_ephys.raw import Raw
from jlh_ephys.nwb_tools import NWB_Tools
from jlh_ephys.plotter import Plotter
from jlh_ephys.preprocess import Preprocess

from ccf_3D.tools import herbs_processing as hp
from ccf_3D.tools.metrics import distance


class Analysis:
    def __init__(self, mouse, date):
        self.mouse = mouse
        self.date = date
        
        self.loader = Loader(self.mouse, self.date)
        self.path = self.loader.path
        self.processed = self.loader.processed
        
        if self.processed:
            self.trials = self.loader.trials
            self.units = self.loader.units
            self.parameters = self.loader.parameters
            self.get_probes()
        else: 
            self.preprocess = Preprocess(self)
            self.stim_df = self.preprocess.stim_df
        
        try:
            self.get_contacts()
        except:
            "Could not get_contacts, check trials dataframe"
        self.raw = Raw(self)
        self.nwb_tools = NWB_Tools(self)
        self.plotter = Plotter(self)

    
    def get_probes(self):
        # Extract unique probe names
        probe_IDs = self.units.probe.unique()
        # Format them as "probeA", "probeB", etc.
        probe_names = [f"probe{probe_ID}" for probe_ID in probe_IDs]
        self.probes = probe_names

    def get_contacts(self):
        if self.processed == False:
            df = self.stim_df
            strings = df['comment']
            
            contact_negative = []
            contact_positive = []
            polarity = []
            
            for string in strings:
                r_number = re.search(r'(\d+)r', string)
                r_value = int(r_number.group(1)) if r_number else None
                contact_negative.append(r_value)

                b_number = re.search(r'(\d+)b', string)
                b_value = int(b_number.group(1)) if b_number else 0
                contact_positive.append(b_value)

                if b_value == 0:
                    polarity.append("monopolar")
                else:
                    polarity.append("bipolar")
            
            df['contact_negative'] = contact_negative
            df['contact_positive'] = contact_positive
            df['polarity'] = polarity
            self.stim_df = df
        
        elif 'contact_negative' not in self.trials.columns:
            df = self.trials
            strings = df.contacts
            contact_negative = []
            contact_positive = []
            polarity = []
            
            for string in strings:
                r_number = re.search(r'(\d+)r', string)
                r_value = int(r_number.group(1)) if r_number else None
                contact_negative.append(r_value)

                b_number = re.search(r'(\d+)b', string)
                b_value = int(b_number.group(1)) if b_number else 0
                contact_positive.append(b_value)

                if b_value == 0:
                    polarity.append("monopolar")
                else:
                    polarity.append("bipolar")
            df['contact_negative'] = contact_negative
            df['contact_positive'] = contact_positive
            df['polarity'] = polarity
            self.trials = df
            print('Contacts added to trials dataframe')
        else:
            pass
    
    def get_zetas(self):
        # get zetascore with no window selected
        if os.path.exists(os.path.join(self.path, 'zetascores.pkl')):
            with open(os.path.join(self.path,'zetascores.pkl'), 'rb') as f:
                self.zetascore = pkl.load(f)
        else:
            import zetapy as zp
            zetascore = {}
            spiketimes = np.array(self.units.spike_times)
            for run in tqdm(self.trials.run.unique()):
                stim_time = np.array(self.trials.loc[self.trials.run==run].start_time)
                zetascore[run] = []
                for unit in range(len(spiketimes)):
                    unit_score = zp.getZeta(spiketimes[unit], stim_time)
                    zetascore[run].append(unit_score) 
            with open(os.path.join(self.path,'zetascores.pkl'), 'wb') as f:
                pkl.dump(zetascore, f)
            self.zetascore = zetascore
        
        # do it for 20ms window
        if os.path.exists(os.path.join(self.path, 'zetascores20ms.pkl')):
            with open(os.path.join(self.path,'zetascores20ms.pkl'), 'rb') as f:
                self.zetascore20ms = pkl.load(f)
        else:
            zetascore20ms = {}
            spiketimes = np.array(self.units.spike_times)
            for run in tqdm(self.trials.run.unique()):
                stim_time = np.array(self.trials.loc[self.trials.run == run].start_time)
                zetascore20ms[run] = []
                for i in range(len(spiketimes)):
                    unit_score = zp.getZeta(spiketimes[i],stim_time, dblUseMaxDur = 0.02)
                    zetascore20ms[run].append(unit_score)
            with open(os.path.join(self.path,'zetascores20ms.pkl'), 'wb') as f:
                pkl.dump(zetascore20ms, f)
            self.zetascore20ms = zetascore20ms
        
        # do it again for 300ms window
        if os.path.exists(os.path.join(self.path, 'zetascores300ms.pkl')):
            with open(os.path.join(self.path,'zetascores300ms.pkl'), 'rb') as f:
                self.zetascore300ms = pkl.load(f)
        else:
            zetascore300ms = {}
            spiketimes = np.array(self.units.spike_times)
            for run in tqdm(self.trials.run.unique()):
                stim_time = np.array(self.trials.loc[self.trials.run == run].start_time)
                zetascore300ms[run] = []
                for i in range(len(spiketimes)):
                    unit_score = zp.getZeta(spiketimes[i],stim_time, dblUseMaxDur = 0.3)
                    zetascore300ms[run].append(unit_score)
            with open(os.path.join(self.path,'zetascores300ms.pkl'), 'wb') as f:
                pkl.dump(zetascore300ms, f)
            self.zetascore300ms = zetascore300ms

    def assign_zeta_sig(self, zetas = None, plot = True):
        if zetas is None:
            zetas = self.zetascore300ms
        for run in self.trials.run.unique():
            p = []
            for unit in range(len(self.units.index)):
                p.append(zetas[run][unit][0])
            
            sig = ['sig' if z < 0.05 else 'non-sig' for z in p]
            self.units[f'r{run}'] = sig
        
        if plot:
            for run in self.trials.run.unique():
                p = []
                for unit in range(len(self.units.index)):
                    p.append(zetas[run][unit][0])
                num_sig = []
                for z in p:
                    if z < 0.05:
                        num_sig.append(z)
                print(f'Run: {run}, {self.parameters[run]}: {(len(num_sig)/len(p))*100}')
                perc_sig = (len(num_sig)/len(p))*100
                plt.bar(run, perc_sig)
            plt.title('Zetascore') 
            plt.xlabel('Run')
            plt.ylabel('Percent of Cells activated')
            tt = plt.xticks(range(len(self.parameters)))
    
    def get_electrode_coords(self):
        
        probes, stim = hp.load_herbs(mouse = self.mouse, probe_names = self.probes)
        probe_coords = hp.multi_neuropixel_coords(probes)
        stim_coords = hp.stim_coords(stim)
        Analysis.probe_coords = probe_coords
        Analysis.stim_coords = stim_coords

        return probe_coords, stim_coords

    def get_dists(self, contact = 10):
        probe_coords, stim_coords = self.get_electrode_coords()
        distances = {}
        for probe, coords in probe_coords.items():
            dist = distance(coords, stim_coords[contact])
            distances[probe] = dist
        Analysis.distances = distances
        return distances


    def get_brain_regs(self, return_ = False):
        
        probes, stim = hp.load_herbs(mouse = self.mouse, probe_names = self.probes)
        brain_regs = []
        for probe in probes: 
            full, _ = hp.get_channel_structures_herbs(probe)
            brain_regs.append(full)
        brainreg_A = [brain_regs[0]]
        brainreg_B = [brain_regs[1]]
        brainreg_C = [brain_regs[2]]
        brainreg_A = brainreg_A[0]
        brainreg_B = brainreg_B[0]
        brainreg_C = brainreg_C[0]
        brain_regs = {'probeA':brainreg_A, 'probeB':brainreg_B, 'probeC': brainreg_C}
        Analysis.brain_regs = brain_regs 
        if return_:
            return brain_regs
