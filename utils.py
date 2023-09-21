import numpy as np
from open_ephys.analysis import Session

def OE(path):
        session = Session(path)
        recording = session.recordnodes[0].recordings[0]
        return recording

# need to add a polarity parameter checker to NWB file that parses the contacts column
def choose_stim_parameter(trials, amp=-100, pulse_number = 1, pulse_duration=100):
    stim_times = trials.loc[
        (trials['amplitude'] == amp) &
        (trials['pulse_number'] == pulse_number) &
        (trials['pulse_duration'] == pulse_duration)
    ]['start_time']
    return np.array(stim_times)

def stim_dictionary(trials):
    parameters = {}
    for run in trials.run.unique():
        amp = np.array(trials.loc[trials.run == run].amplitude)[0]
        pulse_width = np.array(trials.loc[trials.run == run].pulse_duration)[0]
        contacts = np.array(trials.loc[trials.run == run].contacts)[0]
        parameters[int(run)] = f'amp: {amp} ua, pw: {pulse_width} us, contacts: {contacts}'
    return parameters