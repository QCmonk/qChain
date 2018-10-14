from capy import *
from operators import *
from qChain import *
from utility import *
from IPython import embed

import os
import time
import numpy as np 
import scipy.signal as scs
import matplotlib.pyplot as plt

# TODO: figure save and reconstruction save
#       multiprocessing add
#       Add metric information about reconstruction to plots



def recon_pulse(sim_vars, plot=True, savefig=False):

    bfreqs, projs, sim_def = atomic_sense(**sim_vars)

    # add to simulation parameter dictionary if not already defined
    if "measurements" not in sim_vars.keys():
      sim_vars["measurements"] = 50

    # normally distributed noise if requested
    if not np.isclose(sim_vars["noise"], 0.0):
      projs += np.random.normal(loc=0.0, scale=sim_vars["noise"], size=np.size(projs))

    # determine time vector and add to variable dictionary 
    time = np.arange(sim_def["tstart"], sim_def["dett"], 1/sim_vars["fs"])
    sim_vars["time"] = time

    # generate measurement transform
    transform, ifreqs, _ = measure_gen(ndim=len(time), 
                                    time=time, 
                                    basis="fourier", 
                                    measurements=sim_vars["measurements"], 
                                    freqs=bfreqs)
    

    ifreqs = np.asarray(ifreqs)

    # compute measurement record, selecting only those chosen for reconstruction
    meas_record = projs[[i for i in ifreqs]] 
    # remove probability bias (looks like there is a DC component otherwise)
    bias = np.mean(meas_record)
    meas_record -= bias

    # perform single event matched filtering
    if sim_vars["method"]=="match":
      # define template
      exit()
    # perform multievent matched filtering
    elif sim_vars["method"]=="mulmatch":
      # generate matching template 
      t = np.arange(0,1/sim_vars["sig_freq"],1/sim_vars["fs"])
      template = np.sin(2*pi*sim_vars["sig_freq"]*t)
      
      # create optimiser instance
      comp = CAOptimise(svector=meas_record,   # measurement vector in sensor domain
                      transform=transform,   # sensor map from source to sensor
                      template=template,
                      verbose=True,          # give me all the information
                      **sim_vars)           # epsilon and other optional stuff

      # generate actual neural signal for comparison purposes
      signal = pulse_gen(sim_vars["sig_freq"], tau=[sim_vars["tau"]], amp=sim_vars["sig_amp"])(time)

      # perform matched filtering identification
      comp.py_notch_match(osignal=signal)
      
      # extract reconstruction information
      recon = comp.template_recon
      notch = comp.notch

      # format reconstruction information and save
      print("Storing signal_reconstruction")
      recon_sig = np.empty((4,len(notch)), dtype=np.float)
      recon_sig[0,:] = time
      recon_sig[1,:] = comp.correlation
      recon_sig[2,:] = recon
      recon_sig[3,:] = notch
      data_store(sim_vars, recon_sig, root_group="Signal_Reconstructions", verbose=True)

    else:
      if sim_vars["method"] != "default":
        print("Unrecognised reconstruction method, using default")
        sim_vars["method"] = "default"
      # create optimiser instance
      comp = CAOptimise(svector=meas_record,   # measurement vector in sensor domain
                      transform=transform,   # sensor map from source to sensor
                      verbose=True,          # give me all the information
                      **sim_vars)           # epsilon and other optional stuff

      # reconstruct signal
      comp.cvx_recon()
      # extract signal estimate
      recon = comp.u_recon

      # save to archive
      recon_sig = np.empty((2,len(recon)), dtype=np.float)
      recon_sig[0,:] = time
      recon_sig[1,:] = recon
      data_store(sim_vars, recon_sig, root_group="Signal_Reconstructions", verbose=True)

          
    if plot:
      # measurement frequencies used
      comp_f = bfreqs[ifreqs]
      # measurement record adjustment
      comp_p = meas_record + bias
      # plot reconstruction
      plot_gen_2(bfreqs, projs, comp_f, comp_p, time, recon, {**sim_def,**sim_vars}, savefig=savefig)



    # return reconstruction and original signal
    signal = signal_generate(time, sim_vars)
    return signal, recon/np.max(np.abs(recon))


def epsilon_find(sim_vars, erange):
  """
  Computes the reconstruction error using a range of epsilon values for 
  the same simulation parameters. 
  """
  errors = []
  for epsilon in erange:
    sim_vars["epsilon"] = epsilon
    original, recon = recon_pulse(sim_vars, plot=False, savefig=False)
    # compute error
    errors.append(rmse(original,recon))

  # plot RMSE error against epsilon
  plt.plot(erange, errors)
  plt.figure(num=1, size=[16,9])
  plt.xlabel("Epsilon")
  plt.ylabel("RMSE")
  plt.title("Reconstruction Error vs. Epsilon")
  plt.grid(True)
  plt.show()

if __name__ == "__main__":

    ###############################################
    # Compressive Reconstruction of Neural Signal #
    ###############################################
    # set random seed for tuning frequency choice
    np.random.seed(141)
    # define user parameters for simulation run
    sim_vars = {"measurements":        50,           # number of measurements
                "epsilon":           0.01,           # radius of hypersphere in measurement domain
                "sig_amp":             40,           # amplitude of magnetic field signal in Gauss/Hz
                "sig_freq":          5023,           # frequency of magnetic field signal
                "nte":      0.033+50/5023,
                "tau":              0.033,           # time events of pulses
                "f_range":  [4500,5500,2],           # frequency tunings of BECs
                "noise":             0.00,           # noise to add to measurement record SNR percentage e.g 0.1: SNR = 10
                "zlamp":              500,
                "method":       "default",
                "savef":             5000,
                "fs":               2.2e4}           # sasvempling rate for measurement transform

    # epsilon range
    #erange = np.linspace(0.0,0.05, 50)
    #epsilon_find(sim_vars, erange)

    bfreqs1, projs1 = recon_pulse(sim_vars, plot=True, savefig=False)
    
    # sim_vars["zlamp"] = 0
    # bfreqs2, projs2 = recon_pulse(sim_vars, plot=False) v 

    # # plot those bad boys against each other
    # #plt.style.use('dark_background')
    # plt.plot(bfreqs1,projs1, 'r-')
    # plt.plot(bfreqs2, projs2, 'g-')
    # plt.title(r"Effect of $\sigma_z$ line noise ")
    # plt.legend([r"With $\sigma_z $ line noise",r"Without $\sigma_z $ line noise"])
    # plt.ylabel(r"$\langle F_z \rangle $")
    # plt.xlabel("Rabi Frequency (Hz)")
    # plt.figure(num=1, figsize=[16,9])
    # plt.show()
    exit()
    ###############################################
    
    # # set the bajillion simulation parameters. There simply isn't a better way to do this. 
    # # define generic Hamiltonian parameters with Zeeman splitting and rf dressing
    params = {"tstart":        0,              # time range to simulate over
              "tend":       0.95/gyro,
              "dt":         1e-8,
              "larmor":     gyro,              # bias frequency (Hz)
              "rabi":       0,              # dressing amplitude (Hz/2)
              "rff":        gyro,              # dressing frequency (Hz)
              "nf":         5000,              # neural signal frequency (Hz)
              "sA":            0,              # neural signal amplitude (Hz/G)
              "nt":          5e2,              # neural signal time event (s)
              "dett":        0.1,              # detuning sweep start time
              "detA":       5000,              # detuning amplitude
              "dete":       0.25,              # when to truncate detuning
              "beta":         80,              # detuning temporal scaling
              "xlamp":         0,              # amplitude of 
              "xlfreq":       50,
              "xlphase":     0.0,
              "zlamp":         0,
              "zlfreq":       50,
              "zlphase":     0.1,
              "proj": meas1["0"],              # measurement projector
              "savef":         1}              # plot point frequency
 

    # shift = (params['rabi']**2)/(4*params["rff"]) + (params["rabi"]**4/(4*(params["rff"])**3))
    # params["larmor"] += shift

    # atom = SpinSystem(init="super")
    # tdomain, probs, pnts = atom.state_evolve(params=params, bloch=[True, 1], lite=False)
    
    # atom.bloch_plot(pnts)

    # time1, pp, Bfield1 = atom.field_get(params)
    # plt.plot(time1, Bfield1[:,1])
    # plt.show()
    # exit()
    #print(atom.probs[-1])
    #atom.exp_plot(tdomain, probs, "Evolution of <F_z>")



