# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import scipy.signal
import math

plt.rcParams['figure.figsize'] = [8, 12]

pi = math.pi

# core functions

def ricker(t0, t, f):
  w = 2 * pi * f
  t = t - t0
  tsqwsq = t**2 * w**2
  return (1 - 0.5 * tsqwsq) * np.exp(-0.25 * tsqwsq)

def blackman(t0, t, m):
  dt = t[1] - t[0]
  t = t - t0
  half_m = 0.5 * m
  n = t / dt + half_m
  w = np.zeros(t.size)
  blackman = 0.42 - 0.5 * np.cos(2 * pi * n / m) + 0.08 * np.cos(4 * pi * n / m)
  mask = (n >= 0) & (n <= m)
  w[ mask ] = blackman[ mask ]
  return w

def bandlimited_spike(t0, t, m):
  dt = t[1] - t[0]
  return blackman(t0, t, m) * np.sinc((t-t0)/dt)

#### DIX Equation #####

def dix_t_rms2int(vrms1, vrms2, t1, t2):
  return np.sqrt((t2*vrms2**2 - t1*vrms1**2)/(t2-t1))

def vrms(t, vint):
  dt = np.diff(t)
  v_ = np.sqrt(np.cumsum(dt * vint[1:]**2) / t[1:])
  return np.concatenate([[v_[0]], v_])

def vint(t, vrms):
  t1 = t[:-1]
  t2= np.roll(t,-1)[:-1]
  vrms1 = vrms[:-1]
  vrms2 = np.roll(vrms,-1)[:-1]
  v_ = dix_t_rms2int(vrms1, vrms2, t1, t2)
  return np.concatenate([[v_[0]], v_])

##### NMO Equations #######

# 4th order Alkhalifah
def alkh_moveout(t0, x, v, eta):
  tsq = t0**2 + x**2/v**2 - (2 * eta * x**4) / (v**2 * (t0**2 * v**2 + (1 + 2*eta)*x**2))
  return np.sqrt(tsq)

def effective_eta(t, vrms, eta):
  # not yet tested
  return (np.cumsum(t * vint(t, vrms)**4 * (1 + 8*eta)) / (t*vrms) - 1) / 8

#### Model Building ######

def model1d(z, horzlist, valuelist, background=1500):
    mod = np.ones(z.size) * background
    idx = np.argsort(horzlist)
    deltas = [ background if d == None else d for d in valuelist ]
    halfdz = (z[1]-z[0])/2
    for ii in range(len(horzlist)):
        jj = idx[ii]
        mod[ z + halfdz >= horzlist[jj] ] = deltas[jj]
    return mod

# def rescale(x):
#   """rescale after convolution"""
#   return x / np.sqrt(np.abs(x))


# Core

class Location:

  def __init__(self, x=0, y=0, **kwargs):
      self.x = x
      self.y = y

  def vector_to(self, Location):
      dx = Location.x - self.x
      dy = Location.y - self.y
      return np.array([dx, dy])

  def offset_to(self, Location):
      vec = self.vector_to(Location)
      return np.sqrt(np.sum(vec**2))

class TimeSeries:

  def __init__(self, t0=0, nt=2000, dt=0.004, **kwargs):
    self.t0 = t0
    self.nt = nt
    self.dt = dt

  def t(self):
    return self.dt * np.arange(self.nt)

  def tmax(self):
    return self.t0 + self.nt * self.dt

class Ricker:

  def __init__(self, freq=20, dt=0.004, **kwargs):
    self.freq = freq
    self.dt = dt
    # wavelet should be about 4 times dominant period (just by looking)
    period = 1 / freq
    npts = 4 * (period // dt) + 1
    t = dt * ( np.arange(npts) - (npts//2) )
    self.wavelet = ricker(0, t, freq)


# Basic

# class Source(Location):

#   def __init__(self, wavelet_points=50, wavelet_width=5, **kwargs):
#     super().__init__(**kwargs)
#     self.wavelet_points = wavelet_points
#     self.wavelet_width = wavelet_width

#   def wavelet(self):
#     return scipy.signal.ricker(self.wavelet_points, self.wavelet_width)

class Source(Location):

  def __init__(self, **kwargs):
    super().__init__(**kwargs)

class Receiver(Location):

  def __init__(self, **kwargs):
    super().__init__(**kwargs)

class Array:

  def __init__(self, receiver_list=None, **kwargs):
    self.receiver_list = receiver_list

    if receiver_list == None:
      self.receiver_list = [ Receiver(x=x, y=y) for x, y in 
                            zip(np.zeros(500), 12.5 * np.arange(500)) ]

  def plot(self, **kwargs):
    xylist = [ [ r.x, r.y ] for r in self.receiver_list ]
    plt.scatter(*tuple(map(list, zip(*xylist))))
    # plt.gca().set_aspect('equal')
    plt.xlabel('x')
    plt.ylabel('y')
    return plt.gca()

class RegularArray(Array):

  def __init__(self, x0=0, nx=256, dx=25, **kwargs):
    receiver_list = [ Receiver(x=x, y=y) for x, y in 
                            zip(np.zeros(nx), x0 + dx * np.arange(nx)) ]
    super().__init__(receiver_list = receiver_list, **kwargs)
    


class Velocity(TimeSeries):

  def __init__(self, hor_list=[1, 2, 3], vel_list=[2000, 3000, 4000], **kwargs):
    super().__init__(**kwargs)
    self.hor_list = hor_list
    self.vel_list = vel_list

  def vel_int(self, **kwargs):
    return model1d(self.t(), self.hor_list, self.vel_list, **kwargs)

  def vel_rms(self, **kwargs):
    return vrms(self.t(), self.vel_int(**kwargs))

  def plot(self, ax=None, interval=True, rms=True, **kwargs):
    if ax == None:
      fig, ax = plt.subplots()
    if interval:
       ax.plot(self.vel_int(), self.t(), label='interval')
    if rms:
       ax.plot(self.vel_rms(), self.t(), label='rms')
    ax.legend()
    ax.set_ylabel('time (s)')
    ax.set_xlabel('velocity (m/s)')
    plt.gca().invert_yaxis()


class Eta(TimeSeries):

  def __init__(self, hor_list=[1, 2, 3], eta_list=[.1, .2, .3], **kwargs):
    super().__init__(**kwargs)
    self.hor_list = hor_list
    self.eta_list = eta_list

  def eta(self, **kwargs):
    return model1d(self.t(), self.hor_list, self.eta_list, background=0, **kwargs)

  def eta_eff(self, vel, **kwargs):
    return effective_eta(self.t(), vel.vel_rms(**kwargs), self.eta(**kwargs), **kwargs)

  def plot(self, vel=None, **kwargs):
    """if vel provided will plot effective eta."""
    plt.plot(self.t(), self.eta(), label='eta')
    if vel:
       plt.plot(self.t(), self.eta_eff(vel, **kwargs), label='effective eta')
    plt.legend()
    plt.xlabel('time (s)')
    plt.ylabel('eta')
    return plt.gca()

class Density(TimeSeries):

  def __init__(self, hor_list=[1, 2, 3], rho_list=[2000, 3000, 4000], background=1000, **kwargs):
    super().__init__(**kwargs)
    self.hor_list = hor_list
    self.rho_list = rho_list

  def rho(self, **kwargs):
    return model1d(self.t(), self.hor_list, self.rho_list, **kwargs)

  def plot(self, ax=None, **kwargs):
    if ax == None:
      fig, ax = plt.subplots()
    ax.plot(self.rho(), self.t())
    ax.set_ylabel('time (s)')
    ax.set_xlabel('density (kg/m^3)')
    plt.gca().invert_yaxis()
    return ax

class Reflectivity(TimeSeries):

  def __init__(self, hor_list=[1, 2, 3], rfl_list=[1, 0.5, -0.3], **kwargs):
    super().__init__(**kwargs)
    self.hor_list = hor_list
    self.rfl_list = rfl_list

  # def primary(self, offset, vel, **kwargs):
  #   series = np.zeros(self.nt)
  #   vel_list = np.interp(self.hor_list, vel.t(), vel.vel_rms())
  #   # eta=0 for now
  #   hor_list = [ alkh_moveout(self.hor_list[ii], offset, vel_list[ii], 0) for ii in range(len(vel_list)) ]
  #   # idxs = [ round(x) for x in np.interp(hor_list, self.t(), np.arange(self.nt)) ]
  #   # for idx, value in zip(idxs, self.rfl_list):
  #   #   series[idx] += value
  #   # return series 
  #   idxs = np.interp(hor_list, self.t(), np.arange(self.nt), right=np.nan)
  #   for idx, value in zip(idxs, self.rfl_list):
  #     if not np.isnan(idx):
  #       iidx = round(idx)
  #       series[iidx] += value
  #   return series

  # def primary(self, offset, vel, freq=20, **kwargs):
  #   series = np.zeros(self.nt)
  #   vel_list = np.interp(self.hor_list, vel.t(), vel.vel_rms())
  #   # eta=0 for now
  #   hor_list = [ alkh_moveout(self.hor_list[ii], offset, vel_list[ii], 0) for ii in range(len(vel_list)) ]
  #   t = self.t()
  #   for ht, value in zip(hor_list, self.rfl_list):
  #     series += value * ricker(ht, t, freq)
  #   return series

  def primary(self, offset, vel, freq=20, **kwargs):
    series = np.zeros(self.nt)
    vel_list = np.interp(self.hor_list, vel.t(), vel.vel_rms())
    # eta=0 for now
    hor_list = [ alkh_moveout(self.hor_list[ii], offset, vel_list[ii], 0) for ii in range(len(vel_list)) ]
    t = self.t()
    for ht, value in zip(hor_list, self.rfl_list):
      series += value * bandlimited_spike(ht, t, 21)
    return series

  def multiple(self, offset, vel, order=1, **kwargs):
    prim = self.primary(offset/(order+1), vel, **kwargs)
    mult = prim
    for ii in range(order):
      mult = -np.convolve(prim, mult)[0:self.nt] 
      ### normalise amplitudes of multiples due to the multiplication that happens in convolution
      mask = mult!=0
      mult[mask] = mult[mask] / np.sqrt(np.abs(mult[mask]))
    return mult

  def reflectivity(self, offset, vel, order=0, **kwargs):
    record = self.primary(offset, vel, **kwargs)
    for ii in range(order):
      record += self.multiple(offset, vel, order=ii+1, **kwargs)
    return record

  def plot(self, offset=0, vel=Velocity(), **kwargs):
    plt.plot(self.t(), self.reflectivity(offset, vel, **kwargs))
    return plt.gca()


class Trace(TimeSeries):

  def __init__(self, source=None, receiver=None, 
               velocity=None, reflectivity=None, 
               wavelet=None, **kwargs):
    
    super().__init__(**kwargs)
    self.source = source
    self.receiver = receiver
    self.velocity = velocity
    self.reflectivity = reflectivity
    self.wavelet = wavelet

    if source == None:
      self.source = Source(**kwargs)
    
    if receiver == None:
      self.receiver = Receiver(**kwargs)

    if velocity == None:
      self.velocity = Velocity(**kwargs)

    if reflectivity == None:
      self.reflectivity = Reflectivity(**kwargs)

    if wavelet == None:
      self.wavelet = Ricker(**kwargs)

  def offset(self):
    return self.source.offset_to(self.receiver)

  def x(self, **kwargs):
    return np.convolve(self.wavelet.wavelet, 
                       self.reflectivity.reflectivity(self.offset(), 
                           self.velocity, **kwargs), 'same')
    
  # def x(self, **kwargs):
  #   return self.reflectivity.reflectivity(self.offset(), 
  #                          self.velocity, **kwargs)
  
  def plot(self, **kwargs):
    plt.plot(self.t(), self.x(**kwargs))
    return plt.gca()
    
class Model(TimeSeries):

  def __init__(self, hor_list, vel_list, rho_list, eta_list=None, **kwargs):

    super().__init__(**kwargs)
    self.hor_list = hor_list
    self.vel_list = vel_list
    self.rho_list = rho_list
    self.eta_list = eta_list

    if self.hor_list[0] != 0:
      raise Exception('hor_list must start with 0.')
    
    if len(self.hor_list) != len(self.vel_list):
      raise Exception('hor_list must be same length as vel_list.')

    if len(self.hor_list) != len(self.rho_list):
      raise Exception('hor_list must be same length as rho_list.')

  def _refl_list(self, **kwargs):
    vel = np.asarray(self.vel_list)
    rho = np.asarray(self.rho_list)
    mult1 = rho * vel
    mult2 = np.roll(mult1, -1)
    refl = (mult2-mult1) / (mult2 + mult1)
    return list(refl)

  def vel(self, **kwargs):
    return Velocity(self.hor_list, self.vel_list, **kwargs)

  def ref(self, **kwargs):
    return Reflectivity(self.hor_list[1:], self._refl_list()[:-1], **kwargs)

class Shot:

  def __init__(self, source=None, array=None, velocity=None, reflectivity=None, **kwargs):

    self.source = source
    self.array = array
    self.velocity = velocity
    self.reflectivity = reflectivity

    if source == None:
      self.source = Source(**kwargs)
    
    if array == None:
      self.array = RegularArray(**kwargs)

    if velocity == None:
      self.velocity = Velocity(**kwargs)

    if reflectivity == None:
      self.reflectivity = Reflectivity(**kwargs)

  def _get_trace(self, receiver, **kwargs):
    tr = Trace(source=self.source, receiver=receiver, 
               velocity=self.velocity, reflectivity=self.reflectivity, **kwargs)
    return tr.x(**kwargs)

  def _get_offset(self, receiver, **kwargs):
    return receiver.offset_to(self.source)

  def data(self, noise=0, **kwargs):
    # quicktrace = Trace(**kwargs)
    # quickrick = Ricker(**kwargs)
    # # calculate a bit extra and chop off later to avoid ragged crap at bottom of trace
    # kwargs['nt'] = quicktrace.nt + int(0 / (quicktrace.dt * quickrick.freq)) 
    data = np.asarray([ self._get_trace(rcv, **kwargs) for rcv in self.array.receiver_list ])
    # data = data[:,:quicktrace.nt]
    data = data - np.mean(data)
    data = data / np.std(data)
    if noise > 0:
      data += self._noise(scale=noise)
    return data.T
  
  def shape(self):
    return (len(self.array.receiver_list), self.reflectivity.nt)

  def size(self):
    return len(self.array.receiver_list) * self.reflectivity.nt

  def _noise(self, scale=0.05, **kwargs):  
    rng = np.random.default_rng()  
    return rng.normal(scale=scale, size=self.size()).reshape(self.shape())

  # def nmo_data(self, **kwargs):


  def plot(self, ax=None, **kwargs):
    if ax == None:
      fig, ax = plt.subplots()

    data = self.data(**kwargs)
    data = data - np.mean(data)
    data = data / np.std(data)

    mino = self._get_offset(self.array.receiver_list[0])
    maxo = self._get_offset(self.array.receiver_list[-1])
    mint = self.reflectivity.t0
    maxt = self.reflectivity.tmax()

    ax.imshow(data, cmap='gray', vmin=-1, vmax=1, 
              extent=[mino, maxo, maxt, mint], aspect='auto',
              interpolation='lanczos')
    ax.set_xlabel('offset (m)')
    ax.set_ylabel('time (s)')
    # return plt.gca()

  def plot_pcolormesh(self, ax=None, **kwargs):
    if ax == None:
      fig, ax = plt.subplots()

    data = self.data(**kwargs)
    data = data - np.mean(data)
    data = data / np.std(data)

    mino = self._get_offset(self.array.receiver_list[0])
    maxo = self._get_offset(self.array.receiver_list[-1])
    mint = self.reflectivity.t0
    maxt = self.reflectivity.tmax()

    oo, tt = np.meshgrid(np.linspace(mino, maxo, data.shape[1]), np.linspace(mint, maxt, data.shape[0]))
    ax.pcolormesh(oo, tt, data, cmap='gray', vmin=-1, vmax=1)
    ax.set_xlabel('offset (m)')
    ax.set_ylabel('time (s)')
    plt.gca().invert_yaxis()

  def plot2(self, **kwargs):
    fig1, f1_axes = plt.subplots(ncols=2, nrows=1, constrained_layout=True, sharey=True)
    self.plot_pcolormesh(ax=f1_axes[0], **kwargs)
    self.velocity.plot(ax=f1_axes[1], **kwargs)
    # f1_axes[1] = self.velocity.plot()
    plt.gca().invert_yaxis()


def random_shot(n_hors = 3, **kwargs):

  # random number generator
  rng = np.random.default_rng()

  # print(type(n_hors))
  if isinstance(n_hors, str):
    # poisson distribution to randomly set number of horizons
    n_hors = rng.poisson(int(n_hors))
    n_hors = np.maximum(1, n_hors)

  # uniform distribution to set horizon depths
  quicktrace = Trace(**kwargs)
  trace_length = quicktrace.nt * quicktrace.dt
  # hor_list = rng.random(n_hors) * trace_length
  # avoid horizons at edges
  hor_list = (0.95 * rng.random(n_hors) + 0.025) * trace_length

  # normal distribution to get gradient for density trend
  k = rng.normal(loc=500, scale=100)
  c = 1500 - hor_list.min() * k
  rho_list = k * hor_list + c
  # normal distribution to perturb densities
  rho_list = rho_list * np.exp(rng.normal(scale=0.1, size=n_hors))
  # clip min/max
  minrho = 1100
  maxrho = 3500
  rho_list = np.minimum(rho_list, maxrho)
  rho_list = np.maximum(rho_list, minrho)
  # vel = Density(hor_list=hor_list, vel_list=vel_list, **kwargs)

  # # normal distribution to set reflector amplitudes
  # ref_list = rng.normal(size=n_hors)
  # ref = Reflectivity(hor_list, ref_list, **kwargs)

  # normal distribution to get gradient for velocity trend
  k = rng.normal(loc=500, scale=100)
  c = 1500 - hor_list.min() * k
  vel_list = k * hor_list + c
  # normal distribution to perturb velocities
  vel_list = vel_list * np.exp(rng.normal(scale=0.1, size=n_hors))
  # clip min/max
  minvel = 1500
  maxvel = 8000
  vel_list = np.minimum(vel_list, maxvel)
  vel_list = np.maximum(vel_list, minvel)
  # vel = Velocity(hor_list=hor_list, vel_list=vel_list, **kwargs)

  hor_list = [0] + list(hor_list)
  vel_list = [minvel] + list(vel_list)
  rho_list = [1000] + list(rho_list)

  # print('hor_list', hor_list)
  # print('vel_list', vel_list)
  # print('rho_list', rho_list)
  
  m = Model(hor_list, vel_list, rho_list, **kwargs)
  # print(kwargs)
  return Shot(reflectivity = m.ref(**kwargs), velocity = m.vel(**kwargs), **kwargs)
  # return Shot(**kwargs)
