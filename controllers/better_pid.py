from icecream import ic
from . import BaseController
import numpy as np

class Controller(BaseController):
  def __init__(self,):
    self.p = 0.195
    self.i = 0.100
    self.d = -0.053
    self.error_integral = 0
    self.prev_error = 0
    self.prev_lataccel : float|None = None
    self.lam = 0.75

  def update(self, target_lataccel, current_lataccel, state, future_plan):
    if self.prev_lataccel is not None: current_lataccel = current_lataccel * self.lam + self.prev_lataccel * (1-self.lam)
    error = (target_lataccel - current_lataccel)
    self.error_integral += error
    error_diff = error - self.prev_error
    self.prev_error = error
    self.prev_lataccel = current_lataccel
    return self.p * error + self.i * self.error_integral + self.d * error_diff
