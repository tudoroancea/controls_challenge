import argparse
import math
import os
from time import perf_counter

import casadi as ca
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tinygrad.nn as nn
from tinygrad import Tensor, TinyJit
from tinygrad.helpers import trange

ACC_G = 9.81

Tensor.manual_seed(127)


def get_data(data_path: str) -> pd.DataFrame:
  df = pd.read_csv(data_path)
  processed_df = pd.DataFrame(
    {
      "roll_lataccel": np.sin(df["roll"].values) * ACC_G,
      "v_ego": df["vEgo"].values,
      "a_ego": df["aEgo"].values,
      "target_lataccel": df["targetLateralAcceleration"].values,
      "steer_command": -df["steerCommand"].values,
    }
  )
  return processed_df


def create_data():
  past = []
  i = 0
  for f in os.listdir("data"):
    if f.endswith(".csv"):
      data = get_data(os.path.join("data", f))
      past.append(data.dropna().to_numpy())
    i += 1
  np.save("data.npy", np.stack(past))


class LinearModel:
  def __init__(self, n: int):
    self.weights = Tensor.normal(n, requires_grad=True)

  def __call__(self, x: Tensor):
    return x @ self.weights


class MLPModel:
  def __init__(
    self,
    input_size: int,
    hidden_size: int,
    output_size: int,
    hidden_layers: int = 1,
  ):
    self.input = Tensor.normal(input_size, hidden_size, requires_grad=True)
    self.hidden_layers = [
      Tensor.normal(hidden_size, hidden_size, requires_grad=True)
      for _ in range(hidden_layers - 1)
    ]
    self.output = Tensor.normal(hidden_size, output_size, requires_grad=True)

  def __call__(self, x: Tensor):
    x = x.dot(self.input).tanh()
    for layer in self.hidden_layers:
      x = x.dot(layer).tanh()
    return x.dot(self.output)


def train():
  data = Tensor(np.load("data.npy").astype(np.float32))
  model = MLPModel(int(data.shape[-1]), 128, 1, 1)
  optimizer = nn.optim.Adam(nn.state.get_parameters(model), 1e-2)
  batch_size = 10000
  steps = 400

  @TinyJit
  def training_step():
    i, j = (
      Tensor.randint(batch_size, low=0, high=int(data.shape[0])),
      Tensor.randint(batch_size, low=0, high=int(data.shape[1]) - 1),
    )
    loss = (data[i, j + 1, -2] - model(data[i, j]).squeeze(-1)).square().mean()
    optimizer.zero_grad()
    loss.backward()
    return loss.realize(*optimizer.schedule_step(), i, j)

  losses = []
  best_loss = math.inf
  nn.state.safe_load
  with Tensor.train():
    for _ in (pbar := trange(steps, unit="steps")):
      start = perf_counter()
      losses.append(training_step().item())
      end = perf_counter()
      if losses[-1] < best_loss:
        best_loss = losses[-1]
        nn.state.safe_save(nn.state.get_state_dict(model), "best_weights.safetensors")
      pbar.desc = f"Loss: {losses[-1]:.4f}, best loss: {best_loss:.4f}, step time: {end - start:.2f}ms"
  final_loss = losses[-1]

  # eval on the whole dataset
  val_loss = (data[:, 1:, -2] - model(data[:, :-1]).squeeze(-1)).square().mean()
  print(f"Validation loss: {val_loss.item():.4f}")

  plt.semilogy(losses)
  plt.xlabel("Step")
  plt.ylabel("Loss")
  plt.title(
    f"Final training loss: {final_loss:.4f}, best training loss: {best_loss:.4f}, validation loss: {val_loss.item():.4f}"
  )
  plt.show()


class Controller:
  def __init__(self):
    weights = nn.state.safe_load("best_weights.safetensors")
    self.C0 = ca.DM(weights["input"].numpy())
    self.C1 = ca.DM(weights["output"].numpy())

    self.N = 1

    self.opti = ca.Opti()
    self.x0 = self.opti.parameter()
    self.w = self.opti.parameter(self.N + 1, 3)
    self.xref = self.opti.parameter(self.N + 1)

    self.x = self.opti.variable(self.N + 1)
    self.u = self.opti.variable(self.N)

    self.opti.minimize(
      50 * ca.sumsqr(self.x - self.xref) + ca.sumsqr(ca.diff(self.x) / 10)
    )

    self.opti.subject_to(
      self.x[1:]
      == ca.tanh(ca.hcat([self.w[:-1, :], self.x[:-1], self.u]) @ self.C0) @ self.C1
    )

    self.opti.solver(
      "ipopt", {"print_time": 0, "ipopt": {"sb": "yes", "print_level": 0}}
    )

  def update(self, target_lataccel: float, current_lataccel, state, future_plan):
    # TODO:pad future_plan with last value

    self.opti.set_value(self.x0, current_lataccel)
    self.opti.set_value(
      self.w,
      np.concatenate(
        (
          np.array([[state.v_ego, state.a_ego, state.roll_lataccel]]),
          np.stack(
            (
              future_plan.v_ego[: self.N],
              future_plan.a_ego[: self.N],
              future_plan.roll_lataccel[: self.N],
            ),
            axis=1,
          ),
        )
      ),
    )
    self.opti.set_value(
      self.xref,
      np.concatenate(([target_lataccel], future_plan.lataccel[: self.N])),
    )
    self.opti.set_initial(self.x, np.zeros(self.N + 1))
    self.opti.set_initial(self.u, np.zeros(self.N))

    sol = self.opti.solve()
    return sol.value(self.u)


def test_controller():
  Controller()


if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument("task")
  args = parser.parse_args()
  globals()[args.task]()
