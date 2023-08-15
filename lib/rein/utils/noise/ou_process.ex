defmodule Rein.Utils.Noise.OUProcess do
  @moduledoc """
  Ornstein-Uhlenbeck (OU for short) noise generator
  for temporally correlated noise.
  """

  import Nx.Defn

  @derive {Nx.Container, keep: [], containers: [:theta, :sigma, :mu, :x]}
  defstruct [:theta, :sigma, :mu, :x]

  deftransform init(shape, opts \\ []) do
    opts = Keyword.validate!(opts, theta: 0.15, sigma: 0.2, type: :f32, mu: 0)

    theta = opts[:theta]
    sigma = opts[:sigma]
    type = opts[:type]
    mu = opts[:mu]
    mu = Nx.as_type(mu, type)

    x = Nx.broadcast(mu, shape)
    %__MODULE__{theta: theta, sigma: sigma, mu: mu, x: x}
  end

  defn reset(state) do
    x = Nx.broadcast(state.mu, state.x)
    %__MODULE__{state | x: x}
  end

  defn sample(random_key, state) do
    %__MODULE__{x: x, sigma: sigma, theta: theta, mu: mu} = state

    {state, random_key} =
      if sigma == 0 do
        {state, random_key}
      else
        {sample, random_key} = Nx.Random.normal(random_key, shape: Nx.shape(x))
        dx = theta * (mu - x) + sigma * sample
        x = x + dx

        {%__MODULE__{state | x: x}, random_key}
      end

    {state, Nx.as_type(random_key, :u32)}
  end
end
