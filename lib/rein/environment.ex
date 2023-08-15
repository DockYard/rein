defmodule Rein.Environment do
  @moduledoc """
  Defines an environment to be passed to `Rein`.
  """

  @typedoc "An arbitrary `Nx.Container` that holds metadata for the environment"
  @type t :: Nx.Container.t()
  @type rl_state :: Rein.t()

  @doc "The number of possible actions for the environment"
  @callback num_actions() :: pos_integer()

  @doc """
  Initializes the environment state with the given environment-specific options.

  Also calls `c:reset/2` in the end.
  """
  @callback init(random_key :: Nx.t(), opts :: keyword) :: {t(), random_key :: Nx.t()}

  @doc """
  Resets any values that aren't fixed for the environment state.
  """
  @callback reset(random_key :: Nx.t(), environment_state :: t) :: {t(), random_key :: Nx.t()}

  @doc """
  Applies the given action to the environment.

  Returns the updated environment, also updated with the reward
  and a flag indicating whether the new state is terminal.
  """
  @callback apply_action(rl_state, action :: Nx.t()) :: rl_state
end
