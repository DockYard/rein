defmodule Rein.Environment do
  @moduledoc """
  Defines an environment to be passed to `Rein`.
  """

  @typedoc "An arbitrary `Nx.Container` that holds metadata for the environment"
  @type t :: Nx.Container.t()

  @typedoc "The full state of the current Reinforcement Learning process, as stored in the `Rein` struct"
  @type rl_state :: Rein.t()

  @doc """
  Initializes the environment state with the given enviroment-specific options.

  Should be implemented in a way that the result would be semantically
  the same as if `c:reset/2` was called in the end of the function.

  As a suggestion, the implementation should only initialize fixed
  values here, that is values that don't change between sessions
  (epochs for non-episodic tasks, episodes for episodic tasks). Then,
  call `c:reset/2` internally to initialize the rest of variable values.
  """
  @callback init(random_key :: Nx.t(), opts :: keyword) :: {t(), random_key :: Nx.t()}

  @doc """
  Resets any values that vary between sessions (which would be episodes
  for episodic tasks, epochs for non-episodic tasks) for the environment state.
  """
  @callback reset(random_key :: Nx.t(), environment_state :: t) :: {t(), random_key :: Nx.t()}

  @doc """
  Applies the selected action to the environment.

  Returns the updated environment, also updated with the reward
  and a flag indicating whether the new state is terminal.
  """
  @callback apply_action(rl_state, action :: Nx.t()) :: rl_state
end
