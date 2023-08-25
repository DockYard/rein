defmodule Rein.Agent do
  @moduledoc """
  The behaviour that should be implemented by a `Rein` agent module.
  """

  @typedoc "An arbitrary `Nx.Container` that holds metadata for the agent"
  @type t :: Nx.Container.t()
  @typedoc "The full state of the current Reinforcement Learning process, as stored in the `Rein` struct"
  @type rl_state :: Rein.t()

  @doc """
  Initializes the agent state with the given agent-specific options.

  Should be implemented in a way that the result would be semantically
  the same as if `c:reset/2` was called in the end of the function.

  As a suggestion, only initialize fixed values here, and delegate the
  rest to `c:reset/2` internally.
  """
  @callback init(random_key :: Nx.t(), opts :: keyword) :: {t(), random_key :: Nx.t()}

  @doc """
  Resets any values that vary between sessions (which would be episodes
  for episodic tasks) for the agent state.
  """
  @callback reset(random_key :: Nx.t(), rl_state :: t) :: {t(), random_key :: Nx.t()}

  @doc """
  Selects the action to be taken.
  """
  @callback select_action(rl_state, iteration :: Nx.t()) :: {action :: Nx.t(), rl_state}

  @doc """
  Can be used to record the observation in an experience replay buffer.

  If this is not desired, just make this function return the first argument unchanged.
  """
  @callback record_observation(
              rl_state,
              action :: Nx.t(),
              reward :: Nx.t(),
              is_terminal :: Nx.t(),
              next_rl_state :: rl_state
            ) :: rl_state
  @callback optimize_model(rl_state) :: rl_state
end
