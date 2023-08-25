defmodule Rein.Agents.QLearning do
  @moduledoc """
  Q-Learning implementation.

  This implementation uses epsilon-greedy sampling
  for exploration, and doesn't contemplate any kind
  of target network.
  """

  import Nx.Defn

  @behaviour Rein.Agent

  @derive {Nx.Container,
           containers: [
             :q_matrix,
             :observation
           ],
           keep: [
             :num_actions,
             :environment_to_state_vector_fn,
             :learning_rate,
             :gamma,
             :exploration_eps,
             :state_space_shape
           ]}

  defstruct [
    :q_matrix,
    :observation,
    :num_actions,
    :environment_to_state_vector_fn,
    :learning_rate,
    :gamma,
    :exploration_eps,
    :state_space_shape
  ]

  @impl true
  def init(random_key, opts \\ []) do
    opts =
      Keyword.validate!(opts, [
        :state_space_shape,
        :num_actions,
        :environment_to_state_vector_fn,
        :learning_rate,
        :gamma,
        :exploration_eps
      ])

    state_space_shape = opts[:state_space_shape]
    num_actions = opts[:num_actions]

    # q_matrix is a tensor in which the state_vector indexes the axis 0
    # as linear indices, and axis 1 is the action axis
    {q_matrix, random_key} =
      Nx.Random.uniform(random_key, -0.1, 0.1,
        shape: {Tuple.product(state_space_shape), num_actions}
      )

    state = %__MODULE__{
      q_matrix: q_matrix,
      environment_to_state_vector_fn: opts[:environment_to_state_vector_fn],
      learning_rate: opts[:learning_rate],
      gamma: opts[:gamma],
      exploration_eps: opts[:exploration_eps],
      state_space_shape: state_space_shape,
      num_actions: num_actions
    }

    reset(random_key, state)
  end

  @impl true
  def reset(random_key, %Rein{agent_state: agent_state}), do: reset(random_key, agent_state)

  def reset(random_key, %__MODULE__{} = agent_state) do
    zero = Nx.tensor(0, type: :f32)

    observation = %{
      action: 0,
      state: 0,
      next_state: 0,
      reward: zero
    }

    {%__MODULE__{agent_state | observation: observation}, random_key}
  end

  @impl true
  defn select_action(
         %Rein{random_key: random_key, agent_state: agent_state} = state,
         _iteration
       ) do
    {sample, random_key} = Nx.Random.uniform(random_key, shape: {})

    state_vector = agent_state.environment_to_state_vector_fn.(state.environment_state)

    {action, random_key} =
      if sample < agent_state.exploration_eps do
        Nx.Random.randint(random_key, 0, agent_state.num_actions, shape: {})
      else
        idx = state_vector_to_index(state_vector, agent_state.state_space_shape)

        action = Nx.argmax(agent_state.q_matrix[idx])

        {action, random_key}
      end

    {action, %{state | random_key: random_key}}
  end

  @impl true
  deftransform record_observation(
                 %{
                   environment_state: env_state,
                   agent_state: %{
                     environment_to_state_vector_fn: environment_to_state_vector_fn,
                     state_space_shape: state_space_shape
                   }
                 },
                 action,
                 reward,
                 _is_terminal,
                 %{environment_state: next_env_state} = state
               ) do
    observation = %{
      state:
        env_state
        |> environment_to_state_vector_fn.()
        |> state_vector_to_index(state_space_shape),
      next_state:
        next_env_state
        |> environment_to_state_vector_fn.()
        |> state_vector_to_index(state_space_shape),
      reward: reward,
      action: action
    }

    put_in(state.agent_state.observation, observation)
  end

  @impl true
  defn optimize_model(rl_state) do
    %{
      observation: %{
        state: state,
        next_state: next_state,
        reward: reward,
        action: action
      },
      q_matrix: q_matrix,
      gamma: gamma,
      learning_rate: learning_rate
    } = rl_state.agent_state

    # Q_table[current_state, action] =
    # (1-lr) * Q_table[current_state, action] +
    # lr*(reward + gamma*max(Q_table[next_state,:]))

    q =
      (1 - learning_rate) * q_matrix[[state, action]] +
        learning_rate * (reward + gamma * Nx.reduce_max(q_matrix[next_state]))

    q_matrix = Nx.indexed_put(q_matrix, Nx.stack([state, action]), q)

    %{rl_state | agent_state: %{rl_state.agent_state | q_matrix: q_matrix}}
  end

  deftransformp state_vector_to_index(state_vector, shape) do
    {linear_indices_offsets_list, _} =
      shape
      |> Tuple.to_list()
      |> Enum.reverse()
      |> Enum.reduce({[], 1}, fn x, {acc, multiplier} ->
        {[multiplier | acc], multiplier * x}
      end)

    linear_indices_offsets = Nx.tensor(linear_indices_offsets_list)

    Nx.dot(state_vector, linear_indices_offsets)
  end
end
