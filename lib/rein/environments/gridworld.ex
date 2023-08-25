defmodule Rein.Environments.Gridworld do
  @moduledoc """
  Gridworld environment with 4 discrete actions.

  Gridworld is an environment where the agent
  aims to reach a given target from a collection
  of possible targets, only being able to choose
  1 of 4 actions: up, down, left and right.
  """
  import Nx.Defn

  @behaviour Rein.Environment

  @derive {Nx.Container,
           containers: [
             :x,
             :y,
             :prev_x,
             :prev_y,
             :target_x,
             :target_y,
             :reward,
             :is_terminal,
             :possible_targets,
             :has_reached_target
           ],
           keep: []}
  defstruct [
    :x,
    :y,
    :prev_x,
    :prev_y,
    :target_x,
    :target_y,
    :reward,
    :is_terminal,
    :possible_targets,
    :has_reached_target
  ]

  @min_x 0
  @max_x 4
  @min_y 0
  @max_y 4

  def bounding_box, do: {@min_x, @max_x, @min_y, @max_y}

  # x, y, target_x, target_y, has_reached_target, distance_norm
  @doc "The size of the state vector returned by `as_state_vector/1`"
  def state_vector_size, do: 6

  # up, down, left, right
  def num_actions, do: 4

  @impl true
  def init(random_key, opts) do
    opts = Keyword.validate!(opts, [:possible_targets])

    possible_targets =
      opts[:possible_targets] || raise ArgumentError, "missing option :possible_targets"

    reset(random_key, %__MODULE__{possible_targets: possible_targets})
  end

  @impl true
  def reset(random_key, %__MODULE__{} = state) do
    reward = Nx.tensor(0, type: :f32)
    {x, random_key} = Nx.Random.randint(random_key, @min_x, @max_x)

    # possible_targets is a {n, 2} tensor that contains targets that we want to sample from
    # this is so we avoid retraining every episode on the same target, which can lead to
    # overfitting
    {target, random_key} =
      Nx.Random.choice(random_key, state.possible_targets, samples: 1, axis: 0)

    target = Nx.reshape(target, {2})

    target_x = target[0]
    target_y = target[1]

    y = Nx.tensor(0, type: :s64)

    # [x, y, target_x, target_y, zero_bool, _key] =
    #   Nx.broadcast_vectors([x, y, target[0], target[1], random_key, Nx.u8(0)])

    state = %{
      state
      | x: x,
        y: y,
        prev_x: x,
        prev_y: y,
        target_x: target_x,
        target_y: target_y,
        reward: reward,
        is_terminal: Nx.u8(0),
        has_reached_target: Nx.u8(0)
    }

    {state, random_key}
  end

  @impl true
  defn apply_action(state, action) do
    %{x: x, y: y} = env = state.environment_state

    # 0: up, 1: down, 2: right, 3: left
    {new_x, new_y} =
      cond do
        action == 0 ->
          {x, y + 1}

        action == 1 ->
          {x, y - 1}

        action == 2 ->
          {x + 1, y}

        true ->
          {x - 1, y}
      end

    new_env = %{
      env
      | x: Nx.clip(new_x, @min_x, @max_x),
        y: Nx.clip(new_y, @min_y, @max_y),
        prev_x: x,
        prev_y: y
    }

    updated_env =
      new_env
      |> is_terminal_state()
      |> calculate_reward()

    %{state | environment_state: updated_env}
  end

  defnp calculate_reward(env) do
    distance = Nx.abs(env.target_x - env.x) + Nx.abs(env.target_y - env.y)
    reward = -1.0 * distance

    %{env | reward: reward}
  end

  defnp is_terminal_state(env) do
    has_reached_target = has_reached_target(env)
    out_of_bounds = env.x < @min_x or env.x > @max_x or env.y < @min_y or env.y > @max_y

    is_terminal = has_reached_target or out_of_bounds

    %__MODULE__{env | is_terminal: is_terminal, has_reached_target: has_reached_target}
  end

  defnp has_reached_target(%__MODULE__{x: x, y: y, target_x: target_x, target_y: target_y}) do
    target_x == x and target_y == y
  end

  defnp normalize(v, min, max), do: (v - min) / (max - min)

  @doc """
  Default function for turning the environment into a vector representation.
  """
  defn as_state_vector(%{
         x: x,
         y: y,
         target_x: target_x,
         target_y: target_y,
         has_reached_target: has_reached_target
       }) do
    x = normalize(x, @min_x, @max_x)
    y = normalize(y, @min_y, @max_y)

    target_x = normalize(target_x, @min_x, @max_x)
    target_y = normalize(target_y, @min_y, @max_y)

    # max distance is sqrt(1 ** 2 + 1 ** 2) = sqrt(2)
    distance_norm = Nx.sqrt((x - target_x) ** 2 + (y - target_y) ** 2) / Nx.sqrt(2)

    Nx.stack([
      x,
      y,
      target_x,
      target_y,
      has_reached_target,
      distance_norm
    ])
    |> Nx.new_axis(0)
  end
end
