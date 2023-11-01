defmodule Rein.Environments.Turtleworld do
  @moduledoc """
  Environment based on Turtle Graphics with 1 continuous action.

  [Turtle Graphics](https://en.wikipedia.org/wiki/Turtle_graphics) are a way of
  graphing using a cursor that can move around the screen and draw lines. Just like
  trying to draw on paper without lifting the pen from the paper.

  In this environment, the agent starts at a random position at the base of the grid.
  Unlike `Rein.Environments.Gridworld` where the agent can move freely in 4 directions,
  this environment has 1 continuous action that turns the agent left or right.
  The agent will always move after turning with a continuous speed of 0.1, and each
  iteration comprises a $\\delta t$ of 1.0, so the agent will move 0.1 units per iteration.
  """
  import Nx.Defn

  @behaviour Rein.Environment

  @derive {Nx.Container,
           containers: [
             :x,
             :y,
             :direction,
             :target_x,
             :target_y,
             :reward,
             :is_terminal,
             :possible_targets,
             :has_reached_target,
             :remaining_seconds
           ],
           keep: []}
  defstruct [
    :x,
    :y,
    :direction,
    :target_x,
    :target_y,
    :reward,
    :is_terminal,
    :possible_targets,
    :has_reached_target,
    :remaining_seconds
  ]

  @min_x 0
  @max_x 5
  @min_y 0
  @max_y 5

  @speed 0.1
  @dt 1.0

  def bounding_box, do: {@min_x, @max_x, @min_y, @max_y}

  # x, y, direction, target_x, target_y, has_reached_target, distance_norm
  @doc "The size of the state vector returned by `as_state_vector/1`"
  def state_vector_size, do: 7

  # left or right
  def num_actions, do: 1

  @impl true
  def init(random_key, opts) do
    opts = Keyword.validate!(opts, [:possible_targets])

    possible_targets =
      opts[:possible_targets] || raise ArgumentError, "missing option :possible_targets"

    reset(random_key, %__MODULE__{possible_targets: possible_targets})
  end

  @impl true
  def reset(random_key, %__MODULE__{} = state) do
    {x, random_key} = Nx.Random.uniform(random_key, @min_x, @max_x)

    # possible_targets is a {n, 2} tensor that contains targets that we want to sample from
    # this is so we avoid retraining every episode on the same target, which can lead to
    # overfitting
    {target, random_key} =
      Nx.Random.choice(random_key, state.possible_targets, samples: 1, axis: 0)

    target = Nx.reshape(target, {2})

    zero = Nx.tensor(0, type: :f32)

    [x, y, target_x, target_y, zero_bool, _key] =
      Nx.broadcast_vectors([x, zero, target[0], target[1], random_key, Nx.u8(0)])

    state = %{
      state
      | x: x,
        y: y,
        target_x: target_x,
        target_y: target_y,
        direction: zero,
        reward: zero,
        is_terminal: zero_bool,
        has_reached_target: zero_bool,
        remaining_seconds: zero
    }

    {state, random_key}
  end

  @impl true
  defn apply_action(state, action) do
    %{x: x, y: y, direction: direction} = env = state.environment_state

    dtheta = action * Nx.Constants.pi()

    new_x = x + @speed * Nx.cos(dtheta) * @dt
    new_y = y + @speed * Nx.sin(dtheta) * @dt

    new_direction = wrap_phase(direction + dtheta)

    new_env = %{
      env
      | x: Nx.clip(new_x, @min_x, @max_x),
        y: Nx.clip(new_y, @min_y, @max_y),
        direction: new_direction,
        remaining_seconds: env.remaining_seconds - @dt
    }

    updated_env =
      new_env
      |> is_terminal_state()
      |> calculate_reward()

    %{state | environment_state: updated_env}
  end

  defnp wrap_phase(theta) do
    two_pi = 2 * Nx.Constants.pi()
    Nx.remainder(Nx.remainder(theta, two_pi) + two_pi, two_pi)
  end

  defnp calculate_reward(env) do
    distance = Nx.abs(env.target_x - env.x) + Nx.abs(env.target_y - env.y)
    reward = -1.0 * distance

    %{env | reward: reward}
  end

  defnp is_terminal_state(env) do
    has_reached_target = has_reached_target(env)
    out_of_bounds = env.x < @min_x or env.x > @max_x or env.y < @min_y or env.y > @max_y

    is_terminal = has_reached_target or out_of_bounds or env.remaining_seconds < 1

    %__MODULE__{env | is_terminal: is_terminal, has_reached_target: has_reached_target}
  end

  defnp has_reached_target(%__MODULE__{x: x, y: y, target_x: target_x, target_y: target_y}) do
    (target_x - x) ** 2 + (target_y - y) ** 2 <= 1
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
