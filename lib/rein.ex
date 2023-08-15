defmodule Rein do
  @moduledoc """
  Reinforcement Learning training and inference framework
  """

  import Nx.Defn

  @derive {Nx.Container,
           containers: [
             :agent_state,
             :environment_state,
             :random_key,
             :iteration,
             :episode,
             :trajectory
           ],
           keep: [:agent_opts]}
  defstruct [
    :agent,
    :agent_state,
    :environment,
    :environment_state,
    :random_key,
    :iteration,
    :episode,
    :trajectory,
    :agent_opts
  ]

  @spec train(
          {environment :: module, init_opts :: keyword()},
          {agent :: module, init_opts :: keyword},
          epoch_completed_callback :: (map() -> :ok),
          opts :: keyword()
        ) :: Axon.Loop.t()
  def train(
        {environment, environment_init_opts},
        {agent, agent_init_opts},
        epoch_completed_callback,
        state_to_trajectory_fn,
        opts \\ []
      ) do
    opts =
      Keyword.validate!(opts, [
        :random_key,
        :max_iter,
        :model_name,
        :checkpoint_path,
        checkpoint_serialization_fn: &Nx.serialize/1,
        accumulated_episodes: 0,
        num_episodes: 100
      ])

    random_key = opts[:random_key] || Nx.Random.key(System.system_time())
    max_iter = opts[:max_iter]
    num_episodes = opts[:num_episodes]
    model_name = opts[:model_name]

    {init_agent_state, random_key} = agent.init(random_key, agent_init_opts)

    episode = Nx.tensor(opts[:accumulated_episodes], type: :s64)
    iteration = Nx.tensor(0, type: :s64)

    # TO-DO: needs Nx 0.6
    # [episode, iteration, _] =
    #   Nx.broadcast_vectors([episode, iteration, random_key], align_ranks: false)

    {environment_state, random_key} = environment.init(random_key, environment_init_opts)

    {agent_state, agent_opts, random_key} =
      case agent.reset(random_key, %__MODULE__{
             environment_state: environment_state,
             agent: agent,
             agent_state: init_agent_state,
             episode: episode
           }) do
        {s, o, k} -> {s, o, k}
        {s, k} -> {s, [], k}
      end

    initial_state = %__MODULE__{
      agent: agent,
      agent_state: agent_state,
      agent_opts: agent_opts,
      environment: environment,
      environment_state: environment_state,
      random_key: random_key,
      iteration: iteration,
      episode: episode
    }

    %Nx.Tensor{shape: {trajectory_points}} = state_to_trajectory_fn.(initial_state)

    trajectory = Nx.broadcast(Nx.tensor(:nan, type: :f32), {max_iter + 1, trajectory_points})
    [trajectory, _] = Nx.broadcast_vectors([trajectory, random_key], align_ranks: false)

    initial_state = %__MODULE__{initial_state | trajectory: trajectory}

    loop(
      agent,
      environment,
      initial_state,
      epoch_completed_callback: epoch_completed_callback,
      state_to_trajectory_fn: state_to_trajectory_fn,
      num_episodes: num_episodes,
      max_iter: max_iter,
      model_name: model_name,
      checkpoint_path: opts[:checkpoint_path]
    )
  end

  defp loop(agent, environment, initial_state, opts) do
    epoch_completed_callback = Keyword.fetch!(opts, :epoch_completed_callback)
    state_to_trajectory_fn = Keyword.fetch!(opts, :state_to_trajectory_fn)
    num_episodes = Keyword.fetch!(opts, :num_episodes)
    max_iter = Keyword.fetch!(opts, :max_iter)

    loop_fn = &batch_step(&1, &2, agent, environment, state_to_trajectory_fn)

    loop_fn
    |> Axon.Loop.loop()
    |> Axon.Loop.handle_event(
      :epoch_started,
      &{:continue,
       %{&1 | step_state: reset_state(&1.step_state, agent, environment, state_to_trajectory_fn)}}
    )
    |> Axon.Loop.handle_event(:epoch_completed, fn loop_state ->
      loop_state = tap(loop_state, epoch_completed_callback)
      {:continue, loop_state}
    end)
    |> Axon.Loop.handle_event(:iteration_completed, fn loop_state ->
      is_terminal =
        loop_state.step_state.environment_state.is_terminal
        |> Nx.devectorize()
        |> Nx.all()
        |> Nx.to_number()

      if is_terminal == 1 do
        {:halt_epoch, loop_state}
      else
        {:continue, loop_state}
      end
    end)
    |> Axon.Loop.handle_event(:epoch_halted, fn loop_state ->
      loop_state = tap(loop_state, epoch_completed_callback)
      {:halt_epoch, loop_state}
    end)
    |> Axon.Loop.checkpoint(
      event: :epoch_halted,
      filter: [every: 1000],
      file_pattern: fn %{epoch: epoch} ->
        "checkpoint_" <> opts[:model_name] <> "_#{epoch}.ckpt"
      end,
      path: opts[:checkpoint_path],
      serialize_step_state: opts[:checkpoint_serialization_fn]
    )
    |> Axon.Loop.checkpoint(
      event: :epoch_completed,
      filter: [every: 100],
      file_pattern: fn %{epoch: epoch} ->
        "checkpoint_" <> opts[:model_name] <> "_#{epoch}.ckpt"
      end,
      path: opts[:checkpoint_path],
      serialize_step_state: opts[:checkpoint_serialization_fn]
    )
    |> Axon.Loop.run(Stream.cycle([Nx.tensor(1)]), initial_state,
      iterations: max_iter,
      epochs: num_episodes,
      debug: Application.get_env(:rein, :debug, false)
    )
  end

  defp reset_state(
         %__MODULE__{
           environment_state: environment_state,
           random_key: random_key
         } = loop_state,
         agent,
         environment,
         state_to_trajectory_fn
       ) do
    {environment_state, random_key} = environment.reset(random_key, environment_state)

    {agent_state, agent_opts, random_key} =
      case agent.reset(random_key, %{loop_state | environment_state: environment_state}) do
        {state, opts, key} -> {state, opts, key}
        {state, key} -> {state, [], key}
      end

    state = %{
      loop_state
      | agent_state: agent_state,
        agent_opts: agent_opts,
        environment_state: environment_state,
        random_key: random_key,
        trajectory: Nx.broadcast(Nx.tensor(:nan, type: :f32), loop_state.trajectory),
        episode: Nx.add(loop_state.episode, 1),
        iteration: Nx.tensor(0, type: :s64)
    }

    persist_trajectory(state, state_to_trajectory_fn)
  end

  defp batch_step(
         _inputs,
         prev_state,
         agent,
         environment,
         state_to_trajectory_fn
       ) do
    {action, state} = agent.select_action(prev_state, prev_state.iteration)

    %{environment_state: %{reward: reward, is_terminal: is_terminal}} =
      state = environment.apply_action(state, action)

    prev_state
    |> agent.record_observation(
      action,
      reward,
      is_terminal,
      state
    )
    |> agent.optimize_model()
    |> persist_trajectory(state_to_trajectory_fn)
  end

  defnp persist_trajectory(
          %__MODULE__{trajectory: trajectory, iteration: iteration} = step_state,
          state_to_trajectory_fn
        ) do
    updates = state_to_trajectory_fn.(step_state)

    %Nx.Tensor{shape: {_, num_points}} = trajectory

    idx =
      Nx.concatenate([Nx.broadcast(iteration, {num_points, 1}), Nx.iota({num_points, 1})],
        axis: 1
      )

    trajectory = Nx.indexed_put(trajectory, idx, updates)
    %{step_state | trajectory: trajectory, iteration: iteration + 1}
  end
end
