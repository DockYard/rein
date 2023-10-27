defmodule Rein.Agents.SAC do
  @moduledoc """
  Soft Actor-Critic implementation.

  This assumes that the Actor network will output `{nil, num_actions, 2}`,
  where for each action they output the $\\mu$ and $\\sigma$ values of a random
  normal distribution, and that the Critic network accepts `"actions"` input with
  shape `{nil, num_actions}`, where the action is calculated by sampling from
  said random distribution.

  Actions are deemed to be in a continuous space of type `:f32`.

  The Dual Q implementation utilizes two copies of the critic network, `critic1` and `critic2`,
  each with their own separate target network.

  Vectorized axes from `:random_key` are propagated normally throughout
  the agent state for parallel simulations, but all samples are stored in the same
  circular buffer. After all simulations have ran, the optimization steps are run
  on a sample space consisting of all previous experiences, including all of the
  parallel simulations that have just finished executing.
  """
  import Nx.Defn

  import Nx.Constants, only: [pi: 1]

  alias Rein.Utils.CircularBuffer

  @behaviour Rein.Agent

  @derive {Nx.Container,
           containers: [
             :actor_params,
             :actor_target_params,
             :critic1_params,
             :critic2_params,
             :critic1_target_params,
             :critic2_target_params,
             :experience_replay_buffer,
             :loss,
             :loss_denominator,
             :total_reward,
             :actor_optimizer_state,
             :critic1_optimizer_state,
             :critic2_optimizer_state,
             :action_lower_limit,
             :action_upper_limit,
             :gamma,
             :tau,
             :state_features_memory,
             :log_entropy_coefficient,
             :log_entropy_coefficient_optimizer_state,
             :target_entropy
           ],
           keep: [
             :environment_to_state_features_fn,
             :actor_predict_fn,
             :critic_predict_fn,
             :num_actions,
             :actor_optimizer_update_fn,
             :critic_optimizer_update_fn,
             :batch_size,
             :training_frequency,
             :input_entry_size,
             :reward_scale,
             :log_entropy_coefficient_optimizer_update_fn,
             :train_log_entropy_coefficient
           ]}

  defstruct [
    :num_actions,
    :actor_params,
    :actor_target_params,
    :critic1_params,
    :critic2_params,
    :critic1_target_params,
    :critic2_target_params,
    :actor_predict_fn,
    :critic_predict_fn,
    :experience_replay_buffer,
    :environment_to_state_features_fn,
    :gamma,
    :tau,
    :batch_size,
    :training_frequency,
    :actor_optimizer_state,
    :critic1_optimizer_state,
    :critic2_optimizer_state,
    :action_lower_limit,
    :action_upper_limit,
    :loss,
    :loss_denominator,
    :total_reward,
    :actor_optimizer_update_fn,
    :critic_optimizer_update_fn,
    :state_features_memory,
    :input_entry_size,
    :log_entropy_coefficient,
    :reward_scale,
    :train_log_entropy_coefficient,
    :log_entropy_coefficient_optimizer_update_fn,
    :log_entropy_coefficient_optimizer_state,
    :target_entropy
  ]

  @impl true
  def init(random_key, opts \\ []) do
    expected_opts = [
      :actor_net,
      :critic_net,
      :environment_to_state_features_fn,
      :state_features_memory_to_input_fn,
      :state_features_size,
      :actor_optimizer,
      :critic_optimizer,
      :entropy_coefficient_optimizer,
      reward_scale: 1,
      state_features_memory_length: 1,
      gamma: 0.99,
      experience_replay_buffer_max_size: 100_000,
      tau: 0.005,
      batch_size: 64,
      training_frequency: 32,
      action_lower_limit: -1.0,
      action_upper_limit: 1.0,
      entropy_coefficient: 0.2,
      saved_state: %{}
    ]

    opts = Keyword.validate!(opts, expected_opts)

    # TO-DO: use NimbleOptions
    expected_opts
    |> Enum.filter(fn x -> is_atom(x) or (is_tuple(x) and is_nil(elem(x, 1))) end)
    |> Enum.reject(fn k ->
      k in [:state_features_memory, :experience_replay_buffer, :entropy_coefficient_optimizer]
    end)
    |> Enum.reduce(opts, fn
      k, opts ->
        case List.keytake(opts, k, 0) do
          {{^k, _}, opts} -> opts
          nil -> raise ArgumentError, "missing option #{k}"
        end
    end)
    |> Enum.each(fn {k, v} ->
      if is_nil(v) do
        raise ArgumentError, "option #{k} cannot be nil"
      end
    end)

    {actor_optimizer_init_fn, actor_optimizer_update_fn} = opts[:actor_optimizer]
    {critic_optimizer_init_fn, critic_optimizer_update_fn} = opts[:critic_optimizer]

    log_entropy_coefficient = :math.log(opts[:entropy_coefficient])

    {train_log_entropy_coefficient, log_entropy_coefficient_optimizer_init_fn,
     log_entropy_coefficient_optimizer_update_fn} =
      case opts[:entropy_coefficient_optimizer] do
        {init, upd} -> {true, init, upd}
        _ -> {false, fn _ -> 0 end, 0}
      end

    actor_net = opts[:actor_net]
    critic_net = opts[:critic_net]

    environment_to_state_features_fn = opts[:environment_to_state_features_fn]
    state_features_memory_to_input_fn = opts[:state_features_memory_to_input_fn]

    {actor_init_fn, actor_predict_fn} = Axon.build(actor_net, seed: 0)
    {critic_init_fn, critic_predict_fn} = Axon.build(critic_net, seed: 1)

    actor_predict_fn = fn random_key, params, state_features_memory ->
      action_distribution_vector =
        actor_predict_fn.(params, state_features_memory_to_input_fn.(state_features_memory))

      mu = action_distribution_vector[[.., .., 0]]
      log_stddev = action_distribution_vector[[.., .., 1]]

      stddev = Nx.exp(log_stddev)

      eps_shape = Nx.shape(stddev)

      # Nx.Random.normal is treated as a constant, so we obtain `eps` from a mean-0 stddev-1
      # normal distribution and scale it by our stddev below to obtain our sample in a way that
      # the grads a propagated through properly.
      {eps, random_key} = Nx.Random.normal(random_key, shape: eps_shape)

      pre_squash_action = Nx.add(mu, Nx.multiply(stddev, eps))
      action = Nx.tanh(pre_squash_action)

      log_probability = action_log_probability(mu, stddev, log_stddev, pre_squash_action, action)

      {action, log_probability, random_key}
    end

    critic_predict_fn = fn params, state_features_memory, action_vector ->
      input =
        state_features_memory
        |> state_features_memory_to_input_fn.()
        |> Map.put("actions", action_vector)

      critic_predict_fn.(params, input)
    end

    input_template = input_template(actor_net)

    case input_template do
      %{"actions" => _} ->
        raise ArgumentError,
              "the input template for the actor_network must not contain the reserved key \"actions\""

      _ ->
        :ok
    end

    {1, num_actions, 2} = Axon.get_output_shape(actor_net, input_template)

    critic_template = input_template(critic_net)

    case critic_template do
      %{"actions" => action_input} ->
        action_input = %{action_input | vectorized_axes: []}

        unless action_input != Nx.template({nil, num_actions}, :f32) do
          raise ArgumentError,
                "the critic network must accept the \"actions\" input with shape {nil, #{num_actions}} and type :f32, got input template: #{critic_template}"
        end

        critic_template = Map.delete(critic_template, "actions")

        if critic_template != input_template do
          raise ArgumentError,
                "the critic network must have the same input template as the actor network + the \"action\" input"
        end

      _ ->
        :ok
    end

    log_entropy_coefficient_optimizer_state =
      log_entropy_coefficient_optimizer_init_fn.(log_entropy_coefficient)

    actor_params = actor_init_fn.(input_template, %{})
    actor_optimizer_state = actor_optimizer_init_fn.(actor_params)

    actor_target_params = actor_init_fn.(input_template, %{})

    critic1_params = critic_init_fn.(critic_template, %{})
    critic2_params = critic_init_fn.(critic_template, %{})

    critic1_target_params = critic_init_fn.(critic_template, %{})
    critic2_target_params = critic_init_fn.(critic_template, %{})

    critic1_optimizer_state = critic_optimizer_init_fn.(critic1_target_params)
    critic2_optimizer_state = critic_optimizer_init_fn.(critic2_target_params)

    state_features_size = opts[:state_features_size]

    total_reward = loss = loss_denominator = Nx.tensor(0, type: :f32)
    experience_replay_buffer_max_size = opts[:experience_replay_buffer_max_size]
    state_features_memory_length = opts[:state_features_memory_length]
    input_entry_size = state_features_size * state_features_memory_length

    {exp_replay_buffer, random_key} =
      if buffer = opts[:experience_replay_buffer] do
        {buffer, random_key}
      else
        {random_data_1, random_key} =
          Nx.Random.normal(random_key, 0, 10,
            shape: {experience_replay_buffer_max_size, input_entry_size + num_actions}
          )

        init_reward = Nx.broadcast(-1.0e-8, {experience_replay_buffer_max_size, 1})

        {random_data_2, random_key} =
          Nx.Random.normal(random_key, 0, 10,
            shape: {experience_replay_buffer_max_size, state_features_size + 1}
          )

        data =
          [random_data_1, init_reward, random_data_2]
          |> Nx.concatenate(axis: 1)
          |> then(&Nx.revectorize(&1, [], target_shape: Tuple.insert_at(&1.shape, 0, :auto)))

        buffer = %CircularBuffer{
          data: data[[0, .., ..]],
          index: 0,
          size: 0
        }

        {buffer, random_key}
      end

    state = %__MODULE__{
      input_entry_size: input_entry_size,
      log_entropy_coefficient_optimizer_state: log_entropy_coefficient_optimizer_state,
      log_entropy_coefficient_optimizer_update_fn: log_entropy_coefficient_optimizer_update_fn,
      target_entropy: -num_actions,
      train_log_entropy_coefficient: train_log_entropy_coefficient,
      state_features_memory:
        opts[:state_features_memory] ||
          CircularBuffer.new({state_features_memory_length, state_features_size}),
      num_actions: num_actions,
      actor_params: actor_params,
      actor_target_params: actor_target_params,
      critic1_params: critic1_params,
      critic2_params: critic2_params,
      critic1_target_params: critic1_target_params,
      critic2_target_params: critic2_target_params,
      actor_predict_fn: actor_predict_fn,
      critic_predict_fn: critic_predict_fn,
      experience_replay_buffer: exp_replay_buffer,
      environment_to_state_features_fn: environment_to_state_features_fn,
      gamma: opts[:gamma],
      tau: opts[:tau],
      batch_size: opts[:batch_size],
      training_frequency: opts[:training_frequency],
      total_reward: total_reward,
      loss: loss,
      loss_denominator: loss_denominator,
      actor_optimizer_update_fn: actor_optimizer_update_fn,
      critic_optimizer_update_fn: critic_optimizer_update_fn,
      actor_optimizer_state: actor_optimizer_state,
      critic1_optimizer_state: critic1_optimizer_state,
      critic2_optimizer_state: critic2_optimizer_state,
      action_lower_limit: opts[:action_lower_limit],
      action_upper_limit: opts[:action_upper_limit],
      log_entropy_coefficient: Nx.log(opts[:entropy_coefficient]),
      reward_scale: opts[:reward_scale]
    }

    saved_state =
      (opts[:saved_state] || %{})
      |> Map.take(Map.keys(%__MODULE__{}) -- [:__struct__])
      |> Enum.filter(fn {_, v} -> v && not is_function(v) end)
      |> Map.new()

    state = Map.merge(state, saved_state)

    case random_key.vectorized_axes do
      [] ->
        {state, random_key}

      _ ->
        vectorizable_paths = [
          [Access.key(:loss)],
          [Access.key(:loss_denominator)],
          [Access.key(:total_reward)],
          [Access.key(:state_features_memory), Access.key(:data)],
          [Access.key(:state_features_memory), Access.key(:index)],
          [Access.key(:state_features_memory), Access.key(:size)]
        ]

        vectorized_state =
          Enum.reduce(vectorizable_paths, state, fn path, state ->
            update_in(state, path, fn value ->
              [value, _] = Nx.broadcast_vectors([value, random_key], align_ranks: false)
              value
            end)
          end)

        {vectorized_state, random_key}
    end
  end

  defp input_template(model) do
    model
    |> Axon.get_inputs()
    |> Map.new(fn {name, shape} ->
      [nil | shape] = Tuple.to_list(shape)
      shape = List.to_tuple([1 | shape])
      {name, Nx.template(shape, :f32)}
    end)
  end

  @impl true
  def reset(random_key, %Rein{
        environment_state: env,
        agent_state: state
      }) do
    [zero, _] = Nx.broadcast_vectors([Nx.tensor(0, type: :f32), random_key], align_ranks: false)
    total_reward = loss = loss_denominator = zero

    init_state_features = state.environment_to_state_features_fn.(env)

    {n, _} = state.state_features_memory.data.shape

    zero = Nx.as_type(zero, :s64)

    state_features_memory = %{
      state.state_features_memory
      | data: Nx.tile(init_state_features, [n, 1]),
        index: zero,
        size: Nx.add(n, zero)
    }

    {%{
       state
       | total_reward: total_reward,
         loss: loss,
         loss_denominator: loss_denominator,
         state_features_memory: state_features_memory
     }, random_key}
  end

  @impl true
  defn select_action(
         %Rein{random_key: random_key, agent_state: agent_state} = state,
         _iteration
       ) do
    %__MODULE__{
      actor_params: actor_params,
      actor_predict_fn: actor_predict_fn,
      environment_to_state_features_fn: environment_to_state_features_fn,
      state_features_memory: state_features_memory,
      action_lower_limit: action_lower_limit,
      action_upper_limit: action_upper_limit
    } = agent_state

    state_features = environment_to_state_features_fn.(state.environment_state)

    state_features_memory = CircularBuffer.append(state_features_memory, state_features)

    {action_vector, _logprob, random_key} =
      actor_predict_fn.(
        random_key,
        actor_params,
        CircularBuffer.ordered_data(state_features_memory)
      )

    clipped_action_vector =
      action_vector
      |> Nx.max(action_lower_limit)
      |> Nx.min(action_upper_limit)

    {clipped_action_vector,
     %{
       state
       | agent_state: %{
           agent_state
           | state_features_memory: state_features_memory
         },
         random_key: random_key
     }}
  end

  @impl true
  defn record_observation(
         %{
           agent_state: %__MODULE__{
             state_features_memory: state_features_memory,
             environment_to_state_features_fn: environment_to_state_features_fn,
             experience_replay_buffer: experience_replay_buffer,
             reward_scale: reward_scale
           }
         },
         action_vector,
         reward,
         is_terminal,
         %{environment_state: next_env_state} = state
       ) do
    next_state_features = environment_to_state_features_fn.(next_env_state)
    state_data = CircularBuffer.ordered_data(state_features_memory)

    reward = reward * reward_scale

    updates =
      Nx.concatenate([
        Nx.flatten(state_data),
        Nx.flatten(action_vector),
        Nx.new_axis(reward, 0),
        Nx.new_axis(is_terminal, 0),
        Nx.flatten(next_state_features)
      ])

    updates =
      Nx.revectorize(updates, [],
        target_shape: {:auto, Nx.axis_size(experience_replay_buffer.data, -1)}
      )

    experience_replay_buffer = CircularBuffer.append_multiple(experience_replay_buffer, updates)

    ensure_not_vectorized!(experience_replay_buffer.data)

    %{
      state
      | agent_state: %{
          state.agent_state
          | experience_replay_buffer: experience_replay_buffer,
            total_reward: state.agent_state.total_reward + reward
        }
    }
  end

  deftransformp ensure_not_vectorized!(t) do
    case t do
      %{vectorized_axes: []} ->
        :ok

      %{vectorized_axes: _vectorized_axes} ->
        raise "found unexpected vectorized axes"
    end
  end

  @impl true
  defn optimize_model(state) do
    %{
      batch_size: batch_size,
      training_frequency: training_frequency
    } = state.agent_state

    # Run training after all simulations have ended.
    is_terminal =
      state.environment_state.is_terminal
      |> Nx.devectorize()
      |> Nx.all()

    if is_terminal and state.agent_state.experience_replay_buffer.size > batch_size do
      train_loop(
        state,
        training_frequency * vectorized_axes(state.environment_state.is_terminal)
      )
    else
      state
    end
  end

  deftransformp vectorized_axes(t) do
    # flat_size is all entries, inclusing vectorized axes
    # size is just the non-vectorized part
    # So training frequency here is the number of vectorized axes,
    # i.e. we'll run one iteration per episode simulated
    div(Nx.flat_size(t), Nx.size(t))
  end

  deftransformp train_loop(state, training_frequency) do
    if training_frequency == 1 do
      train_loop_step(state)
    else
      train_loop_while(state, training_frequency: training_frequency)
    end
  end

  defnp train_loop_while(state, opts \\ []) do
    training_frequency = opts[:training_frequency]

    while state, _ <- 0..(training_frequency - 1)//1, unroll: false do
      train_loop_step(state)
    end
  end

  defnp train_loop_step(state) do
    {batch, random_key} = sample_experience_replay_buffer(state.random_key, state.agent_state)

    %{state | random_key: random_key}
    |> train(batch)
    |> soft_update_targets()
  end

  defnp train(state, batch) do
    %{
      agent_state: %__MODULE__{
        actor_params: actor_params,
        actor_target_params: actor_target_params,
        actor_predict_fn: actor_predict_fn,
        critic1_params: critic1_params,
        critic2_params: critic2_params,
        critic1_target_params: critic1_target_params,
        critic2_target_params: critic2_target_params,
        critic_predict_fn: critic_predict_fn,
        actor_optimizer_state: actor_optimizer_state,
        critic1_optimizer_state: critic1_optimizer_state,
        critic2_optimizer_state: critic2_optimizer_state,
        actor_optimizer_update_fn: actor_optimizer_update_fn,
        critic_optimizer_update_fn: critic_optimizer_update_fn,
        state_features_memory: state_features_memory,
        input_entry_size: input_entry_size,
        experience_replay_buffer: experience_replay_buffer,
        num_actions: num_actions,
        gamma: gamma,
        log_entropy_coefficient: log_entropy_coefficient,
        log_entropy_coefficient_optimizer_update_fn: log_entropy_coefficient_optimizer_update_fn,
        log_entropy_coefficient_optimizer_state: log_entropy_coefficient_optimizer_state,
        target_entropy: target_entropy,
        train_log_entropy_coefficient: train_log_entropy_coefficient
      },
      random_key: random_key
    } = state

    ks = Nx.Random.split(random_key)

    {random_key, k1} =
      case {Nx.flat_size(random_key), Nx.size(random_key)} do
        {s, s} ->
          {ks[0], ks[1]}

        _ ->
          random_key = ks[0]
          k1 = ks[1] |> Nx.devectorize() |> Nx.take(0)
          {random_key, k1}
      end

    batch_len = Nx.axis_size(batch, 0)
    {num_states, state_features_size} = state_features_memory.data.shape

    state_batch =
      batch
      |> Nx.slice_along_axis(0, input_entry_size, axis: 1)
      |> Nx.reshape({batch_len, num_states, state_features_size})

    action_batch = Nx.slice_along_axis(batch, input_entry_size, num_actions, axis: 1)
    reward_batch = Nx.slice_along_axis(batch, input_entry_size + num_actions, 1, axis: 1)

    is_terminal_batch = Nx.slice_along_axis(batch, input_entry_size + num_actions + 1, 1, axis: 1)

    # we only persisted the new state, so we need to manipulate the `state_batch` to get the actual state
    # with state memory
    next_state_batch =
      batch
      |> Nx.slice_along_axis(input_entry_size + num_actions + 2, state_features_size, axis: 1)
      |> Nx.reshape({batch_len, 1, state_features_size})

    next_state_batch =
      if num_states == 1 do
        next_state_batch
      else
        next_state_batch =
          [
            state_batch,
            next_state_batch
          ]
          |> Nx.concatenate(axis: 1)
          |> Nx.slice_along_axis(1, num_states, axis: 1)

        expected_shape = {batch_len, num_states, state_features_size}
        actual_shape = Nx.shape(next_state_batch)

        case {actual_shape, expected_shape} do
          {x, x} ->
            :ok

          {actual_shape, expected_shape} ->
            raise "incorrect size for next_state_batch, expected #{inspect(expected_shape)}, got: #{inspect(actual_shape)}"
        end

        next_state_batch
      end

    non_final_mask = not is_terminal_batch

    entropy_coefficient = stop_grad(Nx.exp(log_entropy_coefficient))

    ### Train critic_params

    {{critic_loss, k1}, {critic1_gradient, critic2_gradient}} =
      value_and_grad(
        {critic1_params, critic2_params},
        fn {critic1_params, critic2_params} ->
          # y_i = r_i + γ * min_{j=1,2} Q'(s_{i+1}, π(s_{i+1}|θ)|φ'_j)

          {target_actions, log_probability, k1} =
            actor_predict_fn.(k1, actor_target_params, next_state_batch)

          q1_target = critic_predict_fn.(critic1_target_params, next_state_batch, target_actions)
          q2_target = critic_predict_fn.(critic2_target_params, next_state_batch, target_actions)

          %{shape: {k, 1}} = q_target = stop_grad(Nx.min(q1_target, q2_target))

          next_log_prob =
            log_probability
            |> Nx.devectorize()
            |> Nx.sum(axes: [0])

          q_target = q_target - entropy_coefficient * next_log_prob

          %{shape: {m, 1}} = backup = reward_batch + gamma * non_final_mask * q_target

          # q values for each critic network
          %{shape: {n, 1}} = q1 = critic_predict_fn.(critic1_params, state_batch, action_batch)

          %{shape: {_n, 1}} = q2 = critic_predict_fn.(critic2_params, state_batch, action_batch)

          case {k, m, n} do
            {k, m, n} when m != n or m != k or n != k ->
              raise "shape mismatch for batch values"

            _ ->
              1
          end

          backup = Nx.devectorize(backup)
          critic1_loss = Nx.mean((backup - Nx.new_axis(q1, 0)) ** 2)
          critic2_loss = Nx.mean((backup - Nx.new_axis(q2, 0)) ** 2)

          {0.5 * Nx.add(critic1_loss, critic2_loss), k1}
        end,
        &elem(&1, 0)
      )

    {critic1_updates, critic1_optimizer_state} =
      critic_optimizer_update_fn.(critic1_gradient, critic1_optimizer_state, critic1_params)

    critic1_params = Polaris.Updates.apply_updates(critic1_params, critic1_updates)

    {critic2_updates, critic2_optimizer_state} =
      critic_optimizer_update_fn.(critic2_gradient, critic2_optimizer_state, critic2_params)

    critic2_params = Polaris.Updates.apply_updates(critic2_params, critic2_updates)

    ### Train Actor

    {{_, log_probs}, actor_gradient} =
      value_and_grad(
        actor_params,
        fn actor_params ->
          {actions, log_probs, _k1} =
            actor_predict_fn.(k1, actor_params, state_batch)

          q1 = critic_predict_fn.(critic1_params, state_batch, actions)

          q2 = critic_predict_fn.(critic2_params, state_batch, actions)

          q = Nx.min(q1, q2)

          {Nx.mean(entropy_coefficient * log_probs - q), log_probs}
        end,
        &elem(&1, 0)
      )

    {actor_updates, actor_optimizer_state} =
      actor_optimizer_update_fn.(actor_gradient, actor_optimizer_state, actor_params)

    actor_params = Polaris.Updates.apply_updates(actor_params, actor_updates)

    ### Train entropy_coefficient

    {log_entropy_coefficient, log_entropy_coefficient_optimizer_state} =
      case train_log_entropy_coefficient do
        false ->
          # entropy_coef is non-trainable
          {log_entropy_coefficient, log_entropy_coefficient_optimizer_state}

        true ->
          g =
            grad(log_entropy_coefficient, fn log_entropy_coefficient ->
              -Nx.mean(log_entropy_coefficient * (log_probs + target_entropy))
            end)

          {updates, log_entropy_coefficient_optimizer_state} =
            log_entropy_coefficient_optimizer_update_fn.(
              g,
              log_entropy_coefficient_optimizer_state,
              log_entropy_coefficient
            )

          log_entropy_coefficient =
            Polaris.Updates.apply_updates(log_entropy_coefficient, updates)

          {log_entropy_coefficient, log_entropy_coefficient_optimizer_state}
      end

    %{
      state
      | agent_state: %{
          state.agent_state
          | actor_params: actor_params,
            actor_optimizer_state: actor_optimizer_state,
            critic1_params: critic1_params,
            critic1_optimizer_state: critic1_optimizer_state,
            critic2_params: critic2_params,
            critic2_optimizer_state: critic2_optimizer_state,
            loss: state.agent_state.loss + critic_loss,
            loss_denominator: state.agent_state.loss_denominator + 1,
            experience_replay_buffer: experience_replay_buffer,
            log_entropy_coefficient: log_entropy_coefficient,
            log_entropy_coefficient_optimizer_state: log_entropy_coefficient_optimizer_state
        },
        random_key: random_key
    }
  end

  defnp soft_update_targets(state) do
    %{
      agent_state:
        %__MODULE__{
          actor_target_params: actor_target_params,
          actor_params: actor_params,
          critic1_params: critic1_params,
          critic2_params: critic2_params,
          critic1_target_params: critic1_target_params,
          critic2_target_params: critic2_target_params,
          tau: tau
        } = agent_state
    } = state

    merge_fn = &Nx.as_type(&1 * tau + &2 * (1 - tau), Nx.type(&1))

    actor_target_params =
      Axon.Shared.deep_merge(actor_params, actor_target_params, merge_fn)

    critic1_target_params =
      Axon.Shared.deep_merge(critic1_params, critic1_target_params, merge_fn)

    critic2_target_params =
      Axon.Shared.deep_merge(critic2_params, critic2_target_params, merge_fn)

    %{
      state
      | agent_state: %{
          agent_state
          | actor_target_params: actor_target_params,
            critic1_target_params: critic1_target_params,
            critic2_target_params: critic2_target_params
        }
    }
  end

  defnp sample_experience_replay_buffer(
          random_key,
          %__MODULE__{batch_size: batch_size} = agent_state
        ) do
    data = agent_state.experience_replay_buffer.data
    size = agent_state.experience_replay_buffer.size

    # split and devectorize random_key because we want to keep the replay buffer
    # and its samples devectorized at all times
    split_key = Nx.Random.split(random_key)

    random_key = split_key[0]
    vec_k = split_key[1]

    k = Nx.devectorize(vec_k, keep_names: false)

    k =
      case Nx.shape(k) do
        {2} ->
          k

        {_, 2} ->
          Nx.take(k, 0)
      end

    n = Nx.axis_size(data, 0)

    {batch, _} =
      if size < n do
        probabilities = (Nx.iota({n}) < size) / size
        Nx.Random.choice(k, data, probabilities, samples: batch_size, replace: false, axis: 0)
      else
        Nx.Random.choice(k, data, samples: batch_size, replace: false, axis: 0)
      end

    {stop_grad(batch), random_key}
  end

  defnp action_log_probability(mu, stddev, log_stddev, pre_squash_action, action) do
    # x is assumed to be pre tanh squashing
    type = Nx.type(action)
    eps = Nx.Constants.epsilon(type)

    log_prob(mu, stddev, log_stddev, pre_squash_action) -
      Nx.sum(Nx.log(1 - action ** 2 + eps), axes: [-1], keep_axes: true)
  end

  defnp log_prob(mu, stddev, log_stddev, x) do
    # compute the variance
    type = Nx.type(x)
    eps = Nx.Constants.epsilon(type)

    # formula for the log-probability density function of a Normal distribution
    z = (x - mu) / (stddev + eps)

    log_prob = -0.5 * z ** 2 - log_stddev - Nx.log(Nx.sqrt(2 * pi(type)))

    Nx.sum(log_prob, axes: [-1], keep_axes: true)
  end
end
