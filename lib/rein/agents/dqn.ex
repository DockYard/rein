defmodule Rein.Agents.DQN do
  @moduledoc """
  Deep Q-Learning implementation.

  This implementation utilizes a single target network for
  the policy network.
  """
  import Nx.Defn

  @behaviour Rein.Agent

  @learning_rate 1.0e-3
  @adamw_decay 1.0e-2
  @eps 1.0e-7
  @experience_replay_buffer_num_entries 10_000

  @eps_start 1
  @eps_decay_rate 0.995
  @eps_increase_rate 1.005
  @eps_end 0.01

  @train_every_steps 32
  @adamw_decay 0.01

  @batch_size 128

  @gamma 0.99
  @tau 0.001

  @derive {Nx.Container,
           containers: [
             :q_policy,
             :q_target,
             :q_policy_optimizer_state,
             :loss,
             :loss_denominator,
             :experience_replay_buffer,
             :experience_replay_buffer_index,
             :persisted_experience_replay_buffer_entries,
             :total_reward,
             :epsilon_greedy_eps,
             :exploration_decay_rate,
             :exploration_increase_rate,
             :min_eps,
             :max_eps,
             :performance_memory,
             :performance_threshold
           ],
           keep: [
             :optimizer_update_fn,
             :policy_predict_fn,
             :input_template,
             :state_vector_size,
             :num_actions,
             :environment_to_input_fn,
             :environment_to_state_vector_fn,
             :state_vector_to_input_fn,
             :learning_rate,
             :batch_size,
             :training_frequency,
             :target_training_frequency,
             :gamma
           ]}

  defstruct [
    :state_vector_size,
    :num_actions,
    :q_policy,
    :q_target,
    :q_policy_optimizer_state,
    :policy_predict_fn,
    :optimizer_update_fn,
    :experience_replay_buffer,
    :experience_replay_buffer_index,
    :persisted_experience_replay_buffer_entries,
    :loss,
    :loss_denominator,
    :total_reward,
    :environment_to_input_fn,
    :environment_to_state_vector_fn,
    :state_vector_to_input_fn,
    :input_template,
    :learning_rate,
    :batch_size,
    :training_frequency,
    :target_training_frequency,
    :gamma,
    :epsilon_greedy_eps,
    :exploration_decay_rate,
    :exploration_increase_rate,
    :min_eps,
    :max_eps,
    :performance_memory,
    :performance_threshold
  ]

  @impl true
  def init(random_key, opts \\ []) do
    opts =
      Keyword.validate!(opts, [
        :q_policy,
        :q_target,
        :policy_net,
        :experience_replay_buffer,
        :experience_replay_buffer_index,
        :persisted_experience_replay_buffer_entries,
        :environment_to_input_fn,
        :environment_to_state_vector_fn,
        :state_vector_to_input_fn,
        :performance_memory,
        target_training_frequency: @train_every_steps * 4,
        learning_rate: @learning_rate,
        batch_size: @batch_size,
        training_frequency: @train_every_steps,
        gamma: @gamma,
        eps_decay_rate: @eps_decay_rate,
        exploration_decay_rate: @eps_decay_rate,
        exploration_increase_rate: @eps_increase_rate,
        min_eps: @eps_end,
        max_eps: @eps_start,
        performance_memory_length: 500,
        performance_threshold: 0.01
      ])

    policy_net = opts[:policy_net] || raise ArgumentError, "missing :policy_net option"

    environment_to_input_fn =
      opts[:environment_to_input_fn] ||
        raise ArgumentError, "missing :environment_to_input_fn option"

    environment_to_state_vector_fn =
      opts[:environment_to_state_vector_fn] ||
        raise ArgumentError, "missing :environment_to_state_vector_fn option"

    state_vector_to_input_fn =
      opts[:state_vector_to_input_fn] ||
        raise ArgumentError, "missing :state_vector_to_input_fn option"

    {policy_init_fn, policy_predict_fn} = Axon.build(policy_net, seed: 0)

    # TO-DO: receive optimizer as argument
    {optimizer_init_fn, optimizer_update_fn} =
      Polaris.Updates.clip_by_global_norm()
      |> Polaris.Updates.compose(
        Polaris.Optimizers.adamw(learning_rate: @learning_rate, eps: @eps, decay: @adamw_decay)
      )

    initial_q_policy_state = opts[:q_policy] || raise "missing initial q_policy"
    initial_q_target_state = opts[:q_target] || initial_q_policy_state

    input_template = input_template(policy_net)

    q_policy = policy_init_fn.(input_template, initial_q_policy_state)
    q_target = policy_init_fn.(input_template, initial_q_target_state)

    q_policy_optimizer_state = optimizer_init_fn.(q_policy)

    {1, num_actions} = Axon.get_output_shape(policy_net, input_template)

    state_vector_size = state_vector_size(input_template)

    loss = loss_denominator = total_reward = Nx.tensor(0, type: :f32)

    state = %__MODULE__{
      learning_rate: opts[:learning_rate],
      total_reward: total_reward,
      batch_size: opts[:batch_size],
      training_frequency: opts[:training_frequency],
      target_training_frequency: opts[:target_training_frequency],
      gamma: opts[:gamma],
      loss: loss,
      loss_denominator: loss_denominator,
      state_vector_size: state_vector_size,
      num_actions: num_actions,
      input_template: input_template,
      environment_to_input_fn: environment_to_input_fn,
      environment_to_state_vector_fn: environment_to_state_vector_fn,
      state_vector_to_input_fn: state_vector_to_input_fn,
      q_policy: q_policy,
      q_policy_optimizer_state: q_policy_optimizer_state,
      q_target: q_target,
      policy_predict_fn: policy_predict_fn,
      optimizer_update_fn: optimizer_update_fn,
      # prev_state_vector, target_x, target_y, action, reward, is_terminal, next_state_vector
      experience_replay_buffer:
        opts[:experience_replay_buffer] ||
          Nx.broadcast(
            Nx.tensor(:nan, type: :f32),
            {@experience_replay_buffer_num_entries, 2 * state_vector_size + 4}
          ),
      experience_replay_buffer_index:
        opts[:experience_replay_buffer_index] || Nx.tensor(0, type: :s64),
      persisted_experience_replay_buffer_entries:
        opts[:persisted_experience_replay_buffer_entries] || Nx.tensor(0, type: :s64),
      performance_threshold: opts[:performance_threshold],
      performance_memory:
        opts[:performance_memory] ||
          Nx.broadcast(total_reward, {opts[:performance_memory_length]}),
      max_eps: opts[:max_eps],
      min_eps: opts[:min_eps],
      epsilon_greedy_eps: opts[:max_eps],
      exploration_decay_rate: opts[:exploration_decay_rate],
      exploration_increase_rate: opts[:exploration_increase_rate]
    }

    {state, random_key}
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

  defp state_vector_size(input_template) do
    Enum.reduce(input_template, 0, fn {_field, tensor}, acc ->
      div(Nx.size(tensor), Nx.axis_size(tensor, 0)) + acc
    end)
  end

  @impl true
  def reset(random_key, %Rein{agent_state: state, episode: episode}) do
    total_reward = loss = loss_denominator = Nx.tensor(0, type: :f32)

    state = adapt_exploration(episode, state)

    {%{
       state
       | total_reward: total_reward,
         loss: loss,
         loss_denominator: loss_denominator
     }, random_key}
  end

  defnp adapt_exploration(
          episode,
          %__MODULE__{
            exploration_decay_rate: exploration_decay_rate,
            exploration_increase_rate: exploration_increase_rate,
            epsilon_greedy_eps: eps,
            min_eps: min_eps,
            max_eps: max_eps,
            total_reward: reward,
            performance_memory: %Nx.Tensor{shape: {n}} = performance_memory,
            performance_threshold: performance_threshold
          } = state
        ) do
    {eps, performance_memory} =
      cond do
        episode == 0 ->
          {eps, performance_memory}

        episode < n ->
          index = Nx.remainder(episode, n)

          performance_memory =
            Nx.indexed_put(
              performance_memory,
              Nx.reshape(index, {1, 1}),
              Nx.reshape(reward, {1})
            )

          {eps, performance_memory}

        true ->
          index = Nx.remainder(episode, n)

          performance_memory =
            Nx.indexed_put(performance_memory, Nx.reshape(index, {1, 1}), Nx.reshape(reward, {1}))

          index = Nx.remainder(index + 1, n)

          # We want to get our 2 windows in sequence so that we can compare them.
          # The rem(iota + index + 1, n) operation will effectively set it so that
          # we have the oldest window starting at the first position, and then all elements
          # in the circular buffer fall into sequence
          window_indices = Nx.remainder(Nx.iota({n}) + index, n)

          # After we take and reshape, the first row contains the oldest `n//2` samples
          # and the second row, the remaining newest samples.
          windows =
            performance_memory
            |> Nx.take(window_indices)
            |> Nx.reshape({2, :auto})

          # avg[0]: avg of the previous performance window
          # avg[1]: avg of the current performance window
          avg = Nx.mean(windows, axes: [1])

          abs_diff = Nx.abs(avg[0] - avg[1])

          eps =
            if abs_diff < performance_threshold do
              # If decayed to less than an "eps" value,
              # we force it to increase from that "eps" instead.
              Nx.min(eps * exploration_increase_rate, max_eps)
            else
              # can decay to 0
              Nx.max(eps * exploration_decay_rate, min_eps)
            end

          {eps, performance_memory}
      end

    %__MODULE__{
      state
      | epsilon_greedy_eps: eps,
        performance_memory: performance_memory
    }
  end

  @impl true
  defn select_action(
         %Rein{random_key: random_key, agent_state: agent_state} = state,
         _iteration
       ) do
    %{
      q_policy: q_policy,
      policy_predict_fn: policy_predict_fn,
      environment_to_input_fn: environment_to_input_fn,
      num_actions: num_actions,
      epsilon_greedy_eps: eps_threshold
    } = agent_state

    {sample, random_key} = Nx.Random.uniform(random_key)

    {action, random_key} =
      if sample > eps_threshold do
        action =
          q_policy
          |> policy_predict_fn.(environment_to_input_fn.(state.environment_state))
          |> Nx.argmax()

        {action, random_key}
      else
        Nx.Random.randint(random_key, 0, num_actions, type: :s64)
      end

    {action, %{state | random_key: random_key}}
  end

  @impl true
  defn record_observation(
         %{
           environment_state: env_state,
           agent_state: %{
             q_policy: q_policy,
             policy_predict_fn: policy_predict_fn,
             state_vector_to_input_fn: state_vector_to_input_fn,
             environment_to_state_vector_fn: as_state_vector_fn,
             gamma: gamma
           }
         },
         action,
         reward,
         is_terminal,
         %{environment_state: next_env_state} = state
       ) do
    state_vector = as_state_vector_fn.(env_state)
    next_state_vector = as_state_vector_fn.(next_env_state)

    idx = Nx.stack([state.agent_state.experience_replay_buffer_index, 0]) |> Nx.new_axis(0)

    shape = {Nx.size(state_vector) + 4 + Nx.size(next_state_vector), 1}

    index_template = Nx.concatenate([Nx.broadcast(0, shape), Nx.iota(shape, axis: 0)], axis: 1)

    predicted_reward =
      reward +
        policy_predict_fn.(q_policy, state_vector_to_input_fn.(next_state_vector)) * gamma *
          (1 - is_terminal)

    %{shape: {1}} = predicted_reward = Nx.reduce_max(predicted_reward, axes: [-1])

    temporal_difference = Nx.reshape(Nx.abs(reward - predicted_reward), {1})

    updates =
      Nx.concatenate([
        Nx.flatten(state_vector),
        Nx.stack([action, reward, is_terminal]),
        Nx.flatten(next_state_vector),
        temporal_difference
      ])

    experience_replay_buffer =
      Nx.indexed_put(state.agent_state.experience_replay_buffer, idx + index_template, updates)

    experience_replay_buffer_index =
      Nx.remainder(
        state.agent_state.experience_replay_buffer_index + 1,
        @experience_replay_buffer_num_entries
      )

    entries = state.agent_state.persisted_experience_replay_buffer_entries

    persisted_experience_replay_buffer_entries =
      Nx.select(
        entries < @experience_replay_buffer_num_entries,
        entries + 1,
        entries
      )

    %{
      state
      | agent_state: %{
          state.agent_state
          | experience_replay_buffer: experience_replay_buffer,
            experience_replay_buffer_index: experience_replay_buffer_index,
            persisted_experience_replay_buffer_entries:
              persisted_experience_replay_buffer_entries,
            total_reward: state.agent_state.total_reward + reward
        }
    }
  end

  @impl true
  defn optimize_model(state) do
    %{
      persisted_experience_replay_buffer_entries: persisted_experience_replay_buffer_entries,
      experience_replay_buffer_index: experience_replay_buffer_index,
      batch_size: batch_size,
      training_frequency: training_frequency,
      target_training_frequency: target_training_frequency
    } = state.agent_state

    has_at_least_one_batch = persisted_experience_replay_buffer_entries > batch_size
    should_update_policy_net = rem(experience_replay_buffer_index, training_frequency) == 0
    should_update_target_net = rem(experience_replay_buffer_index, target_training_frequency) == 0

    {state, _, _, _} =
      while {state, i = 0, training_frequency,
             pred = has_at_least_one_batch and should_update_policy_net},
            pred and i < training_frequency do
        {train(state), i + 1, training_frequency, pred}
      end

    {state, _, _, _} =
      while {state, i = 0, target_training_frequency,
             pred = has_at_least_one_batch and should_update_target_net},
            pred and i < target_training_frequency do
        {soft_update_targets(state), i + 1, target_training_frequency, pred}
      end

    state
  end

  defnp train(state) do
    %{
      agent_state: %{
        q_policy: q_policy,
        q_target: q_target,
        q_policy_optimizer_state: q_policy_optimizer_state,
        policy_predict_fn: policy_predict_fn,
        optimizer_update_fn: optimizer_update_fn,
        state_vector_to_input_fn: state_vector_to_input_fn,
        state_vector_size: state_vector_size,
        experience_replay_buffer: experience_replay_buffer,
        gamma: gamma
      },
      random_key: random_key
    } = state

    {batch, batch_idx, random_key} =
      sample_experience_replay_buffer(random_key, state.agent_state)

    state_batch =
      batch
      |> Nx.slice_along_axis(0, state_vector_size, axis: 1)
      |> then(state_vector_to_input_fn)

    action_batch = Nx.slice_along_axis(batch, state_vector_size, 1, axis: 1)
    reward_batch = Nx.slice_along_axis(batch, state_vector_size + 1, 1, axis: 1)
    is_terminal_batch = Nx.slice_along_axis(batch, state_vector_size + 2, 1, axis: 1)

    next_state_batch =
      batch
      |> Nx.slice_along_axis(state_vector_size + 3, state_vector_size, axis: 1)
      |> then(state_vector_to_input_fn)

    non_final_mask = not is_terminal_batch

    {{experience_replay_buffer, loss}, gradient} =
      value_and_grad(
        q_policy,
        fn q_policy ->
          action_idx = Nx.as_type(action_batch, :s64)

          %{shape: {m, 1}} =
            state_action_values =
            q_policy
            |> policy_predict_fn.(state_batch)
            |> Nx.take_along_axis(action_idx, axis: 1)

          expected_state_action_values =
            reward_batch +
              policy_predict_fn.(q_target, next_state_batch) * gamma * non_final_mask

          %{shape: {n, 1}} =
            expected_state_action_values =
            Nx.reduce_max(expected_state_action_values, axes: [-1], keep_axes: true)

          case {m, n} do
            {m, n} when m != n ->
              raise "shape mismatch for batch values"

            _ ->
              1
          end

          td_errors = Nx.abs(expected_state_action_values - state_action_values)

          {
            update_priorities(
              experience_replay_buffer,
              batch_idx,
              state_vector_size * 2 + 3,
              td_errors
            ),
            Axon.Losses.huber(expected_state_action_values, state_action_values, reduction: :mean)
          }
        end,
        &elem(&1, 1)
      )

    {scaled_updates, optimizer_state} =
      optimizer_update_fn.(gradient, q_policy_optimizer_state, q_policy)

    q_policy = Polaris.Updates.apply_updates(q_policy, scaled_updates)

    %{
      state
      | agent_state: %{
          state.agent_state
          | q_policy: q_policy,
            q_policy_optimizer_state: optimizer_state,
            loss: state.agent_state.loss + loss,
            loss_denominator: state.agent_state.loss_denominator + 1,
            experience_replay_buffer: experience_replay_buffer
        },
        random_key: random_key
    }
  end

  defnp soft_update_targets(state) do
    %{agent_state: %{q_target: q_target, q_policy: q_policy} = agent_state} = state

    q_target = Axon.Shared.deep_merge(q_policy, q_target, &(&1 * @tau + &2 * (1 - @tau)))

    %{state | agent_state: %{agent_state | q_target: q_target}}
  end

  @alpha 0.6
  defnp sample_experience_replay_buffer(
          random_key,
          %{state_vector_size: state_vector_size} = agent_state
        ) do
    %{shape: {@experience_replay_buffer_num_entries, _}} =
      exp_replay_buffer = slice_experience_replay_buffer(agent_state)

    # Temporal Difference prioritizing:
    # We are going to sort experiences by temporal difference
    # and divide our buffer into 4 slices, from which we will
    # then uniformily sample.
    # The temporal difference is already stored in the end of our buffer.

    temporal_difference =
      exp_replay_buffer
      |> Nx.slice_along_axis(state_vector_size * 2 + 3, 1, axis: 1)
      |> Nx.flatten()

    priorities = temporal_difference ** @alpha
    probs = priorities / Nx.sum(priorities)

    {batch_idx, random_key} =
      Nx.Random.choice(random_key, Nx.iota(temporal_difference.shape), probs,
        samples: @batch_size,
        replace: false,
        axis: 0
      )

    batch = Nx.take(exp_replay_buffer, batch_idx)
    {batch, batch_idx, random_key}
  end

  defnp slice_experience_replay_buffer(state) do
    %{
      experience_replay_buffer: experience_replay_buffer,
      persisted_experience_replay_buffer_entries: entries
    } = state

    if entries < @experience_replay_buffer_num_entries do
      t = Nx.iota({@experience_replay_buffer_num_entries})
      idx = Nx.select(t < entries, t, 0)
      Nx.take(experience_replay_buffer, idx)
    else
      experience_replay_buffer
    end
  end

  defn update_priorities(
         buffer,
         %{shape: {n}} = row_idx,
         target_column,
         td_errors
       ) do
    case td_errors.shape do
      {^n, 1} -> :ok
      shape -> raise "invalid shape for td_errors, got: #{inspect(shape)}"
    end

    indices = Nx.stack([row_idx, Nx.broadcast(target_column, {n})], axis: -1)

    Nx.indexed_put(buffer, indices, Nx.reshape(td_errors, {n}))
  end
end
