defmodule Rein.Utils.CircularBuffer do
  @moduledoc """
  Circular Buffer utility via Nx Containers.
  """

  import Nx.Defn

  @derive {Nx.Container, containers: [:data, :index, :size], keep: []}
  defstruct [:data, :index, :size]

  @doc """
  Creates a new `#{__MODULE__}` with a given shape.

  ## Options

    * `:init_value` - a number or tensor that will be broadcasted
      to `shape`. Defaults to `0`. If the value given is vectorized,
      the buffer will be vectorized accordingly, but all entries will
      share the same current index and size.

    * `:type` - the type for the tensor if `:init_value`.
      Defaults to `:f32`
  """
  def new(shape, opts \\ [init_value: 0, type: :f32]) do
    opts = Keyword.validate!(opts, init_value: 0, type: :f32)

    init_value =
      opts[:init_value]
      |> Nx.to_tensor()
      |> Nx.as_type(opts[:type])

    %__MODULE__{
      data: Nx.broadcast(init_value, shape),
      size: 0,
      index: 0
    }
  end

  @doc """
  Append an item to the current buffer.

  If the `buffer` data has shape `{a, b, c, ...}`,
  `item` must have shape `{b, c, ...}`
  """
  deftransform append(buffer, item) do
    starts = append_start_indices(buffer)
    n = Nx.axis_size(buffer.data, 0)
    index = Nx.remainder(Nx.add(buffer.index, 1), n)
    size = Nx.min(n, Nx.add(buffer.size, 1))

    data =
      case buffer.data.vectorized_axes do
        [] ->
          Nx.put_slice(buffer.data, starts, Nx.new_axis(item, 0))

        _ ->
          [data, item | starts] = Nx.broadcast_vectors([buffer.data, item | starts])
          axes = data.vectorized_axes

          data =
            Nx.revectorize(data, [], target_shape: Tuple.insert_at(buffer.data.shape, 0, :auto))

          starts = Enum.map(starts, &Nx.revectorize(&1, [], target_shape: {:auto}))
          item = Nx.revectorize(item, [], target_shape: Tuple.insert_at(item.shape, 0, :auto))

          for i <- 0..(Nx.axis_size(data, 0) - 1), reduce: data do
            data ->
              starts = Enum.map(starts, & &1[i])

              item = item[i..i]

              Nx.put_slice(
                data,
                [i | starts],
                Nx.reshape(
                  item,
                  Tuple.duplicate(1, Nx.rank(data) - 1) |> Tuple.append(Nx.size(item))
                )
              )
          end
          |> Nx.vectorize(axes)
      end

    %{
      buffer
      | data: data,
        size: size,
        index: index
    }
  end

  deftransformp append_start_indices(buffer) do
    [buffer.index | List.duplicate(0, tuple_size(buffer.data.shape) - 1)]
  end

  @doc """
  Append multiple items to the buffer.

  Works in a similar fashion to `append/2`, but receives
  a tensor with shape equal to the buffer data except
  for the first axis, which will be the number of items to be appended.
  """
  deftransform append_multiple(buffer, items) do
    starts = append_start_indices(buffer)
    n = Nx.axis_size(buffer.data, 0)

    case buffer.data.vectorized_axes do
      [] ->
        for i <- 0..(Nx.axis_size(items, 0) - 1), reduce: buffer do
          buffer ->
            %{
              buffer
              | index: Nx.remainder(Nx.add(buffer.index, 1), n),
                data: Nx.put_slice(buffer.data, starts, Nx.new_axis(items[i], 0)),
                size: Nx.min(n, Nx.add(buffer.size, 1))
            }
        end

      _ ->
        raise "not implemented for vectorized buffer"
    end
  end

  @doc """
  Returns the data starting at the current index.

  The oldest persisted entry will be the first entry in
  the result, and so on.
  """
  defn ordered_data(buffer) do
    n = elem(buffer.data.shape, 0)
    indices = Nx.remainder(Nx.iota({n}) + buffer.index, n)
    Nx.take(buffer.data, indices)
  end
end
