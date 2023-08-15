defmodule Rein.Utils.CircularBufferTest do
  use ExUnit.Case

  alias Rein.Utils.CircularBuffer

  test "persists data and reorders correctly" do
    Nx.default_backend(Nx.BinaryBackend)
    Nx.Defn.default_options(compiler: Nx.Defn.Evaluator)

    buffer = CircularBuffer.new({3, 2}, init_value: 0, type: :s64)

    assert %CircularBuffer{size: size, index: index, data: data} =
             buffer = CircularBuffer.append(buffer, Nx.tensor([0, 1]))

    assert size == Nx.tensor(1)
    assert index == Nx.tensor(1)

    assert data ==
             Nx.tensor([
               [0, 1],
               [0, 0],
               [0, 0]
             ])

    assert %CircularBuffer{size: size, index: index, data: data} =
             buffer = CircularBuffer.append(buffer, Nx.tensor([2, 3]))

    assert size == Nx.tensor(2)
    assert index == Nx.tensor(2)

    assert data ==
             Nx.tensor([
               [0, 1],
               [2, 3],
               [0, 0]
             ])

    assert %CircularBuffer{size: size, index: index, data: data} =
             buffer = CircularBuffer.append(buffer, Nx.tensor([4, 5]))

    assert size == Nx.tensor(3)
    assert index == Nx.tensor(0)

    assert data ==
             Nx.tensor([
               [0, 1],
               [2, 3],
               [4, 5]
             ])

    assert %CircularBuffer{size: size, index: index, data: data} =
             buffer = CircularBuffer.append(buffer, Nx.tensor([6, 7]))

    assert size == Nx.tensor(3)
    assert index == Nx.tensor(1)

    assert data ==
             Nx.tensor([
               [6, 7],
               [2, 3],
               [4, 5]
             ])

    assert CircularBuffer.ordered_data(buffer) ==
             Nx.tensor([
               [2, 3],
               [4, 5],
               [6, 7]
             ])
  end
end
