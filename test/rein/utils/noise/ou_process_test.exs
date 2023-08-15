defmodule Rein.Utils.Noise.OUProcessTest do
  use ExUnit.Case, async: true

  alias Rein.Utils.Noise.OUProcess

  test "generates samples with given shape" do
    Nx.Defn.default_options(compiler: Nx.Defn.Evaluator)
    Nx.default_backend(Nx.BinaryBackend)

    key = Nx.Random.key(1)

    state = OUProcess.init({2})
    range = 1..10

    {values, _key} =
      Enum.map_reduce(range, {key, state}, fn _, {prev_key, state} ->
        {state, key} = OUProcess.sample(prev_key, state)

        assert key.data.__struct__ == Nx.BinaryBackend
        refute key == prev_key

        {state.x, {key, state}}
      end)

    assert values == [
             Nx.tensor([-0.161521315574646, -0.04836982488632202]),
             Nx.tensor([-0.022248566150665283, -0.040264029055833817]),
             Nx.tensor([-0.09898112714290619, 0.007571600377559662]),
             Nx.tensor([0.2752320170402527, 0.27117177844047546]),
             Nx.tensor([0.19806107878684998, 0.3740113377571106]),
             Nx.tensor([0.3326162099838257, 0.45093610882759094]),
             Nx.tensor([0.5560828447341919, 0.3771272897720337]),
             Nx.tensor([0.41871464252471924, 0.24803756177425385]),
             Nx.tensor([0.04342368245124817, 0.10074643790721893]),
             Nx.tensor([-0.3225524425506592, 0.020469389855861664])
           ]
  end
end
