defmodule Rein.MixProject do
  use Mix.Project

  @source_url "https://github.com/DockYard/rein"
  @version "0.1.0"

  def project do
    [
      app: :rein,
      version: "0.1.0",
      elixir: "~> 1.14",
      elixirc_paths: elixirc_paths(Mix.env()),
      compilers: Mix.compilers(),
      start_permanent: Mix.env() == :prod,
      deps: deps(),
      package: package(),
      docs: docs(),
      description: "Reinforcement Learning built with Nx",
      preferred_cli_env: [
        docs: :docs,
        "hex.publish": :docs
      ]
    ]
  end

  # Configuration for the OTP application.
  #
  # Type `mix help compile.app` for more information.
  def application do
    [extra_applications: [:logger, :runtime_tools]]
  end

  # Specifies which paths to compile per environment.
  defp elixirc_paths(:test), do: ["lib", "test/support"]
  defp elixirc_paths(_), do: ["lib"]

  # Specifies your project dependencies.
  #
  # Type `mix help deps` for examples and options.
  defp deps do
    [
      {:ex_doc, "~> 0.30", only: :docs},
      {:nx, "~> 0.6"},
      {:axon, "~> 0.6"}
      | backend()
    ]
  end

  defp backend do
    case System.get_env("REIN_NX_BACKEND") do
      "torchx" ->
        [{:torchx, "~> 0.6"}]

      "binary" ->
        []

      _ ->
        [{:exla, "~> 0.6"}]
    end
  end

  defp package do
    [
      maintainers: ["Paulo Valente"],
      licenses: ["MIT"],
      links: %{"GitHub" => @source_url}
    ]
  end

  defp docs do
    [
      main: "Rein",
      source_url_pattern: "#{@source_url}/blob/v#{@version}/rein/%{path}#L%{line}",
      extras: [
        "guides/gridworld.livemd"
      ],
      before_closing_body_tag: &before_closing_body_tag/1,
      groups_for_functions: [],
      groups_for_modules: [
        Agents: [
          Rein.Agents.QLearning,
          Rein.Agents.DQN,
          Rein.Agents.DDPG,
          Rein.Agents.SAC
        ],
        Environments: [
          Rein.Environments.Gridworld
        ],
        Utils: [
          Rein.Utils.CircularBuffer,
          Rein.Utils.Noise.OUProcess
        ]
      ]
    ]
  end

  defp before_closing_body_tag(:html) do
    """
    <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/katex@0.13.19/dist/katex.min.css" integrity="sha384-beuqjL2bw+6DBM2eOpr5+Xlw+jiH44vMdVQwKxV28xxpoInPHTVmSvvvoPq9RdSh" crossorigin="anonymous">
    <script defer src="https://cdn.jsdelivr.net/npm/katex@0.13.19/dist/katex.min.js" integrity="sha384-aaNb715UK1HuP4rjZxyzph+dVss/5Nx3mLImBe9b0EW4vMUkc1Guw4VRyQKBC0eG" crossorigin="anonymous"></script>
    <script defer src="https://cdn.jsdelivr.net/npm/katex@0.13.19/dist/contrib/auto-render.min.js" integrity="sha384-+XBljXPPiv+OzfbB3cVmLHf4hdUFHlWNZN5spNQ7rmHTXpd7WvJum6fIACpNNfIR" crossorigin="anonymous"
            onload="renderMathInElement(document.body);"></script>
    <script>
      document.addEventListener("DOMContentLoaded", function() {
        renderMathInElement(document.body, {
          delimiters: [
            { left: "$$", right: "$$", display: true },
            { left: "$", right: "$", display: false },
          ]
        });
      });
    </script>
    """
  end

  defp before_closing_body_tag(_), do: ""
end
