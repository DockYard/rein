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
      groups_for_functions: [],
      groups_for_modules: []
    ]
  end
end
