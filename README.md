# Rein

> :warning: **This library is a work in progress!**

Reinforcement Learning algorithms written in [Nx](https://github.com/elixir-nx/nx/tree/main/nx#readme).

## Installation

If [available in Hex](https://hex.pm/docs/publish), the package can be installed
by adding `rein` to your list of dependencies in `mix.exs`:

```elixir
def deps do
  [
    {:rein, "~> 0.1.0"}
  ]
end
```

### Dependencies

This library has no external dependencies. However,
one should be able to run [EXLA](https://github.com/elixir-nx/nx/tree/main/exla#readme), which is the default backend and compiler.

[Torchx](https://github.com/elixir-nx/nx/tree/main/torchx#readme) can also be used through the `REIN_NX_BACKEND` environment variable.

### Environment variables

- REIN_NX_BACKEND
  If set to "torchx", will use Torchx as the default backend. If "binary", uses plain Nx.BinaryBackend.
  Otherwise, will use EXLA as the default backend and compiler.

  For EXLA and Torchx, each have their own available environment variables as well.

## Authors ##

- [Paulo Valente](https://github.com/polvalente)

[We are very thankful for the many contributors](https://github.com/dockyard/rein/graphs/contributors)

## Versioning ##

This library follows [Semantic Versioning](https://semver.org)

## Looking for help with your Elixir project? ##

[At DockYard we are ready to help you build your next Elixir project](https://dockyard.com/phoenix-consulting). We have a unique expertise
in Elixir and Phoenix development that is unmatched. [Get in touch!](https://dockyard.com/contact/hire-us)

At DockYard we love Elixir! You can [read our Elixir blog posts](https://dockyard.com/blog/categories/elixir)

## Legal ##

[DockYard](https://dockyard.com/), Inc. © 2023

[@DockYard](https://twitter.com/DockYard)

[Licensed under the MIT license](https://www.opensource.org/licenses/mit-license.php)
