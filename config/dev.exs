import Config

config :logger, level: :debug
config :logger, :console, format: "$time $metadata[$level] $message\n", metadata: :all
