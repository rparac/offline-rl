from gymnasium.envs.registration import register

register(
    id="SimpleLTLEnv-v0",
    entry_point="env.simple_ltl_env:SimpleLTLEnv",
)