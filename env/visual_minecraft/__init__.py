from gymnasium.envs.registration import register

register(
    id="VisualMinecraft-v0",
    entry_point="env.visual_minecraft.env:GridWorldEnv",
)

register(
    id="FixedLenVisualMinecraft-v0",
    entry_point="env.visual_minecraft.fixed_len_env:GridWorldEnv",
)


register(
    id="DebugVisualMinecraft-v0",
    entry_point="env.visual_minecraft.debug_env:DebugGridWorldEnv",
)