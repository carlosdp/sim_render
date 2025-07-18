import marimo

__generated_with = "0.14.12"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import gymnasium as gym
    from sim_render.gym import InteractiveRenderWrapper
    return InteractiveRenderWrapper, gym


@app.cell
def _(InteractiveRenderWrapper, gym):
    env = InteractiveRenderWrapper(gym.make("Ant-v5"))
    env.reset(seed=42)
    env
    return (env,)


@app.cell
def _(env):
    env.save("output.glb")
    return


if __name__ == "__main__":
    app.run()
