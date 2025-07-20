import marimo

__generated_with = "0.14.12"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import gymnasium as gym
    from simrender.gym import InteractiveRenderWrapper

    return InteractiveRenderWrapper, gym, mo


@app.cell
def _(gym, mo):
    gym_env = gym.make("Ant-v5", render_mode="rgb_array")
    gym_env.reset(seed=42)

    mo.image(gym_env.render())
    return (gym_env,)


@app.cell
def _(InteractiveRenderWrapper, gym_env):
    env = InteractiveRenderWrapper(gym_env)
    env.reset(seed=42)

    with env.animation():
        env.render()

    env
    return


if __name__ == "__main__":
    app.run()
