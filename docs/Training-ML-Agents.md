# Training ML-Agents

For a broad overview of reinforcement learning, imitation learning and all the
training scenarios, methods and options within the ML-Agents Toolkit, see
[ML-Agents Toolkit Overview](ML-Agents-Overview.md).

Once your learning environment has been created and is ready for training, the
next step is to initiate a training run. Training in the ML-Agents Toolkit is
powered by a dedicated Python package, `mlagents`. This package exposes a
command `mlagents-learn` that is the single entry point for all training
workflows (e.g. reinforcement leaning, imitation learning, curriculum learning).
Its implementation can be found at
[ml-agents/mlagents/trainers/learn.py](../ml-agents/mlagents/trainers/learn.py).

## Training with mlagents-learn

### Starting Training

`mlagents-learn` is the main training utility provided by the ML-Agents Toolkit.
It accepts a number of CLI options in addition to a YAML configuration file that
contains all the configurations and hyperparameters to be used during training.
The set of configurations and hyperparameters to include in this file depend on
the agents in your environment and the specific training method you wish to
utilize. Keep in mind that the hyperparameter values can have a big impact on
the training performance (i.e. your agent's ability to learn a policy that
solves the task). In this page, we will review all the hyperparameters for all
training methods and provide guidelines and advice on their values.

To view a description of all the CLI options accepted by `mlagents-learn`, use
the `--help`:

```sh
mlagents-learn --help
```

The basic command for training is:

```sh
mlagents-learn <trainer-config-file> --env=<env_name> --run-id=<run-identifier>
```

where

- `<trainer-config-file>` is the file path of the trainer configuration yaml.
  This contains all the hyperparameter values. We offer a detailed guide on the
  structure of this file and the meaning of the hyperameters (and advice on how
  to set them) in the dedicated [Configuration Files](#configuration-files)
  section below.
- `<env_name>`**(Optional)** is the name (including path) of your
  [Unity executable](Learning-Environment-Executable.md) containing the agents
  to be trained. If `<env_name>` is not passed, the training will happen in the
  Editor. Press the :arrow_forward: button in Unity when the message _"Start
  training by pressing the Play button in the Unity Editor"_ is displayed on
  the screen.
- `<run-identifier>` is a unique name you can use to identify the results of
  your training runs.

See the
[Getting Started Guide](Getting-Started.md#training-a-new-model-with-reinforcement-learning)
for a sample execution of the `mlagents-learn` command.

#### Observing Training

Regardless of which training methods, configurations or hyperparameters you
provide, the training process will always generate three artifacts:

1. Summaries (under the `summaries/` folder): these are training metrics that
   are updated throughout the training process. They are helpful to monitor your
   training performance and may help inform how to update your hyperparameter
   values. See [Using TensorBoard](Using-Tensorboard.md) for more details on how
   to visualize the training metrics.
1. Models (under the `models/` folder): these contain the model checkpoints that
   are updated throughout training and the final model file (`.nn`). This final
   model file is generated once either when training completes or is
   interrupted.
1. Timers file (also under the `summaries/` folder): this contains aggregated
   metrics on your training process, including time spent on specific code
   blocks. See [Profiling in Python](Profiling-Python.md) for more information
   on the timers generated.

These artifacts (except the `.nn` file) are updated throughout the training
process and finalized when training completes or is interrupted.

#### Stopping and Resuming Training

To interrupt training and save the current progress, hit `Ctrl+C` once and wait
for the model(s) to be saved out.

To resume a previously interrupted or completed training run, use the `--resume`
flag and make sure to specify the previously used run ID.

If you would like to re-run a previously interrupted or completed training run
and re-use the same run ID (in this case, overwriting the previously generated
artifacts), then use the `--force` flag.

#### Loading an Existing Model

You can also use this mode to run inference of an already-trained model in
Python by using both the `--resume` and `--inference` flags. Note that if you
want to run inference in Unity, you should use the
[Unity Inference Engine](Getting-Started.md#running-a-pre-trained-model).

Alternatively, you might want to start a new training run but _initialize_ it
using an already-trained model. You may want to do this, for instance, if your
environment changed and you want a new model, but the old behavior is still
better than random. You can do this by specifying
`--initialize-from=<run-identifier>`, where `<run-identifier>` is the old run
ID.

## Configuration Files

The Unity ML-Agents Toolkit provides a wide range of training scenarios, methods
and options. As such, specific training runs may require different training
configurations and may generate different artifacts and TensorBoard statistics.
This section offers a detailed guide into how to manage the different training
set-ups withing the toolkit.

More specifically, this section offers a detailed guide on four command-line
flags for `mlagents-learn` that control the training configurations:

- `<trainer-config-file>`: defines the training hyperparameters for each
  Behavior in the scene
- `--curriculum`: defines the set-up for Curriculum Learning
- `--sampler`: defines the set-up for Environment Parameter Randomization
- `--num-envs`: number of concurrent Unity instances to use during training

Reminder that a detailed description of all command-line options can be found by
using the help utility:

```sh
mlagents-learn --help
```

It is important to highlight that successfully training a Behavior in the
ML-Agents Toolkit involves tuning the training hyperparameters and
configuration. This guide contains some best practices for tuning the training
process when the default parameters don't seem to be giving the level of
performance you would like. We provide sample configuration files for our
example environments in the [config/](../config/) directory. The
`config/trainer_config.yaml` was used to train the 3D Balance Ball in the
[Getting Started](Getting-Started.md) guide. That configuration file uses the
PPO trainer, but we also have configuration files for SAC and GAIL.

Additionally, the set of configurations you provide depend on the training
functionalities you use (see [ML-Agents Toolkit Overview](ML-Agents-Overview.md)
for a description of all the training functionalities). Each functionality you
add typically has its own training configurations or additional configuration
files. For instance:

- Use PPO or SAC?
- Use Recurrent Neural Networks for adding memory to your agents?
- Use the intrinsic curiosity module?
- Ignore the environment reward signal?
- Pre-train using behavioral cloning? (Assuming you have recorded
  demonstrations.)
- Include the GAIL intrinsic reward signals? (Assuming you have recorded
  demonstrations.)
- Use self-play? (Assuming your environment includes multiple agents.)

The answers to the above questions will dictate the configuration files and the
parameters within them. The rest of this section breaks down the different
configuration files and explains the possible settings for each.

### Trainer Config File

We begin with the trainer config file, which includes a set of configurations
for each Behavior in your scene. Some of the configurations are required while
others are optional. To help us get started, below a sample file that includes
all the possible settings if we're using a PPO trainer with all the possible
training functionalities enabled (memory, behavioral cloning, curiosity, GAIL
and self-play). You will notice that curricilumn and environment parameter
randomization settings are not part of this file, but their settings live in
different files that we'll cover in subsequent sections.

```yaml
BehaviorPPO:
  trainer: ppo

  # Trainer configs common to PPO/SAC (excluding reward signals)
  batch_size: 1024
  buffer_size: 10240
  hidden_units: 128
  learning_rate: 3.0e-4
  learning_rate_schedule: linear
  max_steps: 5.0e5
  normalize: false
  num_layers: 2
  time_horizon: 64
  vis_encoder_type: simple

  # PPO-specific configs
  beta: 5.0e-3
  epsilon: 0.2
  lambd: 0.95
  num_epoch: 3

  # memory
  use_recurrent: true
  sequence_length: 64
  memory_size: 256

  # behavior cloning
  behavioral_cloning:
    demo_path: Project/Assets/ML-Agents/Examples/Pyramids/Demos/ExpertPyramid.demo
    strength: 0.5
    steps: 150000
    batch_size: 512
    num_epoch: 3
    samples_per_update: 0
    init_path:

  reward_signals:
    # environment reward
    extrinsic:
      strength: 1.0
      gamma: 0.99

    # curiosity module
    curiosity:
      strength: 0.02
      gamma: 0.99
      encoding_size: 256
      learning_rate: 3e-4

    # GAIL
    gail:
      strength: 0.01
      gamma: 0.99
      encoding_size: 128
      demo_path: Project/Assets/ML-Agents/Examples/Pyramids/Demos/ExpertPyramid.demo
      learning_rate: 3e-4
      use_actions: false
      use_vail: false

  # self-play
  self_play:
    window: 10
    play_against_latest_model_ratio: 0.5
    save_steps: 50000
    swap_steps: 50000
    team_change: 100000
```

Here is an equivalent file if we use an SAC trainer instead. Notice that the
configurations for the additional functionalities (memory, behavioral cloning,
curiosity and self-play) remain unchanged.

```yaml
BehaviorSAC:
  trainer: sac

  # Trainer configs common to PPO/SAC (excluding reward signals)
  # same as PPO config

  # SAC-specific configs (replaces the "PPO-specific configs" section above)
  buffer_init_steps: 0
  tau: 0.005
  num_update: 1
  train_interval: 1
  init_entcoef: 1.0
  save_replay_buffer: false

  # memory
  # same as PPO config

  # pre-training using behavior cloning
  behavioral_cloning:
    # same as PPO config

  reward_signals:
    reward_signal_num_update: 1 # only applies to SAC

    # environment reward
    extrinsic:
      # same as PPO config

    # curiosity module
    curiosity:
      # same as PPO config

    # GAIL
    gail:
      # same as PPO config

  # self-play
  self_play:
    # same as PPO config
```

We now break apart the components of the configuration file and describe what
each of these parameters mean and provide guidelines on how to set them

#### Common Trainer Configurations

One of the first decisions you need to make regarding your training run is which
trainer to use: PPO or SAC. There are some training configurations that are
common to both trainers (which we review now) and others that depend on the
choice of the trainer (which we review on subsequent sections).

| **Setting**            | **Description** |
| :--------------------- | :-------------- |
| trainer                | TBC             |
| init_path              | TBC             |
| summary_freq           | TBC             |
| batch_size             | TBC             |
| buffer_size            | TBC             |
| hidden_units           | TBC             |
| learning_rate          | TBC             |
| learning_rate_schedule | TBC             |
| max_steps              | TBC             |
| normalize              | TBC             |
| num_layers             | TBC             |
| time_horizon           | TBC             |
| vis_encoder_type       | TBC             |

#### Trainer-specific Configurations

Depending on your choice of a trainer, there are additional trainer-specific
configurations. We present them below in two separate tables, but keep in mind
that you only need to include the configurations for the trainer selected (i.e.
the `trainer` setting above).

PPO-specific configurations: | **Setting** | **Description** |
:--------------------- |
:--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
| | beta | TBC | epsilon | TBC | lambd | TBC | num_epoch | TBC

SAC-specific configurations: | **Setting** | **Description** |
:--------------------- |
:--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
| | buffer_init_steps | TBC | init_entcoef | TBC | save_replay_buffer | TBC |
tau | TBC | train_interval | TBC | num_update | TBC

#### Memory-enhanced agents using Recurrent Neural Networks

You can enable your agents to use memory, by setting `use_recurrent` to `true`
and setting `memory_size` and `sequence_length`:

| **Setting**     | **Description** |
| :-------------- | :-------------- |
| use_recurrent   | TBC             |
| memory_size     | TBC             |
| sequence_length | TBC             |

A few considerations when deciding to use memory:

- LSTM does not work well with continuous vector action space. Please use
  discrete vector action space for better results.
- Since the memories must be sent back and forth between Python and Unity, using
  too large `memory_size` will slow down training.
- Adding a recurrent layer increases the complexity of the neural network, it is
  recommended to decrease `num_layers` when using recurrent.
- It is required that `memory_size` be divisible by 4.

#### Behavioral Cloning

To enable Behavioral Cloning as a pre-training option (assuming you have
recorded demonstrations), provide the following configurations under the
`behavior_cloning` section:

| **Setting**        | **Description** |
| :----------------- | :-------------- |
| demo_path          | TBC             |
| strength           | TBC             |
| steps              | TBC             |
| batch_size         | TBC             |
| num_epoch          | TBC             |
| samples_per_update | TBC             |
| init_path          | TBC             |

#### Reward Signals

The `reward_signals` section enables the specification of settings for both
extrinsic (i.e. environment-based) and intrinsic reward signals (e.g. curiosity
and GAIL). Provide the following configurations to design the reward signal for
your training run:

**Extrinsic rewards** - Enable these settings to ensure that your training run
incorporates your environment-based reward signal:

| **Setting**          | **Description** |
| :------------------- | :-------------- |
| extrinsic > strength | TBC             |
| extrinsic > gamma    | TBC             |

**Curiosity intrinsic reward**- To enable curiosity, provide these settings:

| **Setting**               | **Description** |
| :------------------------ | :-------------- |
| curiosity > strength      | TBC             |
| curiosity > gamma         | TBC             |
| curiosity > encoding_size | TBC             |
| curiosity > learning_rate | TBC             |

**GAIL intrinsic reward**- To enable GAIL (assuming you have recorded
demonstrations), provide these settings:

| **Setting**          | **Description** |
| :------------------- | :-------------- |
| gail > strength      | TBC             |
| gail > gamma         | TBC             |
| gail > demo_path     | TBC             |
| gail > encoding_size | TBC             |
| gail > learning_rate | TBC             |
| gail > use_actions   | TBC             |
| gail > use_vail      | TBC             |

#### Self-Play

If your environment contains multiple agents that are divided into teams, you
can leverage our self-play training option by providing these configurations:

| **Setting**                     | **Description** |
| :------------------------------ | :-------------- |
| save_steps                      | TBC             |
| team_change                     | TBC             |
| swap_steps                      | TBC             |
| play_against_latest_model_ratio | TBC             |
| window                          | TBC             |

Training with self-play adds additional confounding factors to the usual issues
faced by reinforcement learning. In general, the tradeoff is between the skill
level and generality of the final policy and the stability of learning. Training
against a set of slowly or unchanging adversaries with low diversity results in
a more stable learning process than training against a set of quickly changing
adversaries with high diversity. With this context, this guide discusses the
exposed self-play hyperparameters and intuitions for tuning them.

**A Note on Reward Signals**

We make the assumption that the final reward in a trajectory corresponds to the
outcome of an episode. A final reward of +1 indicates winning, -1 indicates
losing and 0 indicates a draw. The ELO calculation (discussed below) depends on
this final reward being either +1, 0, -1.

The reward signal should still be used as described in the documentation for the
other trainers and [reward signals.](Reward-Signals.md) However, we encourage
users to be a bit more conservative when shaping reward functions due to the
instability and non-stationarity of learning in adversarial games. Specifically,
we encourage users to begin with the simplest possible reward function (+1
winning, -1 losing) and to allow for more iterations of training to compensate
for the sparsity of reward.

**Note on Swap Steps**

As an example, in a 2v1 scenario, if we want the swap to occur x=4 times during
team-change=200000 steps, the swap_steps for the team of one agent is:

swap_steps = (1 / 2) \* (200000 / 4) = 25000 The swap_steps for the team of two
agents is:

swap_steps = (2 / 1) \* (200000 / 4) = 100000 Note, with equal team sizes, the
first term is equal to 1 and swap_steps can be calculated by just dividing the
total steps by the desired number of swaps.

A larger value of swap_steps means that an agent will play against the same
fixed opponent for a longer number of training iterations. This results in a
more stable training scenario, but leaves the agent open to the risk of
overfitting it's behavior for this particular opponent. Thus, when a new
opponent is swapped, the agent may lose more often than expected.

### Curriculum Learning

```yml
Curriculum:
  measure: progress
  thresholds: [0.1, 0.3, 0.5]
  min_lesson_length: 100
  signal_smoothing: true
  parameters:
    wall_height: [1.5, 2.0, 2.5, 4.0]
```

Each group of Agents under the same `Behavior Name` in an environment can have a
corresponding curriculum. These curricula are held in what we call a
"metacurriculum". A metacurriculum allows different groups of Agents to follow
different curricula within the same environment.

#### Specifying Curricula

In order to define the curricula, the first step is to decide which parameters
of the environment will vary. In the case of the Wall Jump environment, the
height of the wall is what varies. We define this as a `Shared Float Property`
that can be accessed in
`SideChannelUtils.GetSideChannel<FloatPropertiesChannel>()`, and by doing so it
becomes adjustable via the Python API. Rather than adjusting it by hand, we will
create a YAML file which describes the structure of the curricula. Within it, we
can specify which points in the training process our wall height will change,
either based on the percentage of training steps which have taken place, or what
the average reward the agent has received in the recent past is. Below is an
example config for the curricula for the Wall Jump environment.

```yaml
BigWallJump:
  measure: progress
  thresholds: [0.1, 0.3, 0.5]
  min_lesson_length: 100
  signal_smoothing: true
  parameters:
    big_wall_min_height: [0.0, 4.0, 6.0, 8.0]
    big_wall_max_height: [4.0, 7.0, 8.0, 8.0]
SmallWallJump:
  measure: progress
  thresholds: [0.1, 0.3, 0.5]
  min_lesson_length: 100
  signal_smoothing: true
  parameters:
    small_wall_height: [1.5, 2.0, 2.5, 4.0]
```

Once our curriculum is defined, we have to use the environment parameters we
defined and modify the environment from the Agent's `OnEpisodeBegin()` function.
See
[WallJumpAgent.cs](https://github.com/Unity-Technologies/ml-agents/blob/master/Project/Assets/ML-Agents/Examples/WallJump/Scripts/WallJumpAgent.cs)
for an example.

#### Training with a Curriculum

Once we have specified our metacurriculum and curricula, we can launch
`mlagents-learn` using the `–curriculum` flag to point to the config file for
our curricula and PPO will train using Curriculum Learning. For example, to
train agents in the Wall Jump environment with curriculum learning, we can run:

```sh
mlagents-learn config/trainer_config.yaml --curriculum=config/curricula/wall_jump.yaml --run-id=wall-jump-curriculum
```

We can then keep track of the current lessons and progresses via TensorBoard.

### Environment Parameter Randomization

```yml
resampling-interval: 5000

mass:
  sampler-type: "uniform"
  min_value: 0.5
  max_value: 10

gravity:
  sampler-type: "multirange_uniform"
  intervals: [[7, 10], [15, 20]]

scale:
  sampler-type: "uniform"
  min_value: 0.75
  max_value: 3
```

To enable variations in the environments, we implemented
`Environment Parameters`. `Environment Parameters` are values in the
`FloatPropertiesChannel` that can be read when setting up the environment. We
also included different sampling methods and the ability to create new kinds of
sampling methods for each `Environment Parameter`. In the 3D ball environment
example displayed in the figure above, the environment parameters are `gravity`,
`ball_mass` and `ball_scale`.

#### How to Enable Environment Parameter Randomization

We first need to provide a way to modify the environment by supplying a set of
`Environment Parameters` and vary them over time. This provision can be done
either deterministically or randomly.

This is done by assigning each `Environment Parameter` a `sampler-type`(such as
a uniform sampler), which determines how to sample an `Environment Parameter`.
If a `sampler-type` isn't provided for a `Environment Parameter`, the parameter
maintains the default value throughout the training procedure, remaining
unchanged. The samplers for all the `Environment Parameters` are handled by a
**Sampler Manager**, which also handles the generation of new values for the
environment parameters when needed.

To setup the Sampler Manager, we create a YAML file that specifies how we wish
to generate new samples for each `Environment Parameters`. In this file, we
specify the samplers and the `resampling-interval` (the number of simulation
steps after which environment parameters are resampled). Below is an example of
a sampler file for the 3D ball environment.

Below is the explanation of the fields in the above example.

- `resampling-interval` - Specifies the number of steps for the agent to train
  under a particular environment configuration before resetting the environment
  with a new sample of `Environment Parameters`.

- `Environment Parameter` - Name of the `Environment Parameter` like `mass`,
  `gravity` and `scale`. This should match the name specified in the
  `FloatPropertiesChannel` of the environment being trained. If a parameter
  specified in the file doesn't exist in the environment, then this parameter
  will be ignored. Within each `Environment Parameter`

      * `sampler-type` - Specify the sampler type to use for the `Environment Parameter`.
      This is a string that should exist in the `Sampler Factory` (explained
      below).

      * `sampler-type-sub-arguments` - Specify the sub-arguments depending on the `sampler-type`.
      In the example above, this would correspond to the `intervals`
      under the `sampler-type` `"multirange_uniform"` for the `Environment Parameter` called `gravity`.
      The key name should match the name of the corresponding argument in the sampler definition.
      (See below)

The Sampler Manager allocates a sampler type for each `Environment Parameter` by
using the _Sampler Factory_, which maintains a dictionary mapping of string keys
to sampler objects. The available sampler types to be used for each
`Environment Parameter` is available in the Sampler Factory.

#### Included Sampler Types

Below is a list of included `sampler-type` as part of the toolkit.

- `uniform` - Uniform sampler

  - Uniformly samples a single float value between defined endpoints. The
    sub-arguments for this sampler to specify the interval endpoints are as
    below. The sampling is done in the range of [`min_value`, `max_value`).

  - **sub-arguments** - `min_value`, `max_value`

- `gaussian` - Gaussian sampler

  - Samples a single float value from the distribution characterized by the mean
    and standard deviation. The sub-arguments to specify the gaussian
    distribution to use are as below.

  - **sub-arguments** - `mean`, `st_dev`

- `multirange_uniform` - Multirange uniform sampler

  - Uniformly samples a single float value between the specified intervals.
    Samples by first performing a weight pick of an interval from the list of
    intervals (weighted based on interval width) and samples uniformly from the
    selected interval (half-closed interval, same as the uniform sampler). This
    sampler can take an arbitrary number of intervals in a list in the following
    format: [[`interval_1_min`, `interval_1_max`], [`interval_2_min`,
    `interval_2_max`], ...]

  - **sub-arguments** - `intervals`

The implementation of the samplers can be found at
`ml-agents-envs/mlagents_envs/sampler_class.py`.

#### Defining a New Sampler Type

If you want to define your own sampler type, you must first inherit the
_Sampler_ base class (included in the `sampler_class` file) and preserve the
interface. Once the class for the required method is specified, it must be
registered in the Sampler Factory.

This can be done by subscribing to the _register_sampler_ method of the
SamplerFactory. The command is as follows:

`SamplerFactory.register_sampler(*custom_sampler_string_key*, *custom_sampler_object*)`

Once the Sampler Factory reflects the new register, the new sampler type can be
used for sample any `Environment Parameter`. For example, lets say a new sampler
type was implemented as below and we register the `CustomSampler` class with the
string `custom-sampler` in the Sampler Factory.

```python
class CustomSampler(Sampler):

    def __init__(self, argA, argB, argC):
        self.possible_vals = [argA, argB, argC]

    def sample_all(self):
        return np.random.choice(self.possible_vals)
```

Now we need to specify the new sampler type in the sampler YAML file. For
example, we use this new sampler type for the `Environment Parameter` _mass_.

```yaml
mass:
  sampler-type: "custom-sampler"
  argB: 1
  argA: 2
  argC: 3
```

#### Training with Environment Parameter Randomization

After the sampler YAML file is defined, we proceed by launching `mlagents-learn`
and specify our configured sampler file with the `--sampler` flag. For example,
if we wanted to train the 3D ball agent with parameter randomization using
`Environment Parameters` with `config/3dball_randomize.yaml` sampling setup, we
would run

```sh
mlagents-learn config/trainer_config.yaml --sampler=config/3dball_randomize.yaml
--run-id=3D-Ball-randomize
```

We can observe progress and metrics via Tensorboard.

### Training Using Concurrent Unity Instances

Please refer to the general instructions on
[Training ML-Agents](Training-ML-Agents.md). In order to run concurrent Unity
instances during training, set the number of environment instances using the
command line option `--num-envs=<n>` when you invoke `mlagents-learn`.
Optionally, you can also set the `--base-port`, which is the starting port used
for the concurrent Unity instances.

Some considerations:

- **Buffer Size** - If you are having trouble getting an agent to train, even
  with multiple concurrent Unity instances, you could increase `buffer_size` in
  the `config/trainer_config.yaml` file. A common practice is to multiply
  `buffer_size` by `num-envs`.
- **Resource Constraints** - Invoking concurrent Unity instances is constrained
  by the resources on the machine. Please use discretion when setting
  `--num-envs=<n>`.
- **Result Variation Using Concurrent Unity Instances** - If you keep all the
  hyperparameters the same, but change `--num-envs=<n>`, the results and model
  would likely change.
