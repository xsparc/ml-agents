## Recording Demonstrations

If you want to help your agents learn (especially with environments that have
sparse rewards) using pre-recorded demonstrations, you can generally enable both
GAIL and Behavioral Cloning at low strengths in addition to having an extrinsic
reward. An example of this is provided for the Pyramids example environment
under `PyramidsLearning` in `config/gail_config.yaml`.

If you want to train purely from demonstrations, GAIL and BC _without_ an
extrinsic reward signal is the preferred approach. An example of this is
provided for the Crawler example environment under `CrawlerStaticLearning` in
`config/gail_config.yaml`.

In order to record demonstrations from an agent, add the
`Demonstration Recorder` component to a GameObject in the scene which contains
an `Agent` component. Once added, it is possible to name the demonstration that
will be recorded from the agent.

<p align="center">
  <img src="images/demo_component.png"
       alt="BC Teacher Helper"
       width="375" border="10" />
</p>

When `Record` is checked, a demonstration will be created whenever the scene is
played from the Editor. Depending on the complexity of the task, anywhere from a
few minutes or a few hours of demonstration data may be necessary to be useful
for imitation learning. When you have recorded enough data, end the Editor play
session. A `.demo` file will be created in the `Assets/Demonstrations` folder
(by default). This file contains the demonstrations. Clicking on the file will
provide metadata about the demonstration in the inspector.

<p align="center">
  <img src="images/demo_inspector.png"
       alt="BC Teacher Helper"
       width="375" border="10" />
</p>
