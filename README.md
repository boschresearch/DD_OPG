# Trajectory-Based Off-Policy Deep Reinforcement Learning

This is the companion code for the Deep Deterministic Off-Policy Gradient (DD-OPG) method reported in the paper
Trajectory-Based Off-Policy Deep Reinforcement Learning by Andreas Doerr et al., ICML 2019. The paper can
be found here: https://arxiv.org/abs/1905.05710. The code allows the users to experiment with the DD-OPG algorithm. Please cite the
above paper when reporting, reproducing or extending the results.

## Purpose of the project

This software is a research prototype, solely developed for and published as
part of the publication cited above. It will neither be
maintained nor monitored in any way.

## Requirements, how to install and use.

The DD-OPG code depends on tensorflow, garage, gym and baselines.

The required version of garage can be found [here](https://github.com/rlworkgroup/garage/tree/582c3c56ffc3b60f0b371c77aec170a6b7aa7210). By installing the garage framework
the other required dependencies will be installed into a conda environment automatically.

### Prerequesits

A valid path must be provided in `policy_gradients/config.py` to store logs and
tensorboard files of the experiments.
An example for `config.py` can be found in `policy_gradients/config_template.py`.

### DD-OPG Training on Cartpole environment

An example to run DD-OPG on the cartpole environment is provided. However,
the algorithm can be run on other garage/gym environments as well.

To run the sample, execute:
```
python run_ddopg_cartpole.py
```

## License

Deep Deterministic Off-Policy Gradient is open-sourced under the AGPL 3 license. See the
[LICENSE](LICENSE) file for details.
