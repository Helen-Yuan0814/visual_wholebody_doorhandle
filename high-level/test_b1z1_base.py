from isaacgym import gymapi
from isaacgym import gymtorch
from isaacgym.torch_utils import *

import numpy as np
import torch
import os

import torch.nn as nn

from skrl.models.torch import Model, GaussianMixin, DeterministicMixin
from skrl.memories.torch import RandomMemory
from skrl.agents.torch.ppo import PPO, PPO_DEFAULT_CONFIG
from skrl.resources.preprocessors.torch import RunningStandardScaler
from skrl.resources.schedulers.torch import KLAdaptiveRL
from skrl.trainers.torch import SequentialTrainer
from skrl.utils import set_seed

from envs import *

from utils.config import load_cfg, get_params, copy_cfg
import utils.wrapper as wrapper

set_seed(43)

def create_env(cfg, args):
    cfg["env"]["enableDebugVis"] = args.debugvis
    cfg["env"]["cameraMode"] = "full"
    cfg["env"]["smallValueSetZero"] = args.small_value_set_zero
    if args.last_commands:
        cfg["env"]["lastCommands"] = True
    if args.record_video:
        cfg["record_video"] = True
    if args.control_freq is not None:
        cfg["env"]["controlFrequencyLow"] = int(args.control_freq)
    robot_start_pose = (-2.00, 0, 0.55)
    if args.eval:
        robot_start_pose = (-0.85, 0, 0.55)
    _env = eval(args.task)(cfg=cfg, rl_device=args.rl_device, sim_device=args.sim_device, 
                         graphics_device_id=args.graphics_device_id, headless=args.headless, 
                         use_roboinfo=args.roboinfo, observe_gait_commands=args.observe_gait_commands, no_feature=args.no_feature, mask_arm=args.mask_arm, pitch_control=args.pitch_control,
                         rand_control=args.rand_control, arm_delay=args.arm_delay, robot_start_pose=robot_start_pose,
                         rand_cmd_scale=args.rand_cmd_scale, rand_depth_clip=args.rand_depth_clip, stop_pick=args.stop_pick, eval=args.eval)
    wrapped_env = wrapper.IsaacGymPreview3Wrapper(_env)
    return wrapped_env

# define models (stochastic and deterministic models) using mixins
class Policy(GaussianMixin, Model):
    def __init__(self, observation_space, action_space, device, num_features, encode_dim, use_tanh=False, clip_actions=False,
                 clip_log_std=True, min_log_std=-20, max_log_std=2, reduction="sum", deterministic=False):
        Model.__init__(self, observation_space, action_space, device)
        transform_func = torch.distributions.transforms.TanhTransform() if use_tanh else None
        GaussianMixin.__init__(self, clip_actions, clip_log_std, min_log_std, max_log_std, reduction, transform_func=transform_func, deterministic=deterministic)

        self.num_features = num_features
        self.encode_dim = encode_dim
        
        if num_features > 0:
            self.feature_encoder = nn.Sequential(nn.Linear(self.num_features, 512),
                                                  nn.ELU(),
                                                  nn.Linear(512, self.encode_dim),)
        self.net = nn.Sequential(nn.Linear(self.num_observations - self.num_features + self.encode_dim, 512),
                            nn.ELU(),
                            nn.Linear(512, 256),
                            nn.ELU(),
                            nn.Linear(256, 128),
                            nn.ELU(),
                            nn.Linear(128, self.num_actions)
                            )
        self.log_std_parameter = nn.Parameter(torch.zeros(self.num_actions))

    def compute(self, inputs, role):
        if self.num_features > 0:
            features_encode = self.feature_encoder(inputs["states"][..., :self.num_features])
            actions = self.net(torch.cat([inputs["states"][..., self.num_features:], features_encode], dim=-1))
        else:
            actions = self.net(inputs["states"])
        # actions[:, 6] = torch.sigmoid(actions[:, 6])
        return actions, self.log_std_parameter, {}

class Value(DeterministicMixin, Model):
    def __init__(self, observation_space, action_space, device, num_features, encode_dim):
        Model.__init__(self, observation_space, action_space, device)
        DeterministicMixin.__init__(self)
        
        self.num_features = num_features
        self.encode_dim = encode_dim
        
        if num_features > 0:
            self.feature_encoder = nn.Sequential(nn.Linear(self.num_features, 512),
                                                    nn.ELU(),
                                                    nn.Linear(512, self.encode_dim))

        self.net = nn.Sequential(nn.Linear(self.num_observations - self.num_features + self.encode_dim, 512),
                                 nn.ELU(),
                                 nn.Linear(512, 256),
                                 nn.ELU(),
                                 nn.Linear(256, 128),
                                 nn.ELU(),
                                 nn.Linear(128, 1))

    def compute(self, inputs, role):
        if self.num_features > 0:
            feature_encode = self.feature_encoder(inputs["states"][..., :self.num_features])
            return self.net(torch.cat([inputs["states"][..., self.num_features:], feature_encode], dim=-1)), {}
        else:
            return self.net(inputs["states"]), {}

def get_trainer(is_eval=False):
    args = get_params()
    args.eval = is_eval
    args.wandb = args.wandb and (not args.eval) and (not args.debug)
    cfg_file = "b1z1_" + args.task[4:].lower() + ".yaml"
    file_path = "data/cfg/" + cfg_file
    
    if args.resume:
        experiment_dir = os.path.join(args.experiment_dir, args.wandb_name)
        checkpoint_dir = os.path.join(experiment_dir, "checkpoints")
        pt_files = os.listdir(checkpoint_dir)
        pt_files = [file for file in pt_files if file.endswith(".pt") and (not file.startswith("best"))]
        # Find the latest checkpoint
        checkpoint_steps = 0
        if len(pt_files) > 0:
            args.checkpoint = os.path.join(checkpoint_dir, sorted(pt_files, key=lambda x: int(x.split("_")[-1].split(".")[0]))[-1])
            checkpoint_steps = int(args.checkpoint.split("_")[-1].split(".")[0])
        cfg_files = os.listdir(experiment_dir)
        cfg_files = [file for file in cfg_files if file.endswith(".yaml")]
        if len(cfg_files) > 0:
            cfg_file = cfg_files[0]
            file_path = os.path.join(experiment_dir, cfg_file)
        
        print("Find the latest checkpoint: ", args.checkpoint)
    print("Using config file: ", file_path)
        
    cfg = load_cfg(file_path)
    cfg['env']['wandb'] = args.wandb
    cfg['env']["useTanh"] = args.use_tanh
    cfg['env']["near_goal_stop"] = args.near_goal_stop
    cfg['env']["obj_move_prob"] = args.obj_move_prob
    if args.debug:
        cfg['env']['numEnvs'] = 34
        
    if args.eval:
        cfg['env']['numEnvs'] = 34
        cfg["env"]["maxEpisodeLength"] = 1500
        if args.checkpoint:
            checkpoint_steps = int(args.checkpoint.split("_")[-1].split(".")[0])
            cfg["env"]["globalStepCounter"] = checkpoint_steps
    env = create_env(cfg=cfg, args=args)
    
    # Tuple[Dict[str, torch.Tensor], torch.Tensor, torch.Tensor, Dict[str, Any]
    for i in trange(1000, desc="Running"):
        actions = 0. * torch.ones(env.num_envs, env.num_action)
        _ = env.step(actions)
    
if __name__ == "__main__":
    trainer = get_trainer()
    trainer.train()
    
