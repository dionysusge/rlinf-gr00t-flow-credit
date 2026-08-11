# GR00T Policy Decorator 实验说明

这一分支实现的是独立 residual SAC，不复用失败的 residual PPO/FSDP 路径。

## 实际数据流

1. 冻结 GR00T 在 `cuda:0` 上生成 normalized base action chunk，并提取 mean-pooled VLM feature 与 encoded state feature。
2. 独立 residual actor/critic 在 `cuda:1` 上运行。Actor 输入为 `context + base_action[16,7]`，输出为 `unit_residual[16,7]`。
3. 唯一的动作合成公式为：

   ```text
   executed_normalized_action = base_action + 0.1 * unit_residual
   ```

4. GR00T 原有 decoder 将合成后的 normalized action 转为 LIBERO action，LIBERO 一次执行完整的 16-step chunk。
5. Replay 每条记录都是一个真实 chunk transition，没有 trajectory padding、loss mask 或 PPO old/new log-prob。

GR00T feature 在进入网络后做逐样本 `LayerNorm`。Base action 已经处于 GR00T normalized action space，不再做运行均值归一化，避免 replay 中同一条数据因为在线统计量变化而改变语义。

## 运行顺序

在服务器仓库根目录执行：

```bash
git fetch origin
git switch exp/policy-decorator-gr00t
git pull --ff-only origin exp/policy-decorator-gr00t

bash experiments/policy_decorator_gr00t/run.sh zero_smoke
bash experiments/policy_decorator_gr00t/run.sh random_smoke
bash experiments/policy_decorator_gr00t/run.sh train
```

也可以顺序执行全部三段：

```bash
bash experiments/policy_decorator_gr00t/run.sh all
```

脚本默认设置 `CUDA_VISIBLE_DEVICES=2,3`：进程内 `cuda:0` 是物理 GPU 2，加载冻结 GR00T；进程内 `cuda:1` 是物理 GPU 3，放 residual actor、twin-Q 与 optimizer。该入口不初始化 Ray，不连接现存 Ray cluster，也没有 FSDP 和 weight sync。

## 三个阶段的硬门槛

- `zero_smoke`：Set A 固定 100 trials，residual 严格为零。默认要求成功率 100%，并且所有 feature/base/residual/executed/decoded action 都是有限值。
- `random_smoke`：固定 20 trials，添加小随机 residual；默认至少成功 1 次且没有非有限值。
- `train`：只有前两份 `gates/*.json` 都显示 `passed: true` 才启动。若仅做调试，可以显式加 `policy_decorator.skip_gate_check=true`，正式实验不建议跳过。

## 默认训练配置

- 16 个并行 train env
- action chunk：`16 x 7`
- residual scale：`0.1`
- actor/twin-Q：3 层、每层 256 的 ReLU MLP
- actor 输入：GR00T context + base action
- critic 输入：GR00T context + executed normalized action
- replay capacity：20,000 chunk transitions，context 用 fp16 存储，采样后转 fp32
- learning starts：256 transitions
- batch size：256
- 每个 env decision 做 1 次 SAC update
- progressive exploration：前 100 decision 从 0% 线性增加到 100% residual activation
- 总计 150 decisions
- 每 30 decisions 保存 checkpoint，结束额外保存 `final`

## W&B 与本地证据

W&B 继续写入原实验使用的 project `GR00T-Residual-RL`，只通过 `PD-${mode}-GR00T-N1.7-LIBERO-Spatial` 区分三个新 run。主要指标包括：

- `env/success_rate_total`、`env/success_rate_window`
- `env/episode_return_window`、`env/episode_length_window`
- `residual/l2_mean`、`residual/l2_max`、`residual/unit_saturation_fraction`
- `train/critic_loss`、`train/actor_loss`、`train/alpha`
- `train/q1_mean`、`train/q2_mean`、`train/target_q_mean`
- `rollout/base_action_abs_mean`、`rollout/executed_action_abs_mean`
- `replay/size`

服务器原始输出默认在：

```text
/data/Wayne/gzw/rlinf_gr00t_n17/results/policy_decorator_gr00t/
```

其中包含每阶段 config、逐 step JSONL、summary、gate JSON 和 checkpoint。生成便于传回本地的小型分析包：

```bash
python experiments/policy_decorator_gr00t/analyze_metrics.py \
  /data/Wayne/gzw/rlinf_gr00t_n17/results/policy_decorator_gr00t

bash experiments/policy_decorator_gr00t/export_results.sh
```

## 恢复训练

```bash
PD_RESUME=/data/Wayne/gzw/rlinf_gr00t_n17/results/policy_decorator_gr00t/checkpoints/policy_decorator_step_000030.pt \
  bash experiments/policy_decorator_gr00t/run.sh train
```

Checkpoint 同时保存 actor、critic、target critic、alpha、optimizers、replay 和 Python/NumPy/Torch RNG 状态。恢复时 LIBERO simulator 会从新的 episode 重新开始，不尝试序列化 MuJoCo 进程状态；这对 off-policy replay 合法，但不等价于逐 bit 恢复环境轨迹。
