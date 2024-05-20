import flsim.configs  # noqa
import hydra
import torch
from flsim.interfaces.metrics_reporter import Channel
from flsim.utils.config_utils import maybe_parse_json_config
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf

from my_example_utils import CNN_LSTM, MyFLModel, MyMetricsReporter, build_data_provider
from flsim.trainers.async_trainer import AsyncTrainer


def main(
    trainer_config,
    data_config,
    use_cuda_if_available: bool = False,
) -> None:
    cuda_enabled = torch.cuda.is_available() and use_cuda_if_available
    device = torch.device(f"cuda:{0}" if cuda_enabled else "cpu")
    torch.manual_seed(42)
    print(f"Using device: {device}")
    # check if async_weight is in the trainer_config

    model = CNN_LSTM(input_len=12, single_pred=False, d_in=3)
    print(f"Number of parameters: {sum(p.numel() for p in model.parameters())}")
    global_model = MyFLModel(model, device)

    if cuda_enabled:
        global_model.fl_cuda()

    (
        data_provider,
        avg_num_examples,
        std_num_examples,
        min_num_examples,
    ) = build_data_provider(
        local_batch_size=data_config.local_batch_size,
        num_training_clients=data_config.num_training_clients,
        drop_last=False,
    )
    print(
        f"Avg num examples: {avg_num_examples}, std num examples: {std_num_examples}, min num examples: {min_num_examples}"
    )
    if "async_weight" in trainer_config:
        trainer_config.async_weight.example_weight.avg_num_examples = int(
            avg_num_examples
        )

    trainer = instantiate(trainer_config, model=global_model, cuda_enabled=cuda_enabled)

    print(f"Created {trainer_config._target_}")

    metrics_reporter = MyMetricsReporter([Channel.TENSORBOARD, Channel.STDOUT])

    final_model, eval_score = trainer.train(
        data_provider=data_provider,
        metrics_reporter=metrics_reporter,
        num_total_users=data_provider.num_train_users(),
        distributed_world_size=1,
    )

    trainer.test(
        data_provider=data_provider,
        metrics_reporter=MyMetricsReporter([Channel.TENSORBOARD, Channel.STDOUT]),
    )


@hydra.main(
    config_path="/home/taiello/projects/fl-med-devices/conf",
    config_name="replace_bg_config",
    version_base="1.2.0",
)
def run(cfg: DictConfig) -> None:
    print(OmegaConf.to_yaml(cfg))

    trainer_config = cfg.trainer
    data_config = cfg.data

    main(
        trainer_config,
        data_config,
    )


def invoke_main() -> None:
    cfg = maybe_parse_json_config()
    run(cfg)


if __name__ == "__main__":
    invoke_main()  # pragma: no cover
