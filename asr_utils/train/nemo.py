from pathlib import Path
import nemo.collections.asr as nemo_asr
import lightning as L
from peft import LoraConfig, get_peft_model
from typing import Optional
from lightning.pytorch.callbacks import ModelCheckpoint
from datetime import datetime

# Argument format references:
# model_tag = "nvidia/parakeet-rnnt-0.6b"
# train_cfg = {"manifest_filepath": "manifest.json",
#              "sample_rate": 16000, "batch_size": 8}
# val_cfg = {"manifest_filepath": "manifest.json",
#            "sample_rate": 16000, "batch_size": 8}
# optim_cfg = {}

# trainer_kwargs = {}

# lora_config = LoraConfig(
#     r=16,
#     lora_alpha=16,
#     lora_dropout=0.05,
#     target_modules=[
#         "linear_q",
#         "linear_v",
#         "linear1",
#         "linear2",
#     ]
# )


def fine_tune_nemo_model(
    model_tag: str,
    train_data_cfg: dict,
    val_data_cfg: dict,
    optim_cfg: dict,
    trainer_kwargs: dict,
    lora_cfg: Optional[LoraConfig] = None,
):
    """Function for fine-tuning NeMo ASR models.
    Performs full fine-tuning by default.
    To use LoRA, provide a valid LoraConfig.

    Args:
        model_tag (str): Tag for the model to fine-tune.
        train_data_cfg (dict):  Dict containing training data configuration.
                                Keys should contain at least "manifest_datapath", "sample_rate", and "batch_size".
        val_data_cfg (dict): Same as train_data_cfg, but for validation data.
        optim_cfg (dict): Dict containing optimizer configuration, i.e. learning rate, optimizer, etc.
        trainer_kwargs (dict): Keyword arguments used when instantiating the Lightning trainer.
        lora_cfg (LoraConfig, optional): LoRA configuration. If not None, all other layers will be frozen
                                         and only the adapters will be trained. Defaults to None.
    """
    asr_model = nemo_asr.models.ASRModel.from_pretrained(model_tag)
    asr_model.setup_training_data(train_data_cfg)
    asr_model.setup_validation_data(val_data_cfg)
    asr_model.setup_optimization(optim_cfg)  # TIP: increase LR

    saving_tag = model_tag.replace("/", "_")
    checkpoint_callback = ModelCheckpoint(
        monitor="val_wer",
        mode="min",
        save_top_k=1,
        save_last=True,
        filename=saving_tag + "-{epoch:02d}-{val_wer:.4f}",
    )

    trainer = L.Trainer(callbacks=[checkpoint_callback], **trainer_kwargs)

    # trying both at once as a protective measure before committing to deleting the original
    output = Path(trainer.default_root_dir)
    output2 = output / "nemo_models"
    output.mkdir(parents=True, exist_ok=True)
    output2.mkdir(parents=True, exist_ok=True)
    nemo_file = output / f"{saving_tag}.nemo"
    nemo_file2 = output2 / f"{datetime.today()}-{saving_tag}.nemo"

    if lora_cfg:
        print("Using LoRA")

        lora_model = get_peft_model(asr_model, lora_cfg)

        # kind of hacky, but PeftModel doesn't work with Lightning
        lora_model.base_model.model.set_trainer(trainer)
        trainer.fit(lora_model.base_model.model)
        lora_model.base_model.model.save_to(str(nemo_file))
    else:
        print("Full fine-tuning")

        asr_model.set_trainer(trainer)
        trainer.fit(asr_model)
        asr_model.save_to(str(nemo_file))
        asr_model.save_to(nemo_file2)

    print(f"Saved final .nemo model to {nemo_file}\n")
    print(f"Saved final .nemo model to {nemo_file2} as well...\n")

    # doing the best ckpt too
    best_ckpt = checkpoint_callback.best_model_path
    print(f"Best checkpoint: {best_ckpt}\n")
