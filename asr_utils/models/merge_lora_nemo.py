from pathlib import Path
import torch
import nemo.collections.asr as nemo_asr
from peft import LoraConfig, get_peft_model
from typing import Optional, Union


def merge_lora(
    model_tag: str,
    ckpt_path: Union[Path, str],
    out_nemo: Union[Path, str],
    lora_cfg: Optional[LoraConfig] = None,
    device: Optional[str] = "cpu",
):
    """

    Args:
        model_tag: String of model tag. E.g. 'nvidia/birdname-super_saiyan_rnnt-0.6606b'
        ckpt_path: Path or String pointing to LoRA fine-tuned model checkpoints
        out_nemo: Path or String of where to dump merged model. Please end with .nemo
        lora_cfg: LoRA config used in fine-tuning
        device: Defaults to CPU. Pass in GPU if needed.
    """
    # load up the basemodel first then change it to peft
    base_model = nemo_asr.models.ASRModel.from_pretrained(model_tag)
    base_model.to(device)
    base_model.eval()

    lora_model = get_peft_model(base_model, lora_cfg)

    ckpt = torch.load(ckpt_path, map_location=device)
    if "state_dict" in ckpt:
        state_dict = ckpt["state_dict"]
    else:
        state_dict = ckpt
    missing, unexpected = lora_model.load_state_dict(state_dict, strict=False)

    # what we missin?
    print(f"Missing: {missing[:20]}")
    print(f"Unexpected: {unexpected[:20]}")

    # take a look at the docs for .merge_and_unload() but this is what we need to run LoRA models
    merged_model = lora_model.merge_and_unload()
    merged_model.eval()
    merged_model.save_to(out_nemo)  # this should be of type .nemo!
    print(f"Saved model sent to {out_nemo}")
