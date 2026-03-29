from nemo.collections.asr.models.rnnt_bpe_models import EncDecRNNTBPEModel
from nemo.core.classes.mixins import AccessMixin
import torch
from torch import nn

class NemoMTLModel(EncDecRNNTBPEModel):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.mtl_branch_added = False

    def add_mtl_branch(self, mtl_branch: nn.Module, mtl_classes: list, mtl_loss_coeff: float):
        self.mtl_branch = mtl_branch(in_features = 1024, out_features = len(mtl_classes))
        self.mtl_loss_fn = nn.CrossEntropyLoss()
        self.mtl_loss_coeff = mtl_loss_coeff
        self.mtl_classes = mtl_classes

        self.l2i = {e: i for i, e in enumerate(mtl_classes)}
        self.i2l = {i: e for e, i in self.l2i.items()}
        self.mtl_branch_added = True

    def training_step(self, batch, batch_nb):
        if not self.mtl_branch_added:
            raise RuntimeError("Multi-task learning has not been set up. add_mtl_branch() must be called before training.")

        # Reset access registry
        if AccessMixin.is_access_enabled(self.model_guid):
            AccessMixin.reset_registry(self)

        signal, signal_len, transcript, transcript_len, sample_ids = batch

        # ------------------------------------
        # ------------------------------------
        # use idx to load in etiology (must be in `lang` field in manifest)
        # WARNING: Hacky as hell
        mtl_labels = torch.tensor([self.l2i[self._train_dl.dataset.get_manifest_sample(id).lang] for id in sample_ids]).to(self.device)
        mtl_labels = nn.functional.one_hot(mtl_labels, num_classes = len(self.mtl_classes))

        # ------------------------------------
        # ------------------------------------


        # forward() only performs encoder forward
        encoded, encoded_len = self.forward(input_signal=signal, input_signal_length=signal_len)
        del signal

        # ------------------------------------
        # ------------------------------------
        # predict etiology and compute loss
        mtl_preds = self.mtl_branch(encoded)
        mtl_loss = self.mtl_loss_fn(mtl_preds, mtl_labels)

        # ------------------------------------
        # ------------------------------------

        # During training, loss must be computed, so decoder forward is necessary
        decoder, target_length, states = self.decoder(targets=transcript, target_length=transcript_len)

        if hasattr(self, '_trainer') and self._trainer is not None:
            log_every_n_steps = self._trainer.log_every_n_steps
            sample_id = self._trainer.global_step
        else:
            log_every_n_steps = 1
            sample_id = batch_nb

        # If experimental fused Joint-Loss-WER is not used
        if not self.joint.fuse_loss_wer:
            # Compute full joint and loss
            joint = self.joint(encoder_outputs=encoded, decoder_outputs=decoder)
            loss_value = self.loss(
                log_probs=joint, targets=transcript, input_lengths=encoded_len, target_lengths=target_length
            )

            # Add auxiliary losses, if registered
            loss_value = self.add_auxiliary_losses(loss_value)

            # Reset access registry
            if AccessMixin.is_access_enabled(self.model_guid):
                AccessMixin.reset_registry(self)

            tensorboard_logs = {
                'train_loss': loss_value,
                'learning_rate': self._optimizer.param_groups[0]['lr'],
                'global_step': torch.tensor(self.trainer.global_step, dtype=torch.float32),
            }

            if (sample_id + 1) % log_every_n_steps == 0:
                self.wer.update(
                    predictions=encoded,
                    predictions_lengths=encoded_len,
                    targets=transcript,
                    targets_lengths=transcript_len,
                )
                _, scores, words = self.wer.compute()
                self.wer.reset()
                tensorboard_logs.update({'training_batch_wer': scores.float() / words})

        else:
            # If experimental fused Joint-Loss-WER is used
            if (sample_id + 1) % log_every_n_steps == 0:
                compute_wer = True
            else:
                compute_wer = False

            # Fused joint step
            loss_value, wer, _, _ = self.joint(
                encoder_outputs=encoded,
                decoder_outputs=decoder,
                encoder_lengths=encoded_len,
                transcripts=transcript,
                transcript_lengths=transcript_len,
                compute_wer=compute_wer,
            )

            # Add auxiliary losses, if registered
            loss_value = self.add_auxiliary_losses(loss_value)

            # Reset access registry
            if AccessMixin.is_access_enabled(self.model_guid):
                AccessMixin.reset_registry(self)

            tensorboard_logs = {
                'train_loss': loss_value,
                'learning_rate': self._optimizer.param_groups[0]['lr'],
                'global_step': torch.tensor(self.trainer.global_step, dtype=torch.float32),
            }

            if compute_wer:
                tensorboard_logs.update({'training_batch_wer': wer})

        # Log items
        self.log_dict(tensorboard_logs)

        # Preserve batch acoustic model T and language model U parameters if normalizing
        if self._optim_normalize_joint_txu:
            self._optim_normalize_txu = [encoded_len.max(), transcript_len.max()]

        # ------------------------------------
        # ------------------------------------
        # Add MTL loss to losses
        loss_value = self.mtl_loss_coeff * mtl_loss + (1 - self.mtl_loss_coeff) * loss_value

        # ------------------------------------
        # ------------------------------------

        return {'loss': loss_value}