"""
Optional CRF (Conditional Random Field) layer for NER.

Adding a CRF on top of a transformer can improve NER performance by
modelling label transition constraints (e.g. I-Disease should not follow B-Gene).
"""

import torch
import torch.nn as nn
from transformers import PreTrainedModel
from typing import Optional


class CRF(nn.Module):
    """
    Linear-chain CRF layer.

    Implements the forward algorithm (partition function) for training
    and Viterbi decoding for inference.
    """

    def __init__(self, num_tags: int):
        super().__init__()
        self.num_tags = num_tags
        # Transition scores: transitions[i][j] = score of j -> i
        self.transitions = nn.Parameter(torch.randn(num_tags, num_tags))
        self.start_transitions = nn.Parameter(torch.randn(num_tags))
        self.end_transitions = nn.Parameter(torch.randn(num_tags))
        self._reset_parameters()

    def _reset_parameters(self):
        nn.init.uniform_(self.transitions, -0.1, 0.1)
        nn.init.uniform_(self.start_transitions, -0.1, 0.1)
        nn.init.uniform_(self.end_transitions, -0.1, 0.1)

    def forward(
        self,
        emissions: torch.Tensor,
        labels: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute negative log-likelihood loss.

        Parameters
        ----------
        emissions : (batch, seq_len, num_tags)
        labels : (batch, seq_len) with ignored positions set to 0
        mask : (batch, seq_len) boolean
        """
        gold_score = self._score_sentence(emissions, labels, mask)
        forward_score = self._forward_algorithm(emissions, mask)
        return (forward_score - gold_score).mean()

    def _forward_algorithm(self, emissions: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, num_tags = emissions.shape
        # alpha[b, t] = log-sum-exp of all paths ending at tag t
        alpha = self.start_transitions + emissions[:, 0]  # (batch, tags)

        for i in range(1, seq_len):
            emit_score = emissions[:, i].unsqueeze(1)  # (batch, 1, tags)
            trans_score = self.transitions.unsqueeze(0)  # (1, tags, tags)
            alpha_expand = alpha.unsqueeze(2)  # (batch, tags, 1)
            scores = alpha_expand + trans_score + emit_score  # (batch, tags, tags)
            new_alpha = torch.logsumexp(scores, dim=1)  # (batch, tags)
            # Only update where mask is True
            m = mask[:, i].unsqueeze(1)  # (batch, 1)
            alpha = new_alpha * m + alpha * (1 - m)

        alpha = alpha + self.end_transitions
        return torch.logsumexp(alpha, dim=1)  # (batch,)

    def _score_sentence(
        self, emissions: torch.Tensor, labels: torch.Tensor, mask: torch.Tensor,
    ) -> torch.Tensor:
        batch_size, seq_len, _ = emissions.shape
        labels = labels.long()

        score = self.start_transitions[labels[:, 0]]
        score += emissions[:, 0].gather(1, labels[:, 0].unsqueeze(1)).squeeze(1)

        for i in range(1, seq_len):
            m = mask[:, i]
            trans = self.transitions[labels[:, i], labels[:, i - 1]]
            emit = emissions[:, i].gather(1, labels[:, i].unsqueeze(1)).squeeze(1)
            score += (trans + emit) * m

        last_tag_indices = mask.long().sum(dim=1) - 1
        last_tags = labels.gather(1, last_tag_indices.unsqueeze(1)).squeeze(1)
        score += self.end_transitions[last_tags]

        return score

    def viterbi_decode(
        self, emissions: torch.Tensor, mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Viterbi decoding to find the best tag sequence.

        Returns
        -------
        best_tags : (batch, seq_len) tensor of tag indices
        """
        batch_size, seq_len, num_tags = emissions.shape
        viterbi = self.start_transitions + emissions[:, 0]
        backpointers = []

        for i in range(1, seq_len):
            emit_score = emissions[:, i].unsqueeze(1)
            trans_score = self.transitions.unsqueeze(0)
            scores = viterbi.unsqueeze(2) + trans_score + emit_score
            best_scores, best_tags_point = scores.max(dim=1)
            backpointers.append(best_tags_point)
            m = mask[:, i].unsqueeze(1)
            viterbi = best_scores * m + viterbi * (1 - m)

        viterbi += self.end_transitions
        best_last_tags = viterbi.argmax(dim=1)

        # Backtrack
        best_path = [best_last_tags]
        for bp in reversed(backpointers):
            best_last_tags = bp.gather(1, best_last_tags.unsqueeze(1)).squeeze(1)
            best_path.append(best_last_tags)

        best_path.reverse()
        return torch.stack(best_path, dim=1)


class CRFTokenClassificationModel(nn.Module):
    """
    Wraps a HuggingFace token classification model with a CRF layer.
    """

    def __init__(self, base_model: PreTrainedModel, num_labels: int):
        super().__init__()
        self.base_model = base_model
        self.crf = CRF(num_labels)
        self.num_labels = num_labels

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        **kwargs,
    ):
        outputs = self.base_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=None,  # don't compute CE loss
            **kwargs,
        )
        emissions = outputs.logits  # (batch, seq, num_labels)
        mask = attention_mask.float()

        if labels is not None:
            # Replace -100 with 0 for CRF (masked out by attention_mask anyway)
            crf_labels = labels.clone()
            crf_labels[crf_labels == -100] = 0
            loss = self.crf(emissions, crf_labels, mask)
            outputs.loss = loss

        return outputs

    def decode(self, input_ids, attention_mask, **kwargs):
        outputs = self.base_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=None,
            **kwargs,
        )
        emissions = outputs.logits
        mask = attention_mask.float()
        return self.crf.viterbi_decode(emissions, mask)

    @property
    def config(self):
        return self.base_model.config

    def parameters(self, recurse=True):
        yield from self.base_model.parameters(recurse=recurse)
        yield from self.crf.parameters(recurse=recurse)

    def named_parameters(self, prefix="", recurse=True):
        yield from self.base_model.named_parameters(prefix=prefix, recurse=recurse)
        yield from self.crf.named_parameters(prefix="crf", recurse=recurse)

    def save_pretrained(self, output_dir, **kwargs):
        self.base_model.save_pretrained(output_dir, **kwargs)
        torch.save(self.crf.state_dict(), f"{output_dir}/crf_layer.pt")

    def train(self, mode=True):
        self.base_model.train(mode)
        self.crf.train(mode)
        return self

    def eval(self):
        return self.train(False)

    def to(self, *args, **kwargs):
        self.base_model = self.base_model.to(*args, **kwargs)
        self.crf = self.crf.to(*args, **kwargs)
        return self
