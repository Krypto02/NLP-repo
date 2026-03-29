"""
MultiTaskModel — Shared RoBERTa backbone with binary + multi-label heads.

Used by both the training notebook (MultiClassifier.ipynb) and the
Streamlit demo (app.py) for consistent model loading.
"""

from torch import nn
from transformers import AutoModel, AutoConfig

BERT_MODEL = "cardiffnlp/twitter-roberta-base-hate"


class MultiTaskModel(nn.Module):
    """
    Shared RoBERTa backbone  +  binary classification head  +  multi-label head.
    Single forward pass → (binary_logits, ml_logits).
    """

    def __init__(self, model_name=BERT_MODEL, n_ml_labels=4):
        super().__init__()
        config = AutoConfig.from_pretrained(model_name)
        self.backbone = AutoModel.from_pretrained(model_name, config=config)
        h = config.hidden_size
        dp = getattr(config, "classifier_dropout", None) or config.hidden_dropout_prob
        self.dropout = nn.Dropout(dp)
        self.bin_dense = nn.Linear(h, h)
        self.bin_proj = nn.Linear(h, 2)
        self.ml_dense = nn.Linear(h, h)
        self.ml_proj = nn.Linear(h, n_ml_labels)

    def forward(self, input_ids, attention_mask, **_kwargs):
        cls = self.backbone(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state[
            :, 0
        ]
        # Binary head
        b = self.dropout(cls)
        b = self.bin_dense(b).tanh()
        b = self.dropout(b)
        # Multi-label head
        m = self.dropout(cls)
        m = self.ml_dense(m).tanh()
        m = self.dropout(m)
        return self.bin_proj(b), self.ml_proj(m)
