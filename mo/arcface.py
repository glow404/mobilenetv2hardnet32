import torch
import torch.nn as nn
import torch.nn.functional as F


class ArcFace(nn.Module):

    def __init__(
            self,
            embedding_size,
            num_classes,
            s=30,
            m=0.5):

        super().__init__()

        self.s = s
        self.m = m

        self.weight = nn.Parameter(
            torch.FloatTensor(
                num_classes,
                embedding_size
            )
        )

        nn.init.xavier_uniform_(
            self.weight
        )

    def forward(
            self,
            embeddings,
            labels):

        cosine = F.linear(
            F.normalize(embeddings),
            F.normalize(self.weight)
        )

        theta = torch.acos(
            torch.clamp(
                cosine,
                -1 + 1e-7,
                1 - 1e-7
            )
        )

        target = torch.cos(
            theta + self.m
        )

        one_hot = torch.zeros_like(
            cosine
        )

        one_hot.scatter_(
            1,
            labels.view(-1,1),
            1
        )

        logits = (
            one_hot * target
            +
            (1-one_hot)*cosine
        )

        logits *= self.s

        return logits