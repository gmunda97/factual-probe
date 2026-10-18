# random composition alignment rows: random subject + random relation vectors from the training vocab
# so the alignment is fitted without any real fact, used for falcon-mamba and the olmo checkpoints

import torch


def field_rows(space, splits: dict) -> dict:
    tr = splits["train"]
    fields = {"subject": space.x_train_cpu - space.r_train.cpu(), "pred_value": space.r_train.cpu(), "object": space.o_train.cpu()}
    out = {}
    for f in ("subject", "pred_value", "object"):
        seen, idx = set(), []
        for i, s in enumerate(tr[f].astype(str).tolist()):
            if s not in seen:
                seen.add(s); idx.append(i)
        out[f] = fields[f][torch.tensor(idx)]
    return out


def random_pair_rows(fr: dict, n: int, seed: int = 0) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    i_s = torch.randint(fr["subject"].shape[0], (n,), generator=g)
    i_r = torch.randint(fr["pred_value"].shape[0], (n,), generator=g)
    return fr["subject"][i_s] + fr["pred_value"][i_r]
