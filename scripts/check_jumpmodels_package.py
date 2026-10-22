"""Run the authors' `jumpmodels` package end to end on the S&P 500 with the paper's features and refit calendar, and
compare its online shift counts with Shu et al. Table 3 (see DEVIATIONS.md#S6)."""

import numpy as np
import pandas as pd
from jumpmodels.jump import JumpModel
from jumpmodels.preprocess import DataClipperStd, StandardScalerPD

from regime_switch import shu2024
from regime_switch.metrics import shifts_per_year
from scripts.replicate_shu2024 import index_inputs

px, rf = index_inputs("^GSPC", "us")
r = px.pct_change().dropna(); rf = rf.reindex(r.index).ffill(); ex = r - rf
feat = shu2024.features(ex); v = feat.notna().all(axis=1).to_numpy().argmax() + 252
feat, ex = feat.iloc[v:], ex.iloc[v:]
fits = shu2024.refit_dates(feat.index, feat.index[2999], 6)
paper = {0.0: 9.7, 5.0: 2.7, 15.0: 1.7, 35.0: 0.8, 70.0: 0.5, 150.0: 0.4}
def run(lam, mode):
    jm = JumpModel(n_components=2, jump_penalty=lam, cont=False, n_init=10, random_state=0)
    out = pd.Series(np.nan, index=feat.index)
    for i, d in enumerate(fits):
        end = feat.index.get_loc(d) + 1
        stop = feat.index.get_loc(fits[i + 1]) + 1 if i + 1 < len(fits) else len(feat)
        Xtr = feat.iloc[end - 3000:end]
        clip, sc = DataClipperStd(mul=3.), StandardScalerPD()
        Xp = sc.fit_transform(clip.fit_transform(Xtr))
        jm.fit(Xp, ex.iloc[end - 3000:end], sort_by="cumret")
        blk = feat.iloc[end - 1:stop] if mode == "block" else feat.iloc[end - 3000:stop]
        lab = jm.predict_online(sc.transform(clip.transform(blk)))
        out.iloc[end - 1:stop] = lab.to_numpy()[-(stop - end + 1):]
    return shifts_per_year(out["1982":"2023"].dropna().to_numpy())
for mode in ("block", "lookback"):
    print(mode, {lam: round(run(lam, mode), 1) for lam in paper}, flush=True)
print("paper   ", paper)
