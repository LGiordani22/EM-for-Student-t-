# Validation — T=2400 (parziale)

Run `--full --T 2400 --n-iter 2400 --burn-in 800 --only leverage,volatility` **interrotto (killed) durante volatility branch A** — probabilmente memoria. Salvati qui i risultati parziali di `volatility` (il confronto T-lungo che conta); `leverage` (rho a T=2400) NON eseguito.

```

  validatore — modalita' full (T=2400, n_iter=2400, catene=2)

  ── volatility ──────────────────────────────────────────────────
    branch B
      k=0: phi 0.98 (vero 0.97), corr(h) 0.92
      k=1 (DEBOLE): phi 0.89 (vero 0.92), corr(h) 0.63
      k=2: phi 0.91 (vero 0.95), corr(h) 0.80
      sigma^2_eta: vero [0.0625 0.0324 0.0484] | HN [0.0718 0.0406 0.1265] | IG [0.0564 0.0512 0.0677]
      idio: phi 0.69 (vero 0.94), corr(h^eps) 0.46, sigma^2 x11.1
    branch A
      k=0: phi 0.97 (vero 0.97), corr(h) 0.90
      k=1 (DEBOLE): phi 0.84 (vero 0.92), corr(h) 0.59
      k=2: phi 0.93 (vero 0.95), corr(h) 0.77
```
