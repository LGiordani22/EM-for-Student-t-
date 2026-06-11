"""
src/forecast/ — Real-time nowcasting pipeline (rewritten from scratch).

This package replaces the old monolithic src/forecast_evaluation.py with a
clean, modular real-time pipeline, one brick at a time:

    data_import.py   — load historical FRED-MD vintages + ALFRED real-time
                       series (GDPC1, NFCI).  [implemented]

    (to come: panel construction, DFM estimation per vintage, nowcast,
     benchmarks, evaluation, figures — NOT implemented yet.)
"""
