# Traditional frontend baseline

Real AISHELL-1 test subset; direct-path reference. CER not run in this stage.

| Frontend | N | SI-SDR (dB) | PESQ-WB | STOI | CER |
| --- | ---: | ---: | ---: | ---: | --- |
| ref_mic | 200 | -8.3272 | 1.0638 | 0.4860 | not run |
| das | 200 | -6.0904 | 1.0790 | 0.5242 | not run |
| oracle_mvdr | 200 | -2.5008 | 1.1024 | 0.6163 | not run |

Means are utterance averages. Check summary.json valid_counts and per-case errors.
Oracle MVDR uses true DOA and a known target-free prefix; it is a privileged
reference, not a guarantee of higher perceptual scores under steering mismatch.
