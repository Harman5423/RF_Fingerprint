# Physical-Layer Fingerprinting for Wireless Intrusion Detection

Reference implementation for the paper **"Beyond MAC Addresses: Physical-Layer
Fingerprinting to Close the Identity Gap in Wireless Intrusion Detection"**.

The pipeline: IQ capture -> preamble extraction -> CFO correction + normalisation
-> 1-D CNN embedding (triplet loss, cosine distance) -> prototype store
-> decision engine (ACCEPT / MISMATCH / UNKNOWN / ABSTAIN).

## Run

```bash
pip install -r requirements.txt
python rf_fingerprint_ids.py
```

## Notes

- Uses a **synthetic transmitter simulator**, not real SDR captures. Results from
  it are a functional test only and must not be compared with published results
  on real data.
- To use ORACLE / WiSig / SMoRFFI, replace `DeviceSimulator` with a loader that
  returns complex arrays of shape `(N, 256)` per device.
- Paper source: `paper/cyber_security.tex`

## Structure

```
rf_fingerprint_ids.py   # full pipeline + demo
requirements.txt
paper/cyber_security.tex
```
