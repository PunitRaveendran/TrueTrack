# Tower Dataset Notes: OpenCellID India (MCC 404 & 405)

## 1. Overview
The cell tower dataset is located in `india_spec_towers/` and consists of two primary OpenCellID dumps:
- `404.csv`: 1,810,097 records (MCC 404 - India: Vodafone Idea, Airtel, BSNL, Reliance)
- `405.csv`: 684,762 records (MCC 405 - India: Reliance Jio LTE, Airtel, Tata/Docomo)
- Total cell towers: **2,494,859 records**
- Towers in Greater Hyderabad / HITEC City Corridor Bounding Box: **29,987 cell towers**

## 2. Field Availability & Schema Analysis
Each CSV contains the following columns:
```
radio, mcc, mnc, lac, cid, changeable_0, long, lat, range, sample, changeable_1, created, updated, avgsignal
```

| Field | Type | Availability | Description & Usage |
|---|---|---|---|
| `lat` | Float (deg) | 100% | Tower transmitter latitude |
| `long` | Float (deg) | 100% | Tower transmitter longitude |
| `radio` | String | 100% | Cellular technology: `GSM` (900/1800 MHz), `UMTS` (2100 MHz), `LTE` (1800/2300 MHz) |
| `range` | Float (meters) | 100% | **Primary transmit strength proxy**: Estimated coverage radius in meters (typically 1000m to 5000m for macrocells, 200m to 500m for microcells) |
| `avgsignal` | Integer | Sparse (often 0) | Measured RSSI at reference point |
| `mcc` | Integer | 100% | Mobile Country Code (404 / 405 = India) |
| `mnc` | Integer | 100% | Mobile Network Operator Code |

## 3. Path-Loss & EIRP Proxy Model (Phase 3 Specification)
Direct transmit power (EIRP) is not explicitly given in raw crowdsourced OpenCellID dumps. However, per 3GPP TR 36.942 / TR 25.942 and ITU-R P.1411 recommendations:
1. **Nominal Base Station EIRP:**
   - Macrocell (`range >= 1000m`): $P_{tx} = +43\text{ dBm}$ (20W) to $+46\text{ dBm}$ (40W).
   - Microcell / Small Cell (`range < 1000m`): $P_{tx} = +30\text{ dBm}$ (1W) to $+33\text{ dBm}$ (2W).
2. **Frequency Reference $f_c$ by `radio`:**
   - `GSM`: 900 MHz
   - `UMTS`: 2100 MHz
   - `LTE`: 1800 MHz / 2300 MHz
3. **Log-Distance Path-Loss Equation:**
   $$PL(d) = PL(d_0) + 10 \cdot n \cdot \log_{10}\left(\frac{d}{d_0}\right) + X_\sigma$$
   - Reference distance $d_0 = 10\text{ m}$, free-space loss $PL(d_0) \approx 20\log_{10}(f_{\text{MHz}}) - 27.55 + 20\log_{10}(10)$.
   - Path-loss exponent $n = 3.5$ for dense urban HITEC corridor.
   - Shadow fading standard deviation $\sigma = 6.0\text{ dB}$.
4. **Physical Obstruction Override:**
   - Underpasses and subterranean tunnels (`tunnel=yes`, `layer < 0`, `covered=yes`) introduce $-35\text{ dB}$ to $-45\text{ dB}$ heavy non-line-of-sight (NLOS) penetration attenuation, forcing estimated cellular strength well below the $-115\text{ dBm}$ link-disconnect threshold.
