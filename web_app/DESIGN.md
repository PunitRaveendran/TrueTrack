# TrueTrack Design System Specification (DESIGN.md) - Round 3 High-Fidelity

**Aesthetic Direction**: *Precision Flight Recorder / Survey Instrument*. Quiet, calm, mathematically rigorous, Linear/Apple Maps dark mode aesthetic. Clean visual depth, restrained single-accent hierarchy, zero sci-fi tropes or neon glows.

---

## 1. Surfaces & Elevation (Near-Black Neutrals)

Three distinct elevation levels built on neutral charcoal-black tones with 1px hairline borders at ~8% white:

| Token | Hex | Role | Usage |
| :--- | :--- | :--- | :--- |
| `--bg-base` | `#0b0d10` | Elevation 0 (Base Canvas) | Map viewport background, app canvas |
| `--bg-surface` | `#12151a` | Elevation 1 (Panels & Headers) | Side panel, transport bar, top header |
| `--bg-subtle` | `#191d24` | Elevation 2 (Cards & Controls) | Hero comparison card, active controls, drawer panels |
| `--bg-hover` | `#222731` | Interactive Hover | Button hover states, interactive table rows |
| `--border-subtle` | `#232832` | 1px Hairline Rules (~8% white) | Dividers, card boundaries, table gridlines |
| `--border-focus` | `#3b4252` | Accent Hairline | Active control outlines, focused inputs |

---

## 2. Color Discipline (Information-Carrying Only)

- **Blue (`#38bdf8` / `#2563eb`)**: Exclusively reserved for TrueTrack (brand, hero numbers, trajectory, active toggles).
- **Red (`#ef4444` / `#ff5a5f`)**: Strictly represents failure/legacy drift and GPS loss.
- **Amber (`#d97706`)**: Reserved solely for the 45-second underpass blackout zone.
- **Ground Truth (`#cbd5e1`)**: Neutral muted white dashed centerline (40% opacity).

| Semantic Token | Hex | Role |
| :--- | :--- | :--- |
| `--accent-primary` | `#38bdf8` | TrueTrack trajectory line, primary buttons |
| `--accent-blue` | `#2563eb` | TrueTrack 40px directional marker fill |
| `--accent-muted` | `rgba(56, 189, 248, 0.12)` | TrueTrack chart gradient area fill, delta chip |
| `--error-primary` | `#ef4444` | Legacy divergence route, blackout alarm |
| `--error-muted` | `rgba(239, 68, 68, 0.12)` | Legacy drift area tint, warning indicator |
| `--tunnel-amber` | `#d97706` | Underpass corridor span, shaded scrubber window |
| `--tunnel-muted` | `rgba(217, 119, 6, 0.12)` | Underpass shaded time span in error chart & scrubber |

---

## 3. Map Z-Order & Layer Hierarchy

Strict visual layer stacking so TrueTrack ALWAYS renders with dominance over legacy drift:

| Layer Level | Element | Visual Style |
| :--- | :--- | :--- |
| **Z 400** | Ground Truth Road Centerline | Thin 1.5px dashed white line (`#cbd5e1`), 50% opacity |
| **Z 450** | Dim Full Corridor Route | 2px dim line (`#334155`) with 4px dark casing (`#07090d`) |
| **Z 500** | Legacy Traveled Path | 2.5px red line (`#ef4444`) with 5px dark casing (`#07090d`) |
| **Z 600** | TrueTrack Traveled Path | 3.5px bright blue line (`#38bdf8`) with 5.5px dark casing (`#07090d`) |
| **Z 620** | EKF Covariance Ellipse & Tethers | Translucent blue ellipse (`rgba(56, 189, 248, 0.15)`), thin tether lines |
| **Z 700** | Legacy Marker Puck | 28px hollow/outlined red ring (`border: 2.5px solid #ef4444; background: transparent;`) |
| **Z 800** | TrueTrack Marker Puck | **40px directional marker**: solid blue fill (`#2563eb`), 2px crisp white ring, soft 2.5s outer pulse ring |

---

## 4. Typography Scale & Numerals

- **Fonts**: Local system stack with tabular numerals:
  - Sans: `-apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif`
  - Mono: `"SF Mono", "Segoe UI Mono", "Cascadia Code", "Roboto Mono", monospace`
- **Scale**:
  - `56px`: Hero Drift Figures (tabular-nums, line-height 1.0, zero letter-spacing)
  - `20px`: Modal / Intro card titles
  - `15px`: Primary instrument titles & prominent callout numbers
  - `13px`: Body text, primary table cells, control labels
  - `12px`: Section headers (sentence-case sans-serif, muted `#8b94a2`)
  - `11px`: Axis tick labels, table headers, micro badges
- **Numeral Rules**:
  - `font-variant-numeric: tabular-nums; letter-spacing: 0;` across all numbers.
  - Zero letter-spacing gap between numerals and decimal points (e.g. `21.9 m` renders tight).
  - Unit `"m"` aligned cleanly to baseline at 18px.

---

## 5. Spacing, Geometry & Iconography

- **Grid**: 8px spatial grid with 4px sub-grid (`4px`, `8px`, `12px`, `16px`, `20px`, `24px`).
- **Radii**: `6px` for buttons and controls; `10px` for panels and cards.
- **Iconography**: Clean 1.5px stroke inline SVGs, zero external font icons.
