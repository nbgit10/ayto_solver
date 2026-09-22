# AYTO Matches — brand spec

Source: the existing neon system in the linked project
`ayto_solver/frontend/src/styles/global.css` (user-supplied brand source).
The user asked to keep the neon colours while modernising the design, so this
neon palette overrides the active Webflow design system for colour.

**One sentence:** A near-black violet canvas carrying electric data colours —
violet for the brand, blue/pink as the show's male/female pigments, gold for
confirmed matches, and a cold→hot heat ramp that encodes match probability.

## Six core tokens

| Token      | Hex       | OKLch                        | Role                                   |
| ---------- | --------- | ---------------------------- | -------------------------------------- |
| `--bg`     | `#0c0a12` | `oklch(15.2% 0.017 295.7)`   | Page canvas — near-black, violet bias  |
| `--surface`| `#15111d` | `oklch(18.9% 0.025 299.3)`   | Cards / raised panels                  |
| `--fg`     | `#fff8f6` | `oklch(98.4% 0.008 36.6)`    | Primary text — warm bone               |
| `--muted`  | `#8c7d91` | `oklch(61.0% 0.035 317.7)`   | Secondary text, labels                 |
| `--border` | `#30243c` | `oklch(28.5% 0.046 306.6)`   | Hairlines, dividers, card edges        |
| `--accent` | `#b73cff` | `oklch(62.5% 0.271 309.1)`   | Brand + primary action — electric violet |

## Semantic data colours (brand-owned, not decoration)

| Token       | Hex       | Meaning                                  |
| ----------- | --------- | ---------------------------------------- |
| `--him`     | `#65a6ff` | the men (cool blue)                      |
| `--her`     | `#ff6fae` | the women (pink)                         |
| `--match`   | `#b73cff` | a perfect match (blue + pink = violet)   |
| `--match-hi`| `#d47cff` | hover / emphasis violet                   |
| `--gold`    | `#ffd166` | confirmed / locked match                  |
| `--danger`  | `#ff5d5d` | contradictory data                        |
| heat ramp   | `#2a2435 → #3a5fb0 → #7b3fd6 → #b026ff → #ff3d8b` | 0→100% probability |

## Typography

- **Display:** `Space Grotesk` (600–700, tight tracking) — modern geometric
  grotesk, the "broadcast" voice.
- **Body:** `Hanken Grotesk` (400–700) — clear, humanist, kept from the
  original site.
- **Mono:** `Space Mono` (400/700) — every label, percentage, stat and date.
  The design + code duality is part of the brand.
- Fallbacks: system grotesk / Georgia for display, system sans for body.

## Observed rules that define the visual language

1. **Dark violet canvas, neon on top.** Light only ever appears as text or a
   focused glow — never as a surface fill.
2. **Colour is information, not ornament.** Blue = men, pink = women,
   gold = confirmed, violet = the match/probability heat. A given screen uses
   these consistently; no decorative rainbow.
3. **Mono for anything numeric.** Probabilities, counts, episode numbers and
   labels carry `--font-mono` with wide tracking and uppercase.
4. **Glow is reserved for the one focal element** — the clarity gauge, the
   active matrix cell, the live/current marker. Everything else sits flat on
   `--bg-2`.
5. **Conservative geometry.** ~14px card radius, 1px `--border` hairlines, a
   faint top highlight on surfaces. No soft pillowy shadows; depth reads as
   neon emission, not material.