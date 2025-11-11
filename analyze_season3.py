"""Analyze Season 3 data to understand which nights are missing confirmed pairs."""
import json

with open('examples/json/AYTO_Season3_Germany_AfterEp19.json', 'r') as f:
    data = json.load(f)

print("=" * 70)
print("Season 3 Analysis - Checking for Missing Confirmed Pairs")
print("=" * 70)
print()

# Find confirmed truth booths
confirmed_pairs = []
for i, tb in enumerate(data['truth_booths']):
    if tb['match']:
        confirmed_pairs.append((tb['pair'][0], tb['pair'][1], i+1))
        print(f"Truth Booth {i+1}: {tb['pair'][0]}-{tb['pair'][1]} = CONFIRMED MATCH")

print()
print(f"Total confirmed pairs: {len(confirmed_pairs)}")
print()

# Check each matching night
for night_idx, night in enumerate(data['matching_nights']):
    pairs = night['pairs']
    matches = night['matches']

    print(f"Matching Night {night_idx + 1}:")
    print(f"  Pairs in this night: {len(pairs)}")
    print(f"  Expected pairs: 10")
    print(f"  Matches: {matches}")

    # Check which confirmed pairs are present
    present_confirmed = []
    missing_confirmed = []

    for male, female, tb_num in confirmed_pairs:
        pair_in_night = False
        for pair in pairs:
            if (pair[0] == male and pair[1] == female):
                pair_in_night = True
                break

        if pair_in_night:
            present_confirmed.append((male, female, tb_num))
        else:
            missing_confirmed.append((male, female, tb_num))

    if present_confirmed:
        print(f"  Present confirmed pairs:")
        for male, female, tb_num in present_confirmed:
            print(f"    - {male}-{female} (TB{tb_num})")

    if missing_confirmed:
        print(f"  MISSING confirmed pairs:")
        for male, female, tb_num in missing_confirmed:
            print(f"    - {male}-{female} (TB{tb_num})")

    if len(pairs) < 10:
        print(f"  ⚠️  WARNING: Only {len(pairs)} pairs instead of 10!")
        print(f"  ⚠️  Need to add {10 - len(pairs)} missing confirmed pair(s)")
        print(f"  ⚠️  Matches should be increased from {matches} to {matches + len(missing_confirmed)}")

    print()
