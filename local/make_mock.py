"""Synthetic stand-ins for the three supplied files, with known planted effects.

Planted truth (so stages can be tested against it): rents grow 3.5%/yr to 2023 then 10%/yr;
parking is worth $40-110/week by suburb; each floor is worth $4/week. Lease rows after
31 July 2026 stay in the file with rent x3 as a leak canary; the true values go to
_frozen_truth.csv for an end-to-end score. The lease column names are a guess.
"""
import numpy as np
import pandas as pd

from src import config as C

OUT = C.ROOT / "data" / "mock"
rng = np.random.default_rng(7)
N = C.N_INPUT
SUBURBS = ["Sydney", "Waterloo", "Zetland", "Surry Hills", "Pyrmont", "Ultimo", "Redfern", "Glebe", "Potts Point",
           "Darlinghurst", "Rosebery", "Alexandria", "Erskineville", "Newtown", "Chippendale", "Camperdown",
           "Paddington", "Woolloomooloo", "Haymarket", "Millers Point", "Dawes Point", "The Rocks", "Barangaroo",
           "Elizabeth Bay", "Rushcutters Bay", "Beaconsfield", "Darlington", "Forest Lodge", "Annandale",
           "Eveleigh", "St Peters", "Moore Park", "Centennial Park"]
assert len(SUBURBS) == 33


def log_index(d):
    y = (pd.to_datetime(d) - pd.Timestamp("2023-01-01")).days / 365.25
    return np.where(y < 0, 0.035 * y, 0.10 * y)


n_b = 4200
sizes = np.maximum(1, rng.lognormal(2.6, 1.0, n_b).astype(int))
b_of_flat = np.repeat(np.arange(n_b), sizes)
b_of_flat = rng.permutation(np.resize(b_of_flat, N))
s_of_b = rng.choice(33, n_b, p=rng.dirichlet(np.ones(33) * 3))
sub_base = rng.uniform(520, 880, 33)
sub_prem = rng.uniform(40, 110, 33)
b_eff = rng.normal(0, 0.10, n_b)
b_num = rng.integers(1, 400, n_b)
b_street = rng.choice(["GEORGE STREET", "DUNKERLEY PLACE", "BOURKE ROAD", "HARRIS STREET", "CROWN STREET", "OXFORD STREET"], n_b)
b_lat = -33.89 + rng.normal(0, 0.012, 33)[s_of_b] + rng.normal(0, 0.002, n_b)
b_lon = 151.20 + rng.normal(0, 0.012, 33)[s_of_b] + rng.normal(0, 0.002, n_b)
b_mb = 11205000000 + rng.integers(0, 900, n_b) * 100
b_height = rng.integers(2, 40, n_b)
b_addr = np.char.add(np.char.add(b_num.astype(str), " "), b_street) + np.char.add(" #", np.arange(n_b).astype(str))

s = s_of_b[b_of_flat]
beds = rng.choice([0, 1, 2, 3, 4, 15], N, p=[.08, .42, .38, .1, .0199, .0001])
baths = np.clip(np.where(beds >= 2, 2, 1) - rng.binomial(1, .35, N), 1, None)
car = rng.binomial(1, np.clip(.35 + .2 * beds, 0, .9), N)
floor = (rng.random(N) * b_height[b_of_flat]).astype(int)
f_eff = rng.normal(0, 0.06, N)
ids = np.array([f"GANSW{700000000 + i}" for i in rng.permutation(N)])
unit = np.arange(N) % 900 + 1
flats_in_b = pd.Series(b_of_flat).map(pd.Series(b_of_flat).value_counts()).to_numpy()


def true_rent(flat_idx, dates):
    r = sub_base[s[flat_idx]] * np.exp(0.33 * np.minimum(beds[flat_idx], 5) + b_eff[b_of_flat[flat_idx]]
                                       + f_eff[flat_idx] + 0.05 * (baths[flat_idx] - 1) + log_index(dates))
    r = r + 4 * floor[flat_idx] + car[flat_idx] * sub_prem[s[flat_idx]]
    return np.round(r * np.exp(rng.normal(0, .05, len(r))) / 5) * 5


never = rng.choice(N, 16_094, replace=False)
adv = np.setdiff1d(np.arange(N), never)
per = rng.poisson(2.8, len(adv)) + 1
li = np.repeat(adv, per)
days = (pd.Timestamp("2026-10-31") - pd.Timestamp("2016-01-01")).days
ld = pd.Timestamp("2016-01-01") + pd.to_timedelta((rng.beta(2.2, 1, len(li)) * days).astype(int), unit="D")
rent = true_rent(li, ld)
late = ld > C.AS_OF
leases = pd.DataFrame({"gnaf_id": ids[li], "building_address": b_addr[b_of_flat[li]], "advert_date": ld.strftime("%Y-%m-%d"),
                       "weekly_rent": rent, "bedrooms": beds[li], "bathrooms": baths[li],
                       "car_space": np.where(rng.random(len(li)) < .15, np.nan, car[li])})
car_in = car.astype(float)
car_in[rng.choice(N, 5_793, replace=False)] = np.nan

win = (ld >= C.SCORE_START) & (ld <= C.SCORE_END)
truth = leases[win].sample(n=min(1699, int(win.sum())), random_state=1)
truth = truth.rename(columns={"advert_date": "target_date", "weekly_rent": "rent", "car_space": "advert_car_space"})
truth["input_car_space"] = truth.gnaf_id.map(dict(zip(ids, car_in)))
leases.loc[late, "weekly_rent"] *= 3  # leak canary

fl = floor.astype(float)
fl[rng.choice(N, N - 34_396, replace=False)] = np.nan
seen = pd.Timestamp("2018-01-01") + pd.to_timedelta(rng.integers(0, 3200, N), unit="D")  # some run past as-of on purpose
seen_s = seen.strftime("%Y-%m-%d").to_numpy().copy()
seen_s[rng.random(N) < .2] = "0001-01-01"
in_block = rng.random(N) > 4246 / N
homes = np.where(in_block, rng.integers(20, 900, N), np.nan)
inp = pd.DataFrame({
    "gnaf_id": ids, "address": [f"UNIT {u}, {a}" for u, a in zip(unit, b_addr[b_of_flat])],
    "building_address": b_addr[b_of_flat], "suburb": [SUBURBS[i] + " (NSW)" for i in s], "mb": b_mb[b_of_flat],
    "latitude": b_lat[b_of_flat], "longitude": b_lon[b_of_flat], "property_type": rng.choice(["apartment", "unit", "studio"], N),
    "bedrooms": beds, "bathrooms": baths, "car_space": car_in, "car_spaces": np.where(car_in > 0, 1, np.where(car_in == 0, 0, np.nan)),
    "floor_level": fl, "flats_in_building": flats_in_b, "block_new_homes_2021_2026": np.where(in_block, rng.poisson(8, N), np.nan),
    "block_homes_2026": homes, "ever_advertised_for_lease": np.isin(np.arange(N), adv).astype(int),
    "first_ever_listed": (pd.Timestamp("2016-08-24") + pd.to_timedelta(rng.integers(0, 3600, N), unit="D")).strftime("%Y-%m-%d"),
    "attrs_last_seen": seen_s,
})
blocks = pd.DataFrame({"mb": np.unique(b_mb)})
blocks["homes_2021"] = rng.integers(20, 800, len(blocks))
blocks["people_2021"] = blocks.homes_2021 * 2
blocks["homes_2026"] = blocks.homes_2021 + rng.integers(0, 90, len(blocks))
blocks["people_2026"] = blocks.homes_2026 * 2

OUT.mkdir(parents=True, exist_ok=True)
inp.to_csv(OUT / C.INPUT_CSV.name, index=False)
leases.to_csv(OUT / C.LEASES_CSV.name, index=False)
blocks.to_csv(OUT / C.BLOCKS_CSV.name, index=False)
truth.to_csv(C.MOCK_TRUTH_CSV, index=False)
print(f"mock: input {inp.shape}, leases {leases.shape} ({int(late.sum())} after as-of), blocks {blocks.shape}, truth {truth.shape}")
