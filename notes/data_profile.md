# Data profile

## rents-input-flats-city-of-sydney.csv
Shape: 65,777 rows x 19 columns

| column | dtype | nulls | null % | distinct | min | max |
|---|---|---|---|---|---|---|
| gnaf_id | str | 0 | 0.0% | 65,777 |  |  |
| address | str | 0 | 0.0% | 65,753 |  |  |
| building_address | str | 0 | 0.0% | 3,656 |  |  |
| suburb | str | 0 | 0.0% | 33 |  |  |
| mb | int64 | 0 | 0.0% | 1,779 | 10130870000 | 11238060000 |
| latitude | float64 | 0 | 0.0% | 7,370 | -33.92666643 | -33.8556368 |
| longitude | float64 | 0 | 0.0% | 7,393 | 151.16226328 | 151.23709116 |
| property_type | str | 0 | 0.0% | 3 |  |  |
| bedrooms | int64 | 0 | 0.0% | 9 | 0 | 15 |
| bathrooms | float64 | 4 | 0.0% | 8 | 0.0 | 7.0 |
| car_space | float64 | 5,793 | 8.8% | 2 | 0.0 | 1.0 |
| car_spaces | float64 | 5,793 | 8.8% | 7 | 0.0 | 6.0 |
| floor_level | float64 | 31,381 | 47.7% | 162 | 1.0 | 303.0 |
| flats_in_building | int64 | 0 | 0.0% | 250 | 1 | 1062 |
| block_new_homes_2021_2026 | float64 | 4,246 | 6.5% | 148 | 0.0 | 907.0 |
| block_homes_2026 | float64 | 4,246 | 6.5% | 301 | 0.0 | 1036.0 |
| ever_advertised_for_lease | int64 | 0 | 0.0% | 2 | 0 | 1 |
| first_ever_listed | str | 279 | 0.4% | 4,376 | 2005-07-04 | 2026-07-31 |
| attrs_last_seen | str | 0 | 0.0% | 3,905 | 0001-01-01 | 2026-07-31 |

## rents-past-leases-city-of-sydney.csv
Shape: 112,457 rows x 15 columns

| column | dtype | nulls | null % | distinct | min | max |
|---|---|---|---|---|---|---|
| gnaf_id | str | 0 | 0.0% | 52,296 |  |  |
| address | str | 0 | 0.0% | 53,629 |  |  |
| building_address | str | 6,613 | 5.9% | 3,463 |  |  |
| suburb | str | 0 | 0.0% | 33 |  |  |
| mb | int64 | 0 | 0.0% | 1,920 | 10130870000 | 11238060000 |
| listing_date | str | 0 | 0.0% | 4,362 | 2012-01-04 | 2026-07-31 |
| weekly_rent | float64 | 0 | 0.0% | 630 | 150.0 | 5000.0 |
| bedrooms | float64 | 2,152 | 1.9% | 13 | 0.0 | 21.0 |
| bathrooms | float64 | 35 | 0.0% | 10 | 0.0 | 20.0 |
| car_space | float64 | 17,115 | 15.2% | 2 | 0.0 | 1.0 |
| garage_spaces | float64 | 17,115 | 15.2% | 7 | 0.0 | 7.0 |
| property_type | str | 0 | 0.0% | 3 |  |  |
| was_leased | int64 | 0 | 0.0% | 2 | 0 | 1 |
| latitude | float64 | 0 | 0.0% | 8,121 | -33.92666643 | -33.85558443 |
| longitude | float64 | 0 | 0.0% | 8,126 | 151.16226328 | 151.23709116 |

## blocks-homes-and-people-2021-2026.csv
Shape: 2,281 rows x 8 columns

| column | dtype | nulls | null % | distinct | min | max |
|---|---|---|---|---|---|---|
| mb | int64 | 0 | 0.0% | 2,281 | 10741832000 | 11238060000 |
| suburb | str | 0 | 0.0% | 33 |  |  |
| mb_cat | str | 0 | 0.0% | 9 |  |  |
| homes_2021 | float64 | 0 | 0.0% | 266 | 0.0 | 790.0 |
| homes_2026 | float64 | 0 | 0.0% | 309 | 0.0 | 1036.0 |
| people_2021 | float64 | 0 | 0.0% | 359 | 0.0 | 1706.0 |
| people_2026 | float64 | 0 | 0.0% | 404 | 0.0 | 1976.0 |
| new_homes | float64 | 0 | 0.0% | 155 | 0.0 | 907.0 |

## Links and coverage
- Lease rows kept after as-of and rent filters: 112,457 of 112,457
- Lease dates kept: 2012-01-04 to 2026-07-31
- Leases linked to an input flat: 93.9%
- Input flats with a previous lease: 49,683 of 65,777
- Input flats whose building has a lease: 65,432
- Advert parking blank: 15.2%; after input fill: 5.8%
- Input attrs_last_seen after as-of: 0
- Input first_ever_listed after as-of: 0

## Fold test sizes
- Fold A: 2,653 adverts, 2,592 scorable
- Fold B: 2,210 adverts, 2,136 scorable
