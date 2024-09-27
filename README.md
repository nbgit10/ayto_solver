# Are You The One -- Couple Solver

## How to

You will need a Python installation. The code was developed on Python 3.9 but should run on any newer Python3 version. You can install dependencies manually from `requirements.txt` or with `pip install -e .`.
You can then call the script to solve the matching problem with `python ./ayto/ayto.py --yaml_file_path <YAML_FILE_PATH>`, where `<YAML_FILE_PATH>` is a path to a yaml file which decodes the names of all people, matching night couples and results as well as truth booth decisions. An example is given in `./examples/AYTO_Season2_Germany_AfterEp18.yaml`.

## Remarks

This code is untested and rather a PoC than production ready code. It was developed for the German seasons of AYTO airing on RTL. It silently assumes that we have "Males" and "Females" and we only match a Male with a Female. Furthermore, all matches and truth booth couples need to be defined in the yaml file, first with the male name followed by the female name. This was done because of pure laziness and should in no way be meant to discriminate anyone. Code modifications would be needed to enable a season like "Come one, Come all" (season 8 AYTO US), where anyone can match with anyone. This also makes the optimization problem more complex and, hence, harder to solve.

## Optimization details

We use a MIP solver to solve the problem. We solve `min norm(x,1) s.t. Ax=b, x in {0,1}`where we actually already now the minimum and it is not unique until we have enough constraints (i.e. matching nights and truth booth results). This is derived from a compressed sensing approach where we would solve `min norm(x,0) s.t. Ax=b` since we know that our solution is sparse: The vector `x` of length `males * females` decoding all possible matches is `s`-sparse where `s` defines the number of couples. This knowledge allows us to solve the underdetermined linear equation system `Ax=b` nevertheless. We use a mixed integer linear programming approach (MIP). This algorithm will in our case always find a feasible and optimial solution, that means we always find a valid combination of couples that satisfies all information we have from matching nights and truth boothes. However, we do not know and do not check if this solution is unique. This approach is derived from a compressed sensing approach using linear optimization trying to do sparse signal recovery (see [https://en.wikipedia.org/wiki/Compressed_sensing]) and not treating this as a combinatorical problem. For the latter see e.g. [https://blogs.sas.com/content/operations/2018/08/14/are-you-the-one/] for an example.

## Known Issues

- We do not check for uniqueness of solution.
- Therefore, we do not show all possible combinations but just a single one and we don't know once there is just one feasible solution left.
- We cannot enforce two specific people to match to the same person.

## SPOILER ALERT

<details>
<summary>SPOILER ARE YOU THE ONE SEASON 4 VIP GERMANY AFTER EPISODE 18:</summary>

Proposed solution:

- Tim and Linda + Dana ✅ 
- Lucas and Tara
- Nicola and Laura L
- Lars and Jennifer
- Khan and Nadia
- Chris and Emmy ✅ 
- Alex and Gabriela
- Marc Robin and Laura
- Ozan and Anastassia ✅ 
- Antonino and Asena

They win the money in the 9th night in episode 20, despite RTL letting Nicola choose first.

</details>
