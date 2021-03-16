# Are You The One -- Couple Solver

## How to:

You will need a Python installation. The code was developed on Python 3.9 but should run on any newer Python3 version. You can install dependencies manually from `requirements.txt` or with `pip install -e .`.
You can then call the script to solve the matching problem with `python ./ayto/ayto.py --yaml_file_path <YAML_FILE_PATH>`, where `<YAML_FILE_PATH>` is a path to a yaml file which decodes the names of all people, matching night couples and results as well as truth booth decisions. An example is given in `ayto.yaml`.

## Remarks:
This code is untested and rather a PoC than production ready code. It was developed for the German season 2 of AYTO currently airing on TVNow/RTL. It silently assumes that we have "Males" and "Females" and we only match a Male with a Female. Furthermore, all matches and truth booth couples need to be defined in the yaml file, first with the male name followed by the female name. This was done because of pure laziness and should in no way be meant to discriminate anyone. Code modifications would be needed to enable a season like "Come on, Come all" (season 8 AYTO USA), where anyone can match with anyone. This also makes the optimization problem more complex and, hence, harder to solve.

## Optimization details:
We use two solvers to solve the problem. We solve `min norm(x,1) s.t. Ax=b, x in [0,1]` and `min norm(x,1) s.t. Ax=b, x in {0,1}`where we actually already now the minimum and it is not unique until we have enough constraints (i.e. matching nights and truth booth results). This is derived from a compressed sensing approach where we would solve `min norm(x,0) s.t. Ax=b` since we know that our solution is sparse: The vector `x` of length `males * females` decoding all possible matches is `s`-sparse where `s` defines the number of couples. This knowledge allows us to solve the underdetermined linear equation system `Ax=b` nevertheless. We use a mixed integer linear programming approach (MIP) as well as a standard linear programming approach (LP) where we only limit `x in [0,1]` but do not enforce `x` to be strictly binary. If both solutions are identical, we assume the solution to be unique and we have found all couples. Otherwise, we print all solutions our two methods found. Note: We do not actually check if the solution found is unique or if there are several solutions still possible even when both solver agree. This approach is derived from a compressed sensing approach using linear optimization trying to do sparse signal recovery (see https://en.wikipedia.org/wiki/Compressed_sensing) and not treating this as a combinatorical problem. For the latter see e.g. https://blogs.sas.com/content/operations/2018/08/14/are-you-the-one/ for an example.

## Changelog

# SPOILER ALERT

<details>
<summary>SPOILER ARE YOU THE ONE SEASON 2 GERMANY CURRENT RESULTS AFTER EPISODE 16:</summary>
  
- Aaron + Melissa
- Dario + Sabrina  
- Dominik + Vanessa  
- Germain + Christin  
- Marc + Mirjam  
- Marcel + Leonie  
- Marko + Kathleen + Vanessa_2  
- Marvin + Jill
- Maximilian + Victoria  
- Sascha + Laura 

</details>

