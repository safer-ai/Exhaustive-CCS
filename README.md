# Discovering Latent Knowledge Without Supervision

This is a lighty edited version of the original [Discovering Latent Knowledge Without Supervision](https://arxiv.org/pdf/2212.03827.pdf) code, which is the zip file linked in the [Github Repo of the paper](https://github.com/collin-burns/discovering_latent_knowledge/).

Both the Github repo and the zip file contain bugs, which are here fixed.

## How to run

- Install the required packages with `pip install -r requirements.txt`
- Run the `source generate_all.sh` to generate the data (it might take >24h). Edit it if you want to generate only a subset of the data.
- Run the `source extract_relevant.sh` to run all the experiments (it might take >24h too). Edit it if you want to run only a subset of the experiments.
- Run `plots.py` as a Python notebook to generate the plots.

## Additional modifications

- Add a `"Random"` baseline (which is CCS but without any training)
- Save the parameters of the probe for all methods
- Add a `--save_states` flag to `extraction_main.py` which saves the probabilities and labels given by the runs.
- Add a `--test_on_train` flag to `extraction_main.py` to measure train performance instead of test performance
- Add a `requirements.txt` file to install the required packages, and scripts file for ease of use
- Add a `plots.py` file to generate the plots
- Rename `"Prob"`to `"CCS"`
- Use HuggingFace's default cache
- Add Recursive CCS (RCCS).

My aim was to make the diff easily readable, so I haven't done any refactor, major renamings, format changes, etc.

If you're not interested by Recursive CCS, you can use the `without-rccs` branch.

## Recursive CCS

RCCS applies CCS multiple times, with the contraint that iteration n should find a probe which direction is orthogonal to the directions found by iterations 0, ..., n-1.

I wanted to make the diff still relatively small, so each iteration of RCCS is a separate experiment, which uses the parameters of the probes found by previous ones. To run 20 iteration of RCCS, pass RCCS0, ..., RCCS19 to `--method_list` (it should start by RCCS0). Stats will be saved for each iterations as a separate experiment, and the concatenation of probes' parameters will be saved as if you had run a method named `RCCS`.


