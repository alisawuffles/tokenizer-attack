import argparse
import bisect
import sys
import time
from functools import partial
from pathlib import Path

import gurobipy as gp
import numpy as np
import simdjson as json
import tqdm.auto as tqdm
from gurobipy import GRB

from prqrs import PriorityQueue
from utils import load_data


def lazy_optimize(
    merges,
    pair_counts,
    lang_denoms,
    verbose=True,
    num_merges=3000,
    competitor_batch_size=10,
    max_iters=300000000,
    max_add=100,
    debug=False,
):
    langs = list(pair_counts.keys())
    num_langs = len(pair_counts)
    merge_subset = [str(merge) for merge in merges[:num_merges]]
    P = partial(tqdm.tqdm, dynamic_ncols=True) if verbose else lambda x, **_: x
    # precomputation step
    if verbose:
        print("precomputation")

    for lang, apc in pair_counts.items():
        if len(apc) < num_merges:
            print(f"warning: insufficient merges for lang \"{lang}\": {len(apc)}")

    # Build the pair maps
    pair_to_id, id_to_pair = {}, []
    for i in P(range(num_merges), desc="mapping pairs"):
        for lang, apc in pair_counts.items():
            if i < len(apc):
                for k in apc[i].keys():
                    if k not in pair_to_id:
                        pair_to_id[k] = len(id_to_pair)
                        id_to_pair.append(k)
        if i == 0:
            initial_cut = len(id_to_pair)

    # compute the initial pair array
    initial_pair_array = np.zeros((num_langs, initial_cut), np.float64)
    for i, (lang, apc) in enumerate(P(pair_counts.items(), desc="building IPA")):
        for pair, count in apc[0].items():
            initial_pair_array[i, pair_to_id[pair]] = count

    # compute the count deltas
    def compute_count_deltas(all_pair_counts):
        cumulative_pair_counts = {}
        cumulative_pair_counts.update(all_pair_counts[0])
        deltas = []
        for pc in all_pair_counts[1:]:
            deltas.append(
                {
                    pair: count - cumulative_pair_counts.get(pair, 0)
                    for pair, count in pc.items()
                }
            )
            cumulative_pair_counts.update(pc)
        return deltas

    delta_counts = {
        lang: compute_count_deltas(apc[:num_merges])
        for lang, apc in P(pair_counts.items(), desc="taking deltas")
    }

    # build the delta count array
    delta_count_arrays = []
    for i in P(range(num_merges - 1), desc="building DCA"):
        stack = {}
        for j, lang in enumerate(pair_counts):
            if i < len(delta_counts[lang]):
                for pair, dcount in delta_counts[lang][i].items():
                    pid = pair_to_id[pair]
                    if pid not in stack:
                        stack[pid] = np.zeros(num_langs)
                    stack[pid][j] = dcount
        keys = np.array(list(stack.keys()), dtype=np.int64)
        if stack:
            values = np.hstack([stack[k][:, None] for k in keys])
        else:
            values = np.zeros((num_langs, 0))
        delta_count_arrays.append((keys, values))

    # transpose the pair counts
    pair_counts2 = {}
    for lang, apc in P(pair_counts.items(), desc="building PC2"):
        for i, pc in enumerate(apc[:num_merges]):
            for pair, count in pc.items():
                key = lang, pair_to_id[pair]
                pair_counts2.setdefault(key, [])
                pair_counts2[key].append((i, count))

    # initialize the model
    if verbose:
        print("model init")

    with gp.Env(empty=True) as env:
        if not verbose or True:
            env.setParam("OutputFlag", 0)
        # env.setParam("LPWarmStart", 2)
        env.start()
        m = gp.Model("tokenizer_attack", env=env)

    # lang_v = m.addMVar((num_langs,), 0, 1, name="lang_v")
    lang_v = [m.addVar(0, 1, name=name) for name in langs]
    # m.addConstr(lang_v.sum() == 1)
    m.addConstr(sum(lang_v) == 1)
    # viol_v = m.addMVar((num_merges,), 0, name="viol_v")
    viol_v = [m.addVar(lb=0, name=f"viol{i}") for i in range(num_merges)]
    lang_vals = np.ones(len(pair_counts)) / len(pair_counts)
    viol_vals = [0 for _ in range(len(viol_v))]
    pair_viol_v, pair_viol_vals = {}, {}
    # pair_viol_v = m.addVars(range(len(id_to_pair)))
    # pair_viol_vals = {v: 0 for v in pair_viol_v}
    denoms = np.array([lang_denoms[lang] for lang in langs])
    primal_tol = m.getParamInfo("FeasibilityTol")[2]

    if verbose:
        print("do optimization")
    # do the optimization
    active_set = [None] * len(merge_subset)
    all_constraints, missing_merges = set(), set()

    start_time, solver_time = time.perf_counter(), 0
    for epoch in range(max_iters):
        # initialize the priority queue
        mix = lang_vals / denoms
        prios = mix @ initial_pair_array
        prios.resize(len(id_to_pair))
        for pair, pviol in pair_viol_vals.items():
            prios[pair] -= max(0, pviol)

        pq = PriorityQueue.from_numpy(prios)
        priohist = [prios.copy()]

        def pop():
            while True:
                item = pq.pop()
                # cursed float equality test
                if prios[item.value] == item.priority:
                    return item
                elif debug:
                    assert (
                        prios[item.value] < item.priority
                    ), f"{prios[item.value]}, {item.priority}"

        def get_cur_count(i, lang, pair):
            transcript = pair_counts2.get((lang, pair), [])
            if not transcript:
                return 0
            idx = bisect.bisect_right(transcript, (i, 2 << 62))
            if idx == 0:
                return 0
            return transcript[idx - 1][1]

        new_constraints, new_variables = set(), set()
        for i, (merge, viol) in enumerate(zip(P(merge_subset), viol_vals)):
            if merge not in pair_to_id or np.isclose(pair_to_id[merge], 0):
                missing_merges.add(merge)
                # print(f"Warn: could not find merge '{merge}'")
            else:
                mid = pair_to_id[merge]
                # we want the prio without the pair violation
                mprio = prios[mid] + max(0, pair_viol_vals.get(mid, 0))
                popped = []
                active_set[i] = []
                cutoff = mprio + max(0, viol) + primal_tol
                candidates = set()

                while len(candidates) < competitor_batch_size:
                    item = pop()
                    tid, tprio = item.value, item.priority
                    popped.append(item)
                    active_set[i].append((item.value, item.priority))
                    if tprio > cutoff:
                        assert tid != mid, f"{tprio} {pair_viol_vals.get(item, 0)}"
                        if (i, tid) not in all_constraints:
                            candidates.add(tid)
                    else:
                        break
                    # assert not tprio < cutoff, f"{[p.value for p in popped]}, {mid=}, {mprio=}, {tprio=}, {viol=}, {cutoff=}"

                for item in popped:
                    pq.push(item)

                candidates.add(mid)

                cand_counts = {
                    cand: np.array(
                        [get_cur_count(i, lang, cand) / lang_denoms[lang] for lang in langs]
                    )
                    for cand in candidates
                }

                if debug:
                    for pair, coeffs in cand_counts.items():
                        pviol = max(0, pair_viol_vals.get(pair, 0))
                        count_val = lang_vals @ coeffs - pviol
                        prio_val = prios[pair]
                        assert np.isclose(
                            count_val, prio_val
                        ), f"{i}, {pair}, {count_val}, {prio_val}, {pviol}"

                A = []
                for cand in candidates:
                    if cand == mid:
                        continue
                    if cand not in pair_viol_v:
                        pair_viol_v[cand] = m.addVar(lb=0, name=f"pviol{i}")
                        new_variables.add(cand)

                    A.append(cand_counts[mid] - cand_counts[cand])

                    new_constraints.add((i, cand))
                    all_constraints.add((i, cand))

                if A:
                    n = len(cand_counts) - 1
                    A = np.hstack([np.vstack(A), np.ones((n, 1)), np.eye(n)])
                    x = (
                        lang_v
                        + [viol_v[i]]
                        + [pair_viol_v[cand] for cand in candidates if cand != mid]
                    )
                    b = np.zeros(n)
                    m.addMConstr(A, x, ">=", b)

            if i != num_merges - 1:
                items, dcounts = delta_count_arrays[i]
                mixdcounts = mix @ dcounts

                if debug:
                    for idx, mdc in zip(items, mixdcounts):
                        assert (
                            mdc <= 1e-7 or prios[idx] <= 1e-7
                        ), f"{idx=}, {mdc=}, {prios[idx]=}"

                prios[items] += mixdcounts
                pq.push_batch(items, prios[items])
                priohist.append((items, mixdcounts))

            if len(new_constraints) >= max_add:
                break

        print(f"exited at merge {i}")
        if len(new_constraints) == 0:
            print("added no constraints -- exiting")
            break
        elif len(new_constraints) > 10:
            print(f"added {len(new_constraints)} new constraints")
        else:
            print(f"added constraints {new_constraints}")

        if len(new_variables) > 10:
            print(f"added {len(new_variables)} new variables")
        elif new_variables:
            new_vars_lookup = {id_to_pair[v] for v in new_variables}
            print(f"added variables {new_vars_lookup}")

        m.setObjective(
            gp.quicksum(pair_viol_v.values()) + gp.quicksum(viol_v), GRB.MINIMIZE
        )
        solver_start = time.perf_counter()
        m.optimize()
        solver_time += time.perf_counter() - solver_start

        lang_vals = np.array([lv.X for lv in lang_v])
        viol_vals = [v.X for v in viol_v]
        pair_viol_vals = {k: v.X for k, v in pair_viol_v.items()}
        print(f"loss: {m.ObjVal} ({sum(viol_vals)}, {sum(pair_viol_vals.values())})")
        print(
            dict(sorted(zip(langs, lang_vals), key=lambda langfreq: -langfreq[1])[:10])
        )

    return dict(
        lang_vals=dict(zip(langs, lang_vals.tolist())),
        viol_vals=viol_vals,
        pair_viol_vals=pair_viol_vals,
        missing_merges=missing_merges,
        active_set=active_set,
        timing=dict(solver_time=solver_time, opt_time=time.perf_counter() - start_time),
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog="TokenizerInference")
    parser.add_argument("data_root")
    parser.add_argument(
        "--merges", type=int, help="Number of merges to consider", default=30000
    )
    parser.add_argument(
        "--denom", type=str, help="Which normalization to apply", default="pairs"
    )
    parser.add_argument(
        "--variant", type=str, help="Which language subdir to run", default=None
    )
    parser.add_argument(
        "--langlist", type=str, help="Which language list to run", default=None
    )
    args = parser.parse_args()
    root = Path(args.data_root)
    print(Path.cwd(), root)
    assert Path.cwd().exists()
    assert root.resolve().exists()
    if not (root / "merges.txt").exists():
        print("incomplete: no merges")
        sys.exit()

    if (root / "meta.json").exists():
        with (root / "meta.json").open() as f:
            meta = json.load(f)
            langs = meta["byte_count"].keys()
    else:
        langs = [subdir.name for subdir in root.iterdir()]

    for lang in langs:
        subdir = root / lang
        if "." in subdir.name:
            continue
        if args.variant is not None:
            subdir = subdir / args.variant
        if not (subdir / "all_pair_counts.json").exists():
            print(f"incomplete: {lang}")
            sys.exit()

    merges, pair_counts, training_counts = load_data(root, verbose=True, subdir=args.variant, langlist=args.langlist)

    kwargs = dict(
        verbose=True,
        num_merges=args.merges,
        competitor_batch_size=10,
        max_iters=10**10,
        max_add=100,
        debug=False,
    )

    solution = lazy_optimize(merges, pair_counts, training_counts[args.denom], **kwargs)

    # Sort the lang vals for convenience
    solution["lang_vals"] = dict(
        sorted(solution["lang_vals"].items(), key=lambda langfreq: langfreq[1])
    )

    # Convert set to list for JSON
    solution["missing_merges"] = list(solution["missing_merges"])

    # Dump the args into the output as well
    solution['kwargs'] = kwargs
    solution['kwargs']['denom'] = args.denom
    
    print(solution["lang_vals"])

    variant_str = "" if args.variant is None else f"_{args.variant}"
    langlist_str = "" if args.langlist is None else f"_{args.langlist}"
    with (root / f"solution_{args.denom}_{args.merges}{variant_str}{langlist_str}.json").open("w") as f:
        json.dump(solution, f)
