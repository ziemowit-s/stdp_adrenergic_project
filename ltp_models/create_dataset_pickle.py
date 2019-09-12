import argparse
import os
import pickle

from ltp_models.utils import get_data, filter_by_time, exclude_molecules, S845_merge

CACHED = "cached"


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument("--prefix", nargs='+', required=True)

    ap.add_argument("--time_start", nargs='+', type=int, help="in ms")
    ap.add_argument("--num_steps", type=int, required=True)
    ap.add_argument("--step_len", type=int, help="in ms")

    ap.add_argument("--molecules", nargs='+', help="if defined - only those molecules will be extracted from trails. "
                                                   "Order doesn't matter.")
    ap.add_argument("--labels", nargs='+', help="for each prefix - name of its labels (for regression model only")

    ap.add_argument("--morphology", required=True)
    args = ap.parse_args()

    dataset = []
    for i, (paradigm, label) in enumerate(zip(args.prefix, args.labels)):
        time_start = 0
        if args.time_start:
            time_start = args.time_start[i]

        print(paradigm)
        d, header, paradigm_name = get_data(prefix=paradigm, morpho=args.morphology, molecules=args.molecules)
        d = filter_by_time(d, num_steps=args.num_steps, step_len=args.step_len, time_start=time_start)
        d, header = exclude_molecules(d, header=args.molecules, exact=['time', 'Ca', 'Leak'] + S845_merge.split(' '),
                                      wildcard=['out', 'buf'])
        dataset.append((d, header, label, paradigm_name))

    os.makedirs(CACHED, exist_ok=True)
    fname = "%s/regression_dataset_%s_%s_steps.pickle" % (CACHED, args.morphology, args.num_steps)
    with open(fname, "wb") as f:
        pickle.dump(dataset, f)



