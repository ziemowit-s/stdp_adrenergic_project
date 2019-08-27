import argparse
import os
import pickle

from ltp_models.utils import get_data, filter_by_time, exclude_molecules, S845_merge

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument("--prefix", nargs='+', required=True)

    ap.add_argument("--time_start", nargs='+', required=True, type=int, help="in ms")
    ap.add_argument("--num_steps", type=int)
    ap.add_argument("--step_len", type=int, help="in ms")

    ap.add_argument("--molecules", nargs='+', help="if defined - only those molecules will be extracted from trails")
    ap.add_argument("--labels", nargs='+', help="for each prefix - name of its labels (for regression model only")

    ap.add_argument("--morphology", required=True)
    args = ap.parse_args()

    dataset = []
    for paradigm, time_start, label in list(zip(args.prefix, args.time_start, args.labels)):
        print(paradigm)
        d, name = get_data(prefix=paradigm, morpho=args.morphology, molecules=args.molecules)
        d = filter_by_time(d, num_steps=args.num_steps, step_len=args.step_len, time_start=time_start)
        d, header = exclude_molecules(d, header=args.molecules,
                                         exact=['time', 'Ca', 'Leak'] + S845_merge.split(' '), wildcard=['out', 'buf'])
        for i in range(d.shape[0]):
            dataset.append((d[i], label, name))

    os.makedirs("cached", exist_ok=True)
    with open("cached/regression_dataset_%s.pickle" % args.morphology, "wb") as f:
        pickle.dump(dataset, f)



