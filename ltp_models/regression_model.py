import argparse
import matplotlib.pyplot as plt
from ltp_models.utils import get_data, filter_by_time, exclude_molecules, agregate_trails, S845_merge

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument("--prefix", nargs='+', required=True)

    ap.add_argument("--time_start", nargs='+', required=True, type=int, help="in ms")
    ap.add_argument("--num_steps", type=int)
    ap.add_argument("--step_len", type=int, help="in ms")

    ap.add_argument("--molecules", nargs='+', help="if defined - only those molecules will be extracted from trails")
    ap.add_argument("--labels", nargs='+', help="for each prefix - name of its labels (for regression model only")

    ap.add_argument("--filter", type=float, default=None)
    ap.add_argument("--morphology", required=True)
    ap.add_argument("--trials", nargs='+', help="Trial numbers if required. Default: take all trials", default=None,
                    type=int)
    ap.add_argument("--agregation", help="Many trial agregation type: avg, concat. Default: concat", default='concat')
    args = ap.parse_args()

    # Prepare data
    all_probas = []
    all_paradigm_names = []
    for paradigm, time_start in list(zip(args.prefix, args.time_start)):
        print(paradigm)
        # Get data
        data, paradigm_name = get_data(prefix=paradigm, trials=args.trials, morpho=args.morphology,
                                       molecules=args.molecules)
        data = filter_by_time(data, num_steps=args.num_steps, step_len=args.step_len, time_start=time_start)
        data = agregate_trails(data, agregation=args.agregation)
        data, header = exclude_molecules(data, header=args.molecules,
                                         exact=['time', 'Ca', 'Leak'] + S845_merge.split(' '), wildcard=['out', 'buf'])

    plt.show()
