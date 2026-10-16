from argparse import ArgumentParser


def add_management_args(parser: ArgumentParser) -> None:
    parser.add_argument('--csv_log', action='store_true',
                        help='Enable csv logging',default=False)
