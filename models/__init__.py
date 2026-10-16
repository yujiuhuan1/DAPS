from .fggp import fggp


def get_all_models():
    return ['fggp']


def get_model(backbones_list, args, transform):
    if args.model != 'fggp':
        raise ValueError('This package contains the DAPS model only.')
    if any(getattr(args, name, 1) != 1
           for name in ('daps_rel', 'daps_drift', 'daps_alpr')):
        raise ValueError('The complete DAPS model requires Rel., Drift, and ALPR.')
    return fggp(backbones_list, args, transform)
