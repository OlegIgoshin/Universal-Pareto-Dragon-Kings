"""Keep publication figures and numerical checks named in the SI; remove intermediates.

This runs ONLY after a figure has completed successfully. Use --keep-intermediates
on run_all.py when per-trajectory files or intermediate numerical arrays are needed.
"""
from pathlib import Path
import shutil

# Paths are relative to the result directory for each figure.
KEEP = {
    '2': {
        'figure2_new_mc.pdf', 'new_mc_settings.json',
        'new_mc_mean4.csv', 'new_mc_mean10.csv',
    },
    '3': {
        'Figure3_fresh.pdf', 'mc/panelA_summary.csv',
        'C_diagnostics.json',
    },
    '4': {
        'assembly/figures/Figure4_replot.pdf',
        '4AB/verification.json', '4C/verification.json',
        '4D/verification.json', '4EF/verification.json',
        '4D/D_summary.csv', '4EF/EF_summary.csv',
        '4EF/Figure4F_highwealth_partition.csv',
        '4EF/Figure4F_wealth_capture.json',
        '4EF/Figure4F_convergence_checks.csv',
    },
    '5': {
        'figure5.pdf', 'figure5_statistics.json', 'country_statistics.csv',
    },
}


def prune_figure(directory: Path, figure: str, profile: str = 'publication') -> list[str]:
    """Remove unneeded outputs, leaving explicitly retained files only.

    The pruning phase is intentionally separate from the generators. This ensures
    no intermediate output is removed until the plot and supporting checks exist.
    """
    if figure not in KEEP:
        raise ValueError(f'Unknown figure: {figure}')
    keep = set(KEEP[figure])
    if figure == '4' and profile == 'smoke':
        # Short test runs cannot produce the 6-to-12 production checks.
        keep.difference_update({name for name in keep if 'Figure4F_' in name})
    missing = [name for name in sorted(keep) if not (directory / name).is_file()]
    if missing:
        # These are required for the selected output policy; avoid destructive
        # pruning if a generator changed or failed to produce them.
        raise RuntimeError(f'Will not prune {directory}: missing outputs: {missing}')
    deleted = []
    for path in sorted(directory.rglob('*'), reverse=True):
        if path.is_file() and path.relative_to(directory).as_posix() not in keep:
            deleted.append(str(path.relative_to(directory)))
            path.unlink()
    for path in sorted(directory.rglob('*'), key=lambda p: len(p.parts), reverse=True):
        if path.is_dir() and not any(path.iterdir()):
            path.rmdir()
    return deleted
