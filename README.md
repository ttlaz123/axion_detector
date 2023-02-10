# axion_detector
Chao-Lin's concentric cavities

# Description of Scripts (by directory)

## Characterization Analysis

Scripts for performing analysis relevant to the characterization of the cavity (Q, beta, alignment and tuning behaviour, etc.)

### autoalign_hist\_analysis

Functionality is reproduced in polyplotter.py.
Loads in runs of several NM alignments, and plots the distribution of final positions.

### color_map

Functionality is reproduced in polyplotter.py.
Loads in frequency differences recorded when doing a disk-pull measurement of the E field. Plots a color map of the field distribution.

### mapping_formfactor

An attempt to caculate form factor directly from the field maps made with the disk. Consistently overestimates C.
Is the integration truly independent of dx? Assumes each disk data point occupies 1/50th of the volume, since there are 50 points. Also assumes that all field is in z, COMSOL supports this assumption.

### plot_ZfQ

Simple script that plots resonant frequency and Q as a function of shell height Z. The data this script reads is produced by control/z_scan.py.
Should be folded into polyplotter even though it's so simple.

### polyplotter

The main meat of the characterization and plot making. Each function has a docstring commented under it. Access it by reading the code or running help(function) in python.

### stitch_S11s\_into\_Z\_scan

Note: the script control/z_scan semi-automates the process of finding fres and Q at many heights. This script is good if the s11 data was already taken and you need the associated parameters all in one place though.

Takes many S11 files and fits a resonance to each one, then saves the fit parameters of interest (fres and Q) and their errors, as well as the height of the shell where the S11 was taken.

Should be modified to accept the raw s11 data rather than file names, then it could be incorporated into polyplotter and also work on comsol files. The reason it needs filenames now is those encode the Z height. Maybe only a better DB structure can fix that.

Should be upgraded to work on complex data and fit to beta too. Could borrow fitting function from polyplotter.

### tuning_plotter

An old script, an ancient iteration of polyplotter. Has a few unique features but none that are used, including an attempted analytical cable reflection correction. Also uses argparser to be able to plot from the command line. I prefer opening an IDE and importing polyplotter as pp though.

## Control

### analyse

This file contains a lot of helper functions for the old autoalign algorithm, such as peak finding and tracking. Also has an attempt at an fft-based cable reflection filter and a lorentzian magnitude fit.

### automate

The main meat of the control scripts. Many functions for scanning along a wedge axis, tuning with the positioner, and autoaligning. 

Many functions here are long obsolete. This mess must be broached.

### fieldmap

Semi-automated script for collecting disk-based E field data. Requires a human to move the disk and adjust the VNA.

### from_scratch

An attempt at a semi-automated script for aligning the wedge from scratch. Uses the old autoalign, and now I'm not so convinced it can be easily automated. It contains in a big comment at the top some tips and a procedure for aligning by hand. Small misalignments such as from tuning can be handled well with the automatic NM algorithm.

### hexachamber

Functions for communicating with the Newport hexapod.

### manual

Scripts for controlling the hexapod through the CLI. Just use the web interface for that though.

### na_tracer

Oh Tom was this a competitive Overwatch joke?
Scripts for communicating with the VNA over GPIO.

### positioner

Scripts for communicating with the Newport linear positioner.

### z_scan

Semi-automatic script for producing a file of fres and Q at several Z heights. Plotted by plot_ZfQ in characterization\_analysis. Prompts the user to see if the fit was good before saving parameters.

If we ever want to use this again, we should update it to fit complex data (for beta information). We should also have it subtract off cable reflections without the resonance there before fitting (tune, move up a bit, record, move down again, record, subtract the two, fit).

## mynewportxps

Some install files? Ask Tom if we still need these.

## rastor_scanner

Scripts for controlling and looking at data produced by the rastor scanner.
