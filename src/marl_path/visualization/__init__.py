"""Visualization package for plotting the statistics of training runs. Can be
used to show plots directly via using the --stats-file argument. Otherwise, plots
are shown via a Streamlit app."""

from .cli import main


__all__ = ["main"]
