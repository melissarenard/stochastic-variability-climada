# Standard library
import os
import datetime as dt
from typing import Optional
from time import time
from collections import Counter
from bisect import bisect_right

# Third-party libraries
import numpy as np
import numpy.typing as npt
import pandas as pd
import scipy.sparse as sp
import geopandas as gpd

from scipy import integrate
from scipy import optimize
from tqdm import trange, tqdm

# Domain-specific libraries
from climada.hazard import TropCyclone
from climada.entity.exposures import LitPop
from climada.engine import ImpactCalc
from climada.entity.impact_funcs import ImpactFuncSet

ALL_BASINS = ['EP', 'NA', 'NI', 'SI', 'SP', 'WP']
BASIN_TO_IDX = {b: i for i, b in enumerate(ALL_BASINS)}

ENSO_PHASES = ['Nina', 'Neutral', 'Nino']
ENSO_TO_IDX = {p: i for i, p in enumerate(ENSO_PHASES)}


class JaggedArray:
    """
    In the following, we often want to store information about each
    cyclone event, and have it accessible by simulation number, year, and basin.
    However, the number of cyclones by sim/year/basin is not constant (it is random).
    This class is a simple wrapper around a contiguous preallocated array
    so we can store a jagged array of data and access it by a multi-index. 
    """
    def __init__(self, lengths, dtype):
        self.shape = lengths.shape             # e.g. (n_sim, n_years, n_basins)
        flat_lengths = np.ascontiguousarray(lengths, int).ravel()
        self.starts  = np.concatenate(([0], flat_lengths.cumsum()))[:-1]
        self.lengths = flat_lengths            # 1D array of the same total size
        self.data    = np.empty(flat_lengths.sum(), dtype)

    def __getitem__(self, idx):
        i = np.ravel_multi_index(idx, self.shape, order='C')
        s, l = self.starts[i], self.lengths[i]
        return self.data[s:s+l]

    def __setitem__(self, idx, v):
        i = np.ravel_multi_index(idx, self.shape, order='C')
        s, l = self.starts[i], self.lengths[i]
        assert l == len(v)
        self.data[s:s+l] = v


def date_to_enso(date: dt.datetime, enso_dict: dict[dt.datetime, str]) -> str:
    """
    Given a date, return the corresponding ENSO phase (Nina, Neutral, or Nino)
    based on the most recent ENSO phase start date in the given dictionary.

    Parameters
    ----------
    date
        Date to lookup. Should fall within the range of enso_dict keys.
    enso_dict
        Maps datetime keys (phase start dates) to ENSO phase strings.

    Returns
    -------
    str
        ENSO phase for the given date.

    Raises
    ------
    ValueError if the date is outside the range of known ENSO periods.
    """
    keys_sorted = sorted(enso_dict.keys())
    index = bisect_right(keys_sorted, date)

    if index == 0 or date >= keys_sorted[-1] + dt.timedelta(days=366):
        raise ValueError(f"Date {date} is out of range of ENSO phase definitions.")

    return enso_dict[keys_sorted[index - 1]]


def enso_phases_of_historical_tracks(haz: TropCyclone) -> npt.NDArray[np.str_]:
    """
    Get ENSO phase (Nina, Neutral, Nino) for each cyclone event based on
    the date of its first record and basin.

    Parameters
    ----------
    haz
        The hazard object containing cyclone metadata.

    Returns
    -------
    npt.NDArray[object]
        Array of ENSO phases (as strings) for each event in `haz`.
    """
    enso_df = pd.read_csv('Data/ENSO.csv', skiprows=3)
    enso_df = enso_df._append({'year': 2024, 'enso': 'Nino'}, ignore_index=True)

    # Build ENSO phase start dates for NH and SH
    enso_dict_NH = {
        dt.datetime(year, 1, 1): phase
        for year, phase in zip(enso_df['year'], enso_df['enso'])
    }
    enso_dict_SH = {
        dt.datetime(year - 1, 8, 1): phase
        for year, phase in zip(enso_df['year'], enso_df['enso'])
    }

    # Which dictionary to use for each basin
    basin_to_dict = {
        'EP': enso_dict_NH, 'NA': enso_dict_NH, 'NI': enso_dict_NH, 'WP': enso_dict_NH,
        'SI': enso_dict_SH, 'SP': enso_dict_SH
    }

    haz_enso = np.empty(haz.size, dtype=object)

    for i in range(haz.size):
        basin = haz.basin[i]
        date = dt.datetime.fromordinal(haz.date[i])
        enso_dict = basin_to_dict[basin]
        haz_enso[i] = date_to_enso(date, enso_dict)

    return haz_enso


def get_intensity_adjustment_factor(
    basin: str,
    enso_actual: str,
    enso_simulated: str,
    max_intensity: npt.NDArray,
) -> float:
    """
    Return the multiplicative adjustment factor to apply to cyclone intensity
    when simulating it under a different ENSO phase.

    Parameters
    ----------
    basin
        The basin in which the cyclone occurred.
    enso_actual
        The ENSO phase during which the cyclone originally occurred.
    enso_simulated
        The ENSO phase being simulated.
    max_intensity
        Maximum intensity of cyclones in each basin, by ENSO phase.
        Shape: (3, n_basins)

    Returns
    -------
    float
        Multiplicative intensity adjustment factor.
    """
    actual_phase_num = ENSO_TO_IDX[enso_actual]
    sim_phase_num = ENSO_TO_IDX[enso_simulated]
    return max_intensity[sim_phase_num, BASIN_TO_IDX[basin]] / max_intensity[actual_phase_num, BASIN_TO_IDX[basin]]


def compute_loss_catalogue(
    haz: TropCyclone,
    exp: LitPop,
    impfset: ImpactFuncSet,
    output_dir="Outputs",
    filename_base="loss_catalogue",
    save_catalogue=False,
) -> dict[str, sp.csr_matrix]:
    """
    Compute loss catalogues for all ENSO phases, with and without intensity adjustment.

    Returns
    -------
    dict[str, sp.csr_matrix]
        Dictionary mapping ENSO phase to sparse loss matrix of shape n_events x n_locations.
    """
    os.makedirs(output_dir, exist_ok=True)

    loss_catalogues = {}

    # Average maximum TC intensity by ENSO phase, split by basin
    max_intensity_df = pd.read_csv(
        "R tables/severity_enso.csv"
    )
    assert all(max_intensity_df.columns == ALL_BASINS), "Invalid column names in severity_enso.csv"
    max_intensity = np.array(max_intensity_df.values)

    # --- No intensity adjustment ---
    print("Computing loss catalogue: No adjustment", flush=True)
    loss_no_adjust = ImpactCalc(exp, impfset, haz).impact(assign_centroids=False).imp_mat
    loss_catalogues["No adjustment"] = loss_no_adjust
    if save_catalogue:
        sp.save_npz(os.path.join(output_dir, f"{filename_base}_no_adjustment.npz"), loss_no_adjust)

    # --- ENSO-phase-specific adjustments ---
    original_enso = enso_phases_of_historical_tracks(haz)
    original_intensities = haz.intensity.copy()

    for sim_enso in ["Nina", "Neutral", "Nino"]:
        print(f"Computing loss catalogue: {sim_enso}", flush=True)

        # Scale the intensity of each cyclone based on a given simulated ENSO phase
        adjustment_factors = np.ones(haz.size)

        for i in range(haz.size):
            basin = haz.basin[i]
            actual_enso = original_enso[i]
            adjustment_factors[i] = get_intensity_adjustment_factor(basin, actual_enso, sim_enso, max_intensity)

        haz.intensity = sp.diags(adjustment_factors) @ original_intensities

        # Compute and store
        loss_mat = ImpactCalc(exp, impfset, haz).impact(assign_centroids=False).imp_mat
        loss_catalogues[sim_enso] = loss_mat
        if save_catalogue:
            sp.save_npz(os.path.join(output_dir, f"{filename_base}_{sim_enso}.npz"), loss_mat)

    haz.intensity = original_intensities

    return loss_catalogues

def previous_period_observed_losses(
    haz: TropCyclone,
    synthetic_start_year: int,
    loss_catalogue: sp.csr_matrix,
    ) -> tuple[sp.csr_matrix, npt.NDArray[np.float64]]:
    """
    Compute the loss catalogue for the previous 6 months, using the historical hazard data.
    This is used to for the consecutive loss adjustment during the first synthetic year.
    """
    start = dt.date.toordinal(dt.date(synthetic_start_year-1, 2, 1))
    end = dt.date.toordinal(dt.date(synthetic_start_year-1, 8, 1))
    end_1yr_bf = dt.date.toordinal(dt.date(synthetic_start_year-2, 8, 1))
    mask = (start <= haz.date) & (haz.date < end) & haz.orig
    previous_losses = loss_catalogue[mask, :].copy().tocsr()

    # Also get the dates of the previous 6 months events, and convert them.
    # Make Aug 1 of the previous year become -1 and Aug 1 of this year be 0.
    previous_times = (haz.date[mask] - end_1yr_bf) / (end - end_1yr_bf)  # Normalize to [0, 1) --- will be in [0.5, 1)
    previous_times -= 1  # Shift to [-1, 0) --- will be in [-0.5, 0)

    # Sort the losses and times by the event time
    order = np.argsort(previous_times)
    previous_losses = previous_losses[order, :]
    previous_times = previous_times[order]

    return previous_losses, previous_times

def simulate_enso_time_series(
    n_sim: int,
    n_years: int,
    starting_enso: str,
    show_progress=False
) -> npt.NDArray[np.uint8]:
    """
    Simulate ENSO phase time series for all simulations using a Markov chain.

    Parameters
    ----------
    n_sim
        Number of simulations to run.
    n_years
        Number of years per simulation.
    starting_enso
        ENSO phase to start the chain from ('Nina', 'Neutral', 'Nino').
    

    Returns
    -------
    npt.NDArray[np.unint8]
        ENSO phases per [simulation][year], as integers (0, 1, 2).
    """
    # 3x3 ENSO Markov transition matrix.
    matrix = pd.read_csv('R tables/enso_MC.csv', index_col = 0).values

    simulations = np.empty((n_sim, n_years), dtype=np.uint8)
    start_val = ENSO_TO_IDX[starting_enso]

    for sim in trange(n_sim, desc="Simulating ENSO time series", disable=not show_progress):
        prev_phase = start_val
        for year in range(n_years):
            next_phase = np.random.choice([0, 1, 2], p=matrix[prev_phase])
            simulations[sim, year] = next_phase
            prev_phase = next_phase

    return simulations


def simulate_number_of_cyclones(
    n_sim: int,
    n_years: int,
    poisson_params: npt.NDArray[np.float64],
    basins: list[str],
    enso_simulations: npt.NDArray[np.uint8],
    show_progress=False,
) -> npt.NDArray[np.uint8]:
    """
    Simulate the number of tropical cyclones per basin and year.

    Parameters
    ----------
    n_sim
        Number of simulations.
    n_years
        Number of years per simulation.
    poisson_params
        MMNHPP parameters of shape (len(basins), 6).
    basins
        list of basin names.
    enso_simulations
        ENSO phase per [simulation][year], as integers.

    Returns
    -------
    npt.NDArray[np.uint8]
        Cyclone counts of shape (n_sim, n_years, n_basins)
    """

    mean_yearly_arrivals = expected_poisson_arrivals(poisson_params, basins)

    counts = np.zeros((n_sim, n_years, len(basins)), dtype=np.uint8)

    for sim_idx in trange(n_sim, desc="Simulating number of cyclones", disable=not show_progress):
        for year_idx in range(n_years):
            enso_idx = enso_simulations[sim_idx, year_idx]
            for b_idx, _ in enumerate(basins):
                counts[sim_idx, year_idx, b_idx] = np.random.poisson(mean_yearly_arrivals[b_idx, enso_idx])

    return counts

def enso_probs(enso_idx: int, p: float) -> npt.NDArray[np.float64]:
    """
    Return a probability vector for drawing events from ENSO phases,
    biased toward the given ENSO phase.

    Parameters
    ----------
    enso_idx
        Index of the simulated ENSO phase (0: Nina, 1: Neutral, 2: Nino).
    p
        Probability of selecting an event from the current ENSO phase.

    Returns
    -------
    npt.NDArray[np.float64]
        Array of probabilities [p_nina, p_neutral, p_nino], summing to 1.
    """
    if enso_idx not in (0, 1, 2):
        raise ValueError("Invalid ENSO index. Must be 0 (Nina), 1 (Neutral), or 2 (Nino).")

    q = (1 - p) / 2
    probs = np.full(3, q)
    probs[enso_idx] = p
    return probs


def sample_synthetic_cyclones(
    n_sim: int,
    basins: list[str],
    haz_basin: npt.NDArray[np.str_],
    haz_enso: npt.NDArray[np.str_],
    enso_simulations: npt.NDArray[np.uint8],        # shape (n_sim, n_years)
    event_numbers: npt.NDArray[np.uint8],           # shape (n_sim, n_years, n_basins)
    loc: bool,
    p_loc: float,
    replace=True,
    show_progress=False,
) -> JaggedArray:
    """
    Sample synthetic cyclone events per simulation/year/basin with or without replacement.

    Parameters
    ----------
    basins
        Names of the cyclone basins (ordered).
    haz_basin
        Basin name for each historical event.
    haz_enso
        ENSO phase for each historical event (e.g. "Nina", "Neutral", "Nino").
    
    Returns
    -------
    JaggedArray[np.uint32]
        Array of all sampled cyclone indices, indexed by (sim, year, basin)
    """
    n_years = enso_simulations.shape[1]
    n_hazs = haz_basin.size

    cyclone_inds = JaggedArray(event_numbers, dtype=np.uint32)

    # Pre-index the historical event data
    event_inds = np.arange(n_hazs)
    event_inds_by_basin = {b: event_inds[haz_basin == b] for b in basins}
    enso_by_basin = {b: haz_enso[haz_basin == b] for b in basins}

    for sim_idx in trange(n_sim, desc="Sampling synthetic cyclone indices", disable=not show_progress):
        used_inds : Optional[set[int]] = set() if not replace else None

        for year_idx in range(n_years):
            enso_idx = enso_simulations[sim_idx, year_idx]
            selection_probs = enso_probs(enso_idx, p_loc) if loc else None

            for b_idx, basin in enumerate(basins):
                n_events = event_numbers[sim_idx, year_idx, b_idx]

                all_inds_in_basin = event_inds_by_basin[basin]

                if loc:
                    enso_labels = enso_by_basin[basin]
                    sampled_phases = np.random.choice(ENSO_PHASES, size=n_events, p=selection_probs)
                    counts = Counter(sampled_phases)

                    selected_inds : list[int] = []
                    for phase, count in counts.items():
                        available = all_inds_in_basin[enso_labels == phase]
                        if not replace:
                            available = np.setdiff1d(available, list(used_inds), assume_unique=True)
                        
                        if len(available) < count:
                            raise ValueError(f"Not enough events in basin '{basin}' for ENSO phase '{phase}'.")
                        selected = np.random.choice(available, size=count, replace=replace)
                        selected_inds.extend(selected)
                        if not replace:
                            used_inds.update(selected)
                else:
                    if replace:
                        available = all_inds_in_basin
                    else:
                        available = np.setdiff1d(all_inds_in_basin, list(used_inds), assume_unique=True)
                        
                    if len(available) < n_events:
                        raise ValueError(f"Not enough events in basin '{basin}' to sample without replacement.")
                    selected_inds = np.random.choice(available, size=n_events, replace=replace)
                    if not replace:
                        used_inds.update(selected_inds)

                cyclone_inds[sim_idx, year_idx, b_idx] = selected_inds

    return cyclone_inds


def get_poisson_parameters(haz: TropCyclone, basins: list[str], homogeneous: bool) -> npt.NDArray[np.float64]:
    """
    Load MMNHPP parameters and return as array of shape (len(basins), 6).
    """
    params = np.zeros((len(basins), 6)) # (basin, params)
   
    if homogeneous:
        haz_basin = np.array(haz.basin)
        for b, basin in enumerate(basins):
            lambda_for_basin = haz.frequency[haz_basin == basin].sum()
            params[b, 0] = lambda_for_basin   # store λ in column 0
        return params
    
    par_df = pd.read_csv("R tables/LuGarrido2006params.csv")

    for b, basin in enumerate(basins):
        params[b, :] = par_df[basin].values

    # However, not all cyclones hit Australia, so we scale the parameters
    # by the proportion of cyclones that hit Australia.
    aus_cyclone_basins = np.array(haz.basin)[haz.orig]
    n_tracks_CLIMADA = pd.read_csv('R tables/n_tracks_CLIMADA.csv')

    for b, basin in enumerate(basins):
        n_hist_tracks = np.sum(aus_cyclone_basins == basin)
        total_tracks = n_tracks_CLIMADA[basin].values[0]
        scale = n_hist_tracks / total_tracks
    
        params[b, 0] *= scale
        params[b, 1] *= scale
        params[b, 2] *= scale

    return params


def nonhomogeneous_poisson_intensity(t: float, coefs: npt.NDArray[np.float64], enso_phase: int, south_hemis=False) -> float:
    """
    Evaluate intensity function lambda(t) from seasonal model.

    Applies hemisphere-based time shift for Southern Hemisphere basins.

    Parameters
    ----------
    t
        Time in [0, 1) to evaluate lambda(t) at.
    coefs
        Coefficients of the MMNHPP intensity function.
    south_hemis
        Whether to apply the August shift for Southern Hemisphere basins.

    Returns
    -------
    float
        Value of lambda(t)
    """
    if (coefs[1] == 0 and coefs[2] == 0 and coefs[3] == 0 and coefs[4] == 0 and coefs[5] == 0):
        return coefs[0]
    
    if south_hemis:
        t += 7 / 12
    risk_level = coefs[enso_phase]
    p = coefs[3]
    q = coefs[4]
    shift_time = coefs[5]/12 - 0.5 / 12
    t_adjusted = t - shift_time
    if t_adjusted < 0:
        t_adjusted += 1
    elif t_adjusted > 1:
        t_adjusted -= 1

    j = (p - 1)/(p + q - 2)
    alpha = 1/(j**(p - 1) * (1 - j)**(q - 1))
    
    lambda_ = risk_level * alpha * t_adjusted**(p - 1) * (1 - t_adjusted)**(q - 1)
    return lambda_


def expected_poisson_arrivals(poisson_params: npt.NDArray[np.float64], basins: list[str]) -> npt.NDArray[np.float64]:
    """
    Precompute expected arrival rates per basin and ENSO phase.

    Parameters
    ----------
    poisson_params
        MMNHPP parameters of shape (len(basins), 6).
    basins
        list of basin names.

    Returns
    -------
    npt.NDArray[np.float64]
        Expected arrival rates of shape (len(basins), 3).
    """
    mean_yearly_arrivals = np.zeros((len(basins), len(ENSO_PHASES)))
    for b_idx, basin in enumerate(basins):
        south_hemis = basin in ['SI', 'SP']
        for enso_idx, _ in enumerate(ENSO_PHASES):
            coefs = poisson_params[b_idx]
            lam = integrate.quad(lambda t: nonhomogeneous_poisson_intensity(t, coefs, enso_idx, south_hemis), 0, 1)[0]
            mean_yearly_arrivals[b_idx, enso_idx] = lam
    return mean_yearly_arrivals


def maximum_poisson_intensities(params: npt.NDArray[np.float64], basins: list[str]) -> npt.NDArray[np.float64]:
    """
    Compute maximum of lambda(t) over [0, 1] for each (basin, enso).

    Parameters
    ----------
    params
        MMNHPP parameters of shape (len(basins), 3, 4)

    Returns
    -------
    npt.NDArray[np.float64]
        Maximum lambda values of shape (len(basins), 3)
    """
    n_basins = len(basins)
    max_intensities = np.zeros((n_basins, 3))

    for b_idx, basin in enumerate(basins):
        south_hemis = basin in ['SI', 'SP']
        for e_idx in range(3):
            coefs = params[b_idx]
            result = optimize.minimize_scalar(
                lambda t: -nonhomogeneous_poisson_intensity(t, coefs, e_idx, south_hemis),
                bounds=(0, 1),
                method='bounded'
            )
            max_intensities[b_idx, e_idx] = -result.fun

    return max_intensities


def sample_poisson_arrivals(
    n_events: int,
    coefs: npt.NDArray[np.float64],
    south_hemis: bool,
    max_intensity: float,
    enso_idx: int
) -> npt.NDArray[np.float32]:
    """
    Sample an arrival time from an MMNHPP via rejection sampling.

    Parameters
    ----------
    n_events
        Number of events to sample.
    coefs
        Coefficients of the MMNHPP intensity function.
    south_hemis
        Whether to apply the August shift for Southern Hemisphere basins.
    max_intensity
        Maximum intensity of the MMNHPP function.
    Returns
    -------
    npt.NDArray[np.float32]
        Array of sampled arrival times in [0, 1).
    """
    samples = np.empty(n_events, dtype=np.float32)

    for i in range(n_events):
        while True:
            x = np.random.uniform(0, 1)
            y = np.random.uniform(0, max_intensity)

            if y < nonhomogeneous_poisson_intensity(x, coefs, enso_idx, south_hemis):
                samples[i] = x
                break
    return samples


def simulate_arrival_times_of_cyclones(
    n_sim: int,
    n_years: int,
    poisson_params: npt.NDArray[np.float64],    # shape (len(basins), 3, 4)
    basins: list[str],
    enso_simulations: npt.NDArray[np.uint8],  # shape (n_sim, n_years), int8
    event_numbers: npt.NDArray,     # shape (n_sim, n_years, n_basins), int
    show_progress=False,
) -> JaggedArray:
    """
    Simulate within-season arrival times for tropical cyclones and store in a flat array.

    Parameters
    ----------
    n_sim
        Number of simulations.
    n_years
        Number of years per simulation.
    poisson_params
        MMNHPP parameters of shape (len(basins), 6).
    basins
        list of basin names.
    enso_simulations
        ENSO phase per [simulation][year].
    event_numbers
        Cyclone counts per [simulation][year][basin].
    
    Returns
    -------
    JaggedArray[float32]
        Array of all arrival times, indexed by (sim, year, basin)
    """
    arrival_times = JaggedArray(event_numbers, dtype=np.float32)
    
    max_intensities = maximum_poisson_intensities(poisson_params, basins)

    for sim_idx in trange(n_sim, desc="Simulating arrival times", disable=not show_progress):
        for year_idx in range(n_years):
            enso_idx = enso_simulations[sim_idx, year_idx]

            for basin_idx, basin in enumerate(basins):
                n_events = event_numbers[sim_idx, year_idx, basin_idx]
                coefs = poisson_params[basin_idx]
                south_hemis = basin in ['SI', 'SP']
                times = year_idx + sample_poisson_arrivals(n_events, coefs, south_hemis, max_intensities[basin_idx, enso_idx], enso_idx)
                times.sort()

                arrival_times[sim_idx, year_idx, basin_idx] = times.astype(np.float32)

    return arrival_times


def create_year_loss_table_for_batch(
    sim_indices: list[int],
    enso_simulations: npt.NDArray[np.uint8],              # shape (n_sim, n_years)
    arrival_times: JaggedArray,               # shape (n_sim, n_years, n_basins)
    cyclone_inds: JaggedArray,                 # shape (n_sim, n_years, n_basins)
    basins: list[str],
    synthetic_start_year: int,
) -> pd.DataFrame:
    """
    Create a tidy DataFrame for a batch of simulation indices.

    Parameters
    ----------
    sim_indices
        Indices of simulations to include in this batch.
    enso_simulations
        ENSO phases for all simulations.
    arrival_times
        The arrival time (np.float32) for each resampled cyclone.
    cyclone_inds
        The indices (np.uint32) of the resampled cyclones.
    basins
        Ordered list of basin names.

    Returns
    -------
    pd.DataFrame
        Tidy year loss table (losses yet to be added) for the selected simulations.
    """
    records = []
    n_years = enso_simulations.shape[1]

    for sim_idx in sim_indices:
        for year_idx in range(n_years):
            enso_phase = ENSO_PHASES[enso_simulations[sim_idx, year_idx]]

            for basin_idx, basin in enumerate(basins):
                times = arrival_times[sim_idx, year_idx, basin_idx]
                inds = cyclone_inds[sim_idx, year_idx, basin_idx]

                assert len(times) == len(inds), "Mismatch in arrival time and event index counts"

                for time_i, ind_i in zip(times, inds):
                    # Convert the time_i to a proper date
                    year = year_idx + synthetic_start_year
                    date_i = dt.date(synthetic_start_year-1, 8, 1) + dt.timedelta(days=int(time_i * 365.25))  # Approximate leap years

                    record = {
                        "simulation": sim_idx + 1,
                        "year": year_idx + 1,
                        "event_time": time_i,
                        "event_date": date_i,
                        "basin": basin,
                        "event_ind": ind_i,
                        "enso_phase": enso_phase,
                    }
                    records.append(record)

    df = pd.DataFrame.from_records(records)
    return df.sort_values(["simulation", "year", "event_time"]).reset_index(drop=True)

def summarise_large_losses(matrix: sp.csr_matrix, sigfigs: int = 4, min_loss = 10_000) -> list[str]:
    """
    Summarise larger entries in a CSR matrix as strings of the form:
    "loc_5:12.34,loc_78:56.78"
    
    Parameters:
    - matrix: scipy.sparse.csr_matrix (rows = events, cols = loc_* values)
    - col_offset: integer to add to column indices (e.g., if loc_0 is col 0, offset=0)
    - precision: number of decimal places to keep

    Returns:
    - list of summary strings, one per row
    """
    row_summaries = []
    fmt_str = f"{{:.{sigfigs}g}}"

    for i in range(matrix.shape[0]):
        start, end = matrix.indptr[i], matrix.indptr[i+1]
        indices = matrix.indices[start:end]
        values = matrix.data[start:end]
        summary = "|".join(
            f"{j}:{fmt_str.format(v)}"
            for j, v in zip(indices, values)
            if v >= min_loss
        )
        row_summaries.append(summary)

    return row_summaries

def assign_state_to_exposure(exp_gdf: gpd.GeoDataFrame, state_shapefile_path: str) -> gpd.GeoDataFrame:
    """
    Add a 'state' column (based on STE_NAME21) to a GeoDataFrame of exposure points using spatial join.

    Parameters
    ----------
    exp_gdf
        GeoDataFrame with point geometries and any CRS.
    state_shapefile_path
        Path to ABS 2021 state boundary shapefile.

    Returns
    -------
    gpd.GeoDataFrame
        Copy of the input GeoDataFrame with an added 'STE_NAME21' column indicating state.
    """
    # Load and prepare state polygons
    state_gdf = gpd.read_file(state_shapefile_path).iloc[:-1]  # Drop "Other Territories"
    state_gdf = state_gdf.to_crs("EPSG:7844")
    exp_gdf = exp_gdf.to_crs("EPSG:7844")

    exp_with_state = gpd.sjoin(exp_gdf.to_crs("EPSG:7844"), state_gdf[['STE_NAME21', 'geometry']], how='left', predicate='within')

    # Handle unmatched points using nearest fallback
    missing = exp_with_state[exp_with_state["STE_NAME21"].isna()].copy()
    missing = missing.drop(columns=['index_right', 'STE_NAME21'], errors='ignore')

    if not missing.empty:
        nearest = gpd.sjoin_nearest(
            missing,
            state_gdf[['STE_NAME21', 'geometry']],
            how='left',
            distance_col='distance'
        )
        exp_with_state.update(nearest[['STE_NAME21']])

    return exp_with_state

def save_number_of_cyclones(
    number_of_cyclones: npt.NDArray[np.uint8],
    basins: list[str],
    n_sim: int,
    n_years: int,
    output_dir: str,
    base_filename: str,
) -> None:
    """
    Save the number of simulated cyclones per simulation, year, and basin to a CSV.
    This is mostly for debugging purposes.
    The number of cyclones here will be larger than the number of cyclones in the year loss tables.
    That is because some cyclones cause no losses, so they are not included in the year loss tables.

    Parameters
    ----------
    number_of_cyclones
        Cyclone counts with shape (n_sim, n_years, n_basins).
    basins
        list of basin names (in order corresponding to last axis of number_of_cyclones).
    n_sim
        Number of simulation runs.
    n_years
        Number of years in each simulation.
    output_dir
        Directory to save the CSV file to.
    base_filename
        Base name for the output file (will append "_cyclone_counts.csv").
    """
    n_basins = len(basins)
    simulations = np.repeat(np.arange(1, n_sim + 1), n_years * n_basins)
    years = np.tile(np.repeat(np.arange(1, n_years + 1), n_basins), n_sim)
    basin_column = np.tile(basins, n_sim * n_years)
    cyclone_counts = number_of_cyclones.reshape(-1)

    cyclone_df = pd.DataFrame({
        'simulation': simulations,
        'year': years,
        'basin': basin_column,
        'number_of_cyclones': cyclone_counts
    })

    cyclone_df_wide = cyclone_df.pivot(
        index=['simulation', 'year'],
        columns='basin',
        values='number_of_cyclones'
    ).reset_index()

    cyclone_df_wide.columns = ['simulation', 'year'] + [f"Number of cyclones ({basin} basin)" for basin in basins]
    cyclone_df_wide.to_csv(os.path.join(output_dir, f"{base_filename}_cyclone_counts.csv"), index=False)

def boost_losses_sliding_window(
    batch_df: pd.DataFrame,
    previous_losses: sp.csr_matrix,
    previous_times: npt.NDArray[np.float64],
    losses: sp.csr_matrix,
    region_names: list[str],
    region_col_map: dict[str, np.ndarray],
    region_thresholds: dict[str, float],
    compound_factor: float,
    debug: bool = False,
) -> tuple[sp.csr_matrix, dict[str, np.ndarray]]:
    """
    Computes raw per-region losses internally, then does the sliding-window compounding.
    Returns the boosted loss matrix, plus debug arrays if requested.
    """
    # 1) sort synthetic events
    batch_df = batch_df.reset_index(drop=True)
    order = batch_df.sort_values(['simulation','event_time']).index.to_numpy()
    inv_order = np.empty_like(order); inv_order[order] = np.arange(len(order))
    times = batch_df.loc[order, 'event_time'].to_numpy()
    sims  = batch_df.loc[order, 'simulation'].to_numpy()
    n_synth = len(order)

    # 2) raw per-region losses for synthetic events (in sorted order)
    raw_synth = {}
    for r in region_names:
        cols = region_col_map[r]
        arr = losses[:, cols].sum(axis=1).A1      # shape (n_syn_events,)
        raw_synth[r] = arr[order]                   # now aligned with sorted ‘times’

    # 3) raw per-region losses for prior-year events (they are sorted beforehand)
    prev_raw = {}
    for r in region_names:
        cols = region_col_map[r]
        prev_raw[r] = previous_losses[:, cols].sum(axis=1).A1

    # sort the prior events by their time
    prev_idx = np.argsort(previous_times)
    prev_times_sorted = previous_times[prev_idx]
    for r in region_names:
        prev_raw[r] = prev_raw[r][prev_idx]

    # 4) prepare CSC copy of the *synthetic* loss matrix for in-place boosting
    csc = losses.tocsc(copy=True)

    # 5) allocate sliding‐window state
    window_sums = {r: 0.0 for r in region_names}
    boost_mult  = {r: np.ones(n_synth, dtype=float) for r in region_names}
    prev_vals   = {r: np.zeros(n_synth, dtype=float) for r in region_names}

    # 6) run sliding window, per simulation
    for sim in np.unique(sims):
        idxs      = np.where(sims == sim)[0]
        head_synth  = 0
        head_prev = 0

        # start with *all* prior losses in the window
        for r in region_names:
            window_sums[r] = prev_raw[r].sum()

        for i_local, tail in enumerate(idxs):
            t = times[tail]

            # a) evict any prior events older than t−1
            while head_prev < len(prev_times_sorted) and prev_times_sorted[head_prev] < t - 1.0:
                for r in region_names:
                    window_sums[r] -= prev_raw[r][head_prev]
                head_prev += 1

            # b) evict any *synthetic* events older than t−1
            while head_synth < i_local and times[idxs[head_synth]] < t - 1.0:
                ev = idxs[head_synth]
                for r in region_names:
                    window_sums[r] -= boost_mult[r][ev] * raw_synth[r][head_synth]
                head_synth += 1

            # c) record & decide boost for this event
            for r in region_names:
                prev_vals[r][tail] = window_sums[r]
                if window_sums[r] > region_thresholds[r]:
                    boost_mult[r][tail] = compound_factor

            # d) add this event’s *boosted* loss into the window
            for r in region_names:
                window_sums[r] += boost_mult[r][tail] * raw_synth[r][tail]

    # 7) apply the boosts into your CSC matrix of *synthetic* losses
    for r in region_names:
        mult = boost_mult[r][inv_order]
        for col in region_col_map[r]:
            start, end = csc.indptr[col], csc.indptr[col+1]
            for ptr in range(start, end):
                row = int(csc.indices[ptr])
                csc.data[ptr] *= mult[row]

    # 8) package debug info if requested
    debug_info: dict[str, np.ndarray] = {}
    if debug:
        for r in region_names:
            key = r.lower().replace(' ', '_')
            debug_info[f"{key}_raw_loss"]      = raw_synth[r][inv_order]
            debug_info[f"{key}_prev_win_loss"] = prev_vals[r][inv_order]
            debug_info[f"{key}_boosted"]       = (boost_mult[r][inv_order] != 1.0).astype(int)

    return csc.tocsr(), debug_info

def generate_year_loss_tables(
    haz: TropCyclone,
    loss_catalogues: dict[str, sp.csr_matrix],
    homogeneous: bool,
    n_sim: int,
    n_years: int,
    starting_enso="Nina",
    loc=False,
    p_loc=0.5,
    intensity=False,
    damage=False,
    synthetic_start_year=2024,
    regions: Optional[pd.Series] = None,
    region_thresholds: dict[str, float] = {},
    compound_factor = 1.2,
    batch_size=10_000,
    output_dir="Outputs",
    save_large_losses=True,
    replace=True,
    show_progress=False,
    seed=None,
    save_poisson_debug=False,
    save_damage_debug=False,
) -> str:
    if seed:
        np.random.seed(seed)

    os.makedirs(output_dir, exist_ok=True)
    base_filename = f"{homogeneous}_{n_sim}_{n_years}_{loc}_{p_loc}_{intensity}_{damage}"

    if damage:
        previous_losses, previous_times = previous_period_observed_losses(haz, synthetic_start_year, loss_catalogues["No adjustment"])

    print(f"{base_filename}: 1/5) Simulating ENSO time series", flush=True)
    enso_simulations: npt.NDArray[np.uint8] = simulate_enso_time_series(
        n_sim, n_years, starting_enso, show_progress
    )

    print(f"{base_filename}: 2/5) Simulating number of cyclones", flush=True)
    basins: list[str] = list(np.sort(np.unique(haz.basin)))
    poisson_params = get_poisson_parameters(haz, basins, homogeneous=homogeneous)
    number_of_cyclones: npt.NDArray[np.uint8] = simulate_number_of_cyclones(
        n_sim, n_years, poisson_params, basins, enso_simulations, show_progress
    )

    if save_poisson_debug:
        save_number_of_cyclones(
            number_of_cyclones,
            basins,
            n_sim,
            n_years,
            output_dir,
            base_filename
        )
        
    print(f"{base_filename}: 3/5) Simulating arrival times of cyclones", flush=True)
    arrival_times = simulate_arrival_times_of_cyclones(
        n_sim,
        n_years,
        poisson_params,
        basins,
        enso_simulations,
        number_of_cyclones,
        show_progress
    )

    print(f"{base_filename}: 4/5) Sampling synthetic cyclone indices", flush=True)

    haz_basin = np.array(haz.basin)
    haz_enso: npt.NDArray[np.str_] = enso_phases_of_historical_tracks(haz)

    cyclone_inds = sample_synthetic_cyclones(
        n_sim,
        basins,
        haz_basin,
        haz_enso,
        enso_simulations,
        number_of_cyclones,
        loc,
        p_loc,
        replace=replace,
        show_progress=show_progress
    )

    print(f"{base_filename}: 5/5) Creating year loss table in batches", flush=True)
    start_time = time()

    sim_batches = [list(range(i, min(i + batch_size, n_sim))) for i in range(0, n_sim, batch_size)]
    
    if regions is not None:
        # Map regions to column indices
        region_names = sorted(regions.unique())
        region_col_map = {region: np.where(regions.values == region)[0] for region in region_names}

    for batch_num, sim_indices in enumerate(tqdm(sim_batches, desc=f"{base_filename}: Calculate losses of resampled hazards", disable=not show_progress)):

        batch_year_loss_table = create_year_loss_table_for_batch(
            sim_indices=sim_indices,
            enso_simulations=enso_simulations,
            arrival_times=arrival_times,
            cyclone_inds=cyclone_inds,
            basins=basins,
            synthetic_start_year=synthetic_start_year,
        )

        if batch_year_loss_table.empty:
            continue

        # Compute losses
        if intensity:
            enso_phases = batch_year_loss_table["enso_phase"].values
            losses = []
            for eind, phase in zip(batch_year_loss_table.event_ind, enso_phases):
                losses.append(loss_catalogues[phase].getrow(eind))
            losses = sp.vstack(losses, format="csr")
        else:
            losses = loss_catalogues["No adjustment"][list(batch_year_loss_table.event_ind)]
        
        # Only keep the rows which have a positive loss
        positive = np.array((losses.sum(axis=1) > 0)).ravel()
        batch_year_loss_table = batch_year_loss_table.loc[positive, :]
        losses = losses[positive, :]

        if damage:
            losses, debug_info = boost_losses_sliding_window(
                batch_year_loss_table,
                previous_losses,
                previous_times,
                losses,
                region_names,
                region_col_map,
                region_thresholds,
                compound_factor,
                debug=save_damage_debug,
            )
            if save_damage_debug:
                # dump all debug arrays as new columns
                for col_name, arr in debug_info.items():
                    batch_year_loss_table[col_name] = arr

        if regions is not None:
            # Sum loss by region and insert as new columns
            for region in region_names:
                cols = region_col_map[region]
                region_loss = losses[:, cols].sum(axis=1).A1  # Convert to 1D array
                batch_year_loss_table[f"{region.lower().replace(' ', '_')}_loss"] = region_loss

        batch_year_loss_table["total_loss"] = losses.sum(axis=1)


        if save_large_losses:
            batch_year_loss_table["larger_losses"] = summarise_large_losses(losses)

        # Append batch to CSV
        year_loss_table_file = os.path.join(output_dir, f"{base_filename}.csv")
        write_header = batch_num == 0  # only write header on first batch
        batch_year_loss_table.to_csv(year_loss_table_file, index=False, mode="w" if write_header else "a", header=write_header)

    print(f"{base_filename}: All batches processed in {time() - start_time:.2f} seconds", flush=True)

    # Convert the CSV to parquet
    df = pd.read_csv(year_loss_table_file)
    df['year'] = df['year'].astype(np.uint32)
    df['simulation'] = df['simulation'].astype(np.uint32)
    df['event_time'] = df['event_time'].astype(np.float32)
    df['event_ind'] = df['event_ind'].astype(np.uint32)
    df['total_loss'] = df['total_loss'].astype(np.float32)
    df['enso_phase'] = df['enso_phase'].astype("category")
    df['basin'] = df['basin'].astype("category")
    df.to_parquet(year_loss_table_file.replace('.csv', '.parquet'), compression="zstd", index=False)
    os.remove(year_loss_table_file)
    print(f"{base_filename}: Year loss table saved as parquet", flush=True)

    return base_filename
