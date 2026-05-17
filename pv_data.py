import argparse
from pathlib import Path

import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).parent

# New Mexico microgrid — Albuquerque area
NM_LAT = 35.0844
NM_LON = -106.6504
NM_TZ = "America/Denver"


def simulate_pv_park(
    capacity_kwp: float,
    lat: float,
    lon: float,
    start_date: str,
    end_date: str,
    tilt: float = 20,
    azimuth: float = 180,
) -> pd.DataFrame:
    try:
        import pvlib
    except ImportError as exc:
        raise SystemExit("pvlib is required. pip install pvlib") from exc

    tz = NM_TZ

    # TMY weather — PVGIS ERA5 covers global locations including the US
    weather, _ = pvlib.iotools.get_pvgis_tmy(lat, lon)
    weather.index = weather.index.tz_convert(tz)

    site = pvlib.location.Location(lat, lon, tz=tz)
    solar_position = site.get_solarposition(weather.index)

    poa = pvlib.irradiance.get_total_irradiance(
        tilt,
        azimuth,
        solar_position["apparent_zenith"],
        solar_position["azimuth"],
        weather["dni"],
        weather["ghi"],
        weather["dhi"],
    )

    temp_cell = pvlib.temperature.sapm_cell(
        poa["poa_global"],
        weather["temp_air"],
        weather["wind_speed"],
        **pvlib.temperature.TEMPERATURE_MODEL_PARAMETERS["sapm"][
            "open_rack_glass_glass"
        ],
    )

    pdc = pvlib.pvsystem.pvwatts_dc(
        poa["poa_global"], temp_cell, pdc0=capacity_kwp, gamma_pdc=-0.004
    )
    pac = pvlib.inverter.pvwatts(pdc, pdc0=capacity_kwp, eta_inv_nom=0.96)

    df = pd.DataFrame({"p_kw": pac})

    # Prepend midnight so the profile starts at 00:00
    midnight = df.index[0].normalize()
    df = pd.concat(
        [pd.DataFrame({"p_kw": [df["p_kw"].iloc[0]]}, index=[midnight]), df]
    )

    df_15min = df.resample("15min").interpolate()

    start = pd.Timestamp(start_date, tz=tz)
    end = pd.Timestamp(end_date, tz=tz) - pd.Timedelta(minutes=15)
    desired_index = pd.date_range(start=start, end=end, freq="15min")
    repeats = int(len(desired_index) / len(df_15min)) + 2
    extended = pd.concat([df_15min] * repeats)
    extended.index = pd.date_range(
        start=start, periods=len(extended), freq="15min", tz=tz
    )
    result = extended.loc[start:end]
    result.index.name = "timestamp"
    return result


def build_pv_profile(
    capacity_kwp: float,
    lat: float,
    lon: float,
    start_date: str,
    end_date: str,
    tilt: float,
    azimuth: float,
) -> pd.DataFrame:
    df = simulate_pv_park(
        capacity_kwp, lat, lon, start_date, end_date, tilt=tilt, azimuth=azimuth
    )
    df.index = df.index.tz_localize(None)
    return df


def build_load_profile(start_date: str, end_date: str) -> pd.DataFrame:
    timestamps = pd.date_range(start_date, end_date, freq="15min", inclusive="left")
    hours = timestamps.hour.to_numpy() + timestamps.minute.to_numpy() / 60
    rng = np.random.default_rng(seed=0)
    load = (
        200
        + 100 * np.exp(-((hours - 8) ** 2) / 2)
        + 80 * np.exp(-((hours - 18) ** 2) / 2)
    ) * (1 + rng.normal(0, 0.05, len(timestamps)))
    return pd.DataFrame({"timestamp": timestamps, "load_kw": load.clip(min=0)})


def build_tou_tariffs(start_date: str, end_date: str) -> pd.DataFrame:
    timestamps = pd.date_range(start_date, end_date, freq="15min", inclusive="left")
    hours = timestamps.hour
    tou = np.where(
        (hours >= 22) | (hours < 6),
        0.05,
        np.where((hours >= 16) & (hours < 22), 0.40, 0.15),
    )
    return pd.DataFrame({"timestamp": timestamps, "tou_usd_kwh": tou})


def build_grid_availability(
    start_date: str, end_date: str, outage_prob: float = 0.005
) -> pd.DataFrame:
    timestamps = pd.date_range(start_date, end_date, freq="15min", inclusive="left")
    rng = np.random.default_rng(seed=42)
    available = (rng.random(len(timestamps)) > outage_prob).astype(int)
    return pd.DataFrame({"timestamp": timestamps, "grid_available": available})


def build_all_data(
    df_pv: pd.DataFrame,
    df_load: pd.DataFrame,
    df_tou: pd.DataFrame,
    df_grid: pd.DataFrame,
    output_dir: Path,
) -> None:
    df = df_pv.reset_index().merge(df_load, on="timestamp")
    df = df.merge(df_tou, on="timestamp")
    df = df.merge(df_grid, on="timestamp")
    df.to_csv(output_dir / "all_data.csv", index=False)

    total_kwh = (df["p_kw"] * 0.25).sum()
    outage_slots = (df["grid_available"] == 0).sum()
    print(f"Total PV yield:    {total_kwh:.1f} kWh")
    print(f"Outage slots:      {outage_slots} ({outage_slots * 0.25:.1f} h)")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Prepare all data artifacts for the NM microgrid dispatch problem."
    )
    parser.add_argument("--capacity-kwp", type=float, default=500)
    parser.add_argument("--lat", type=float, default=NM_LAT)
    parser.add_argument("--lon", type=float, default=NM_LON)
    parser.add_argument("--start-date", type=str, default="2025-06-01")
    parser.add_argument("--end-date", type=str, default="2025-07-01",
                        help="Exclusive end date (last slot is end_date - 15min)")
    parser.add_argument("--tilt", type=float, default=20)
    parser.add_argument("--azimuth", type=float, default=180)
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args()

    output_dir = args.output_dir or BASE_DIR

    pv = build_pv_profile(
        capacity_kwp=args.capacity_kwp,
        lat=args.lat,
        lon=args.lon,
        start_date=args.start_date,
        end_date=args.end_date,
        tilt=args.tilt,
        azimuth=args.azimuth,
    )
    load = build_load_profile(args.start_date, args.end_date)
    tou = build_tou_tariffs(args.start_date, args.end_date)
    grid = build_grid_availability(args.start_date, args.end_date)
    build_all_data(pv, load, tou, grid, output_dir)


if __name__ == "__main__":
    main()
