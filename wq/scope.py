"""Which study a station belongs to: a bathing beach, or a shellfish growing area.

The project is sold as a beach product. Its largest analyte is not a beach
measurement. Of 1,692 fitted fecal-coliform pairs outside California, ZERO
come from a station published under the EPA BEACH Act, and the great majority
are fecal-coliform-only estuarine stations with numeric codes run by New
Jersey's Bureau of Marine Water Monitoring. NSSP classifies shellfish growing
areas on fecal coliform; the BEACH Act posts bathing beaches on enterococcus.
Pooling the two and calling the result "FECAL" answers neither question.

This module labels the population so the two can be reported apart. It does
NOT drop anything, and it is NOT a stratum: wq/strata.py may not read a sample
value, because strata have to be assignable before results exist, and one half
of the rule below is a fact about which analytes a station reports. So it
lives here, is written to a committed CSV, and is applied at reporting time.

HOW RELIABLE IS THE LABEL

  beach        SOURCED, two ways. Either the WQP location type begins with
               "BEACH Program Site", which is EPA's own BEACH Act programme
               saying what the station is for; or the station shares a
               site_cluster with one that does, which is co-location at 150 m
               and catches California's CEDEN identifiers describing the same
               sand as a CABEACH_WQX beach station. Both are evidence from
               outside this pipeline. 19 stations enter this way.

  shellfish    INFERRED, and the inference is stated rather than hidden:
               a station that is NOT a BEACH Act station, sits in an estuary
               or on the coast, and reports FECAL COLIFORM. The reasoning is
               the regulatory split -- fecal coliform is the NSSP growing-area
               indicator and enterococcus is the BEACH Act one -- and it is
               supported here by the fact that the two sets do not overlap at
               all. It is still an inference.

               Its error mode: an ambient or river-mouth monitoring station
               that happens to measure fecal coliform for some other reason is
               labelled shellfish. Those exist, and they are why nothing is
               dropped on the strength of this label.

  unclassified Everything else. Mostly USGS river gauges, one tribal EPA
               estuary station, and non-beach stations with no fittable
               analyte at all. Reported, never silently folded into either.
"""

import os

import pandas as pd

from . import config, strata

SCOPE_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                          "study_scope.csv")
NSSP_INDICATOR = "FECAL"


def classify(sites, samples):
    """(frame) with station_id, study_scope and the basis for each label."""
    frame = sites.copy()
    frame["station_id"] = frame["station_id"].astype(str)
    frame["programme"] = [strata.programme_of(t)
                          for t in frame.get("site_type", "")]

    beach_clusters = set(frame.loc[frame["programme"] == "beach_act",
                                   "site_cluster"].astype(str))
    counts = (samples.assign(station_id=samples["station_id"].astype(str))
              .groupby(["station_id", "analyte"]).size())
    fittable = counts[counts >= config.MIN_SAMPLES_PER_SITE].reset_index()
    reports_indicator = set(
        fittable.loc[fittable["analyte"] == NSSP_INDICATOR, "station_id"])

    scopes, bases = [], []
    for row in frame.itertuples():
        if row.programme == "beach_act":
            scopes.append("beach")
            bases.append("SOURCED: EPA BEACH Act location type")
        elif str(row.site_cluster) in beach_clusters:
            scopes.append("beach")
            bases.append("SOURCED: shares a 150 m site_cluster with a "
                         "BEACH Act station")
        elif row.station_id in reports_indicator:
            scopes.append("shellfish")
            bases.append("INFERRED: not a BEACH Act station and reports "
                         "fecal coliform, the NSSP growing-area indicator")
        else:
            scopes.append("unclassified")
            bases.append("neither: no BEACH Act type, no beach neighbour, "
                         "no fittable fecal coliform")
    frame["study_scope"] = scopes
    frame["scope_basis"] = bases
    return frame[["station_id", "study_scope", "scope_basis", "programme",
                  "site_type", "organization", "state", "site_cluster"]]


def read(path=None):
    path = path or SCOPE_PATH
    if not os.path.exists(path):
        return pd.DataFrame(columns=["station_id", "study_scope"])
    return pd.read_csv(path, dtype={"station_id": str})


def stations_in(scope, path=None):
    frame = read(path)
    if frame.empty:
        return set()
    return set(frame.loc[frame["study_scope"] == scope, "station_id"])


def restrict(frame, scope, path=None):
    """Rows whose station is in `scope`. Nothing is dropped from disk."""
    if scope in (None, "all") or "station_id" not in frame.columns:
        return frame
    keep = stations_in(scope, path)
    return frame[frame["station_id"].astype(str).isin(keep)]


def main():
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--write", action="store_true")
    args = parser.parse_args()
    sites = pd.read_csv(os.path.join(config.DATA_DIR,
                                     "stations_stratified.csv"),
                        low_memory=False)
    samples = pd.read_csv(os.path.join(config.DATA_DIR, "samples_clean.csv"),
                          low_memory=False, usecols=["station_id", "analyte"])
    table = classify(sites, samples)
    if args.write:
        table.to_csv(SCOPE_PATH, index=False)
        print(f"wrote {SCOPE_PATH}  ({len(table)} stations)")
    print(table.groupby(["study_scope", "scope_basis"]).size()
          .rename("stations").to_string())


if __name__ == "__main__":
    main()
