import re
import pandas as pd

# Electrode names (RH, RSS, etc.)
ELECTRODE_RE = re.compile(r'^[A-Za-z]{2,4}$')

# Contact lines
CONTACT_RE = re.compile(
    r'^\s*(\d+):\s*'
    r'(Gray|White|Mixed|Unknown)\s*'
    r'\(Gray:\s*(\d+)%,\s*White:\s*(\d+)%,\s*Unk:\s*(\d+)%\)\s*;'
)

# Atlas labels following the semicolon
LABEL_RE = re.compile(r'([A-Za-z0-9_\-]+)\s*\((\d+)%\)')


def parse_atlas_distribution(label_string):
    """
    Return all anatomical labels and their percentages.

    Removes Wm and Unk because these are tissue classes
    rather than anatomical regions.
    """
    atlas_dist = {}

    for label, pct in LABEL_RE.findall(label_string):
        if label not in {"Wm", "Unk"}:
            atlas_dist[label] = int(pct)

    return atlas_dist


def get_primary_atlas(atlas_dist):
    """
    Return the highest-percentage anatomical label.
    """
    if not atlas_dist:
        return None

    return max(atlas_dist, key=atlas_dist.get)


def parse_anatomical_labels(txt_file):
    """
    Parse an sEEG Anatomical_Labels.txt file.

    Returns
    -------
    pandas.DataFrame
    """
    rows = []
    current_electrode = None

    with open(txt_file, "r", encoding="utf-8", errors="ignore") as f:

        for raw_line in f:

            line = raw_line.strip()

            if not line:
                continue

            # Electrode header
            if ELECTRODE_RE.fullmatch(line):
                current_electrode = line
                continue

            match = CONTACT_RE.match(raw_line)

            if not match or current_electrode is None:
                continue

            (
                contact_num,
                classification,
                gray_pct,
                white_pct,
                unk_pct,
            ) = match.groups()

            try:
                label_section = raw_line.split(";", 1)[1]
            except IndexError:
                label_section = ""

            atlas_distribution = parse_atlas_distribution(label_section)
            primary_atlas = get_primary_atlas(atlas_distribution)

            rows.append(
                {
                    "electrode": current_electrode,
                    "contact": int(contact_num),
                    "classification": classification,
                    "gray_pct": int(gray_pct),
                    "white_pct": int(white_pct),
                    "unk_pct": int(unk_pct),
                    "primary_atlas": primary_atlas,
                    "atlas_distribution": atlas_distribution,
                }
            )

    return pd.DataFrame(rows)
