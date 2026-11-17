"""Schema probe for the ATD pair. Prints structure and aggregates only, never rows or values.

    uv run python analysis/00_schema_probe.py

Answers: which XML elements/channels exist, how a raw line is laid out (sections, groups, fields
per group), and for every ball field its NaN rate and (if it looks categorical) its value counts.
"""

import collections
import os
import sys
import xml.etree.ElementTree as ET

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from games import asset_paths  # noqa: E402

# Params
CATEGORICAL_MAX_UNIQUE = 8   # a ball field with at most this many distinct values is shown as counts
NUMERIC_SAMPLE_EVERY = 1     # stream every line


def strip_ns(tag):
    return tag.split("}", 1)[-1]


def xml_schema(path):
    tree = ET.parse(path)
    paths = collections.Counter()
    attrs = collections.defaultdict(set)

    def walk(node, prefix):
        here = prefix + "/" + strip_ns(node.tag)
        paths[here] += 1
        attrs[here].update(node.attrib.keys())
        for child in node:
            walk(child, here)

    walk(tree.getroot(), "")
    print("XML element paths (count, attribute names):")
    for p, n in sorted(paths.items()):
        print("  {:>5}  {}  {}".format(n, p, sorted(attrs[p]) or ""))

    # Channel / data-spec ids are format definitions, not match data.
    print("\nchannel-like ids:")
    for node in tree.iter():
        tag = strip_ns(node.tag)
        if tag in ("Channel", "PlayerChannel", "SensorChannel", "Sensor", "BallChannel"):
            ident = node.attrib.get("id") or node.attrib.get("channelId") or ""
            name = node.findtext("{*}Name") or ""
            unit = node.findtext("{*}Unit") or ""
            print("  {:<14} id={!r:<28} name={!r:<28} unit={!r}".format(tag, ident, name, unit))


def raw_layout(path, source):
    n_sections = collections.Counter()
    n_player_groups = collections.Counter()
    fields_per_player_group = collections.Counter()
    ball_n_fields = collections.Counter()
    ball_fields = collections.defaultdict(collections.Counter)  # index -> value counter (capped)
    ball_nan = collections.Counter()
    lines = 0
    with open(path) as handle:
        for line in handle:
            lines += 1
            parts = line.rstrip("\n").split(":")
            n_sections[len(parts)] += 1
            if len(parts) >= 2:
                groups = [g for g in parts[1].split(";") if g]
                n_player_groups[len(groups)] += 1
                for g in groups[:3]:
                    fields_per_player_group[len(g.split(","))] += 1
            if len(parts) >= 3:
                ball = parts[-1].strip().rstrip(";").split(",")
                ball_n_fields[len(ball)] += 1
                for i, v in enumerate(ball):
                    if v.lower() in ("nan", ""):
                        ball_nan[i] += 1
                    c = ball_fields[i]
                    if len(c) <= CATEGORICAL_MAX_UNIQUE or v in c:
                        c[v] += 1
                    else:
                        c["<many>"] += 1
    print("\n[{}] raw lines: {}".format(source, lines))
    print("  sections per line ':'        {}".format(dict(n_sections)))
    print("  player groups per line       {}".format(dict(n_player_groups)))
    print("  fields per player group      {}".format(dict(fields_per_player_group)))
    print("  fields in ball group         {}".format(dict(ball_n_fields)))
    for i in sorted(ball_fields):
        c = ball_fields[i]
        distinct = [k for k in c if k != "<many>"]
        categorical = "<many>" not in c and len(distinct) <= CATEGORICAL_MAX_UNIQUE
        print("  ball field {}: NaN {:.1%}  {}".format(
            i, ball_nan[i] / lines,
            ("categorical counts " + str(dict(c))) if categorical else "numeric / high-cardinality"))


if __name__ == "__main__":
    for source in ("tactical", "broadcast"):
        paths = asset_paths(source=source)
        print("=" * 78, "\n", source)
        if source == "tactical":
            xml_schema(paths["atd_meta"])
        raw_layout(paths["atd_raw"], source)
