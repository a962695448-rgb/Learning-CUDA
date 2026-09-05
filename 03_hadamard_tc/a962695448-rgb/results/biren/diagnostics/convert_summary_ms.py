#!/usr/bin/env python3
"""Convert summary *_us columns to *_ms exactly; keep original CSVs unchanged."""

import argparse
import csv
from decimal import Decimal, InvalidOperation
from pathlib import Path


def milliseconds(text):
    if text == "":
        return text
    try:
        value = Decimal(text)
    except InvalidOperation as error:
        raise ValueError(f"Invalid decimal time: {text!r}") from error
    if not value.is_finite():
        raise ValueError(f"Non-finite time: {text!r}")
    sign, digits, exponent = value.as_tuple()
    # Exact decimal division by 1000; no binary float or context rounding.
    converted = Decimal((sign, digits, exponent - 3))
    return format(converted, "f")


def convert(source):
    destination = source.with_name(source.stem + "_ms.csv")
    if source.name.endswith("_ms.csv"):
        raise ValueError("Input already has an _ms filename")
    if destination.exists():
        raise FileExistsError(f"Refuse to overwrite: {destination}")
    with source.open("r", encoding="utf-8", newline="") as stream:
        reader = csv.reader(stream)
        columns = next(reader)
        indices = [i for i, name in enumerate(columns) if name.endswith("_us")]
        if not indices:
            raise ValueError(f"No *_us columns: {source}")
        renamed = [name[:-3] + "_ms" if name.endswith("_us") else name for name in columns]
        if len(set(renamed)) != len(renamed):
            raise ValueError("Renaming would produce duplicate column names")
        rows = []
        for row in reader:
            if len(row) != len(columns):
                raise ValueError("CSV row length differs from header")
            for index in indices:
                row[index] = milliseconds(row[index])
            rows.append(row)
    with destination.open("x", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream, lineterminator="\n")
        writer.writerow(renamed)
        writer.writerows(rows)
    print(f"{source} -> {destination}: {len(rows)} rows, {len(indices)} time columns")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("sources", type=Path, nargs="+", help="Existing method_summary.csv files")
    args = parser.parse_args()
    for source in args.sources:
        convert(source)


if __name__ == "__main__":
    main()
