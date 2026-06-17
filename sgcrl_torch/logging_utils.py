"""Lightweight terminal/CSV logger."""

from __future__ import annotations

import csv
import os
import time
import uuid
from typing import Dict, Optional


class Logger:
    def __init__(
        self,
        label: str,
        save_dir: str,
        add_uid: bool = False,
        time_delta: float = 10.0,
        save_data: bool = True,
    ):
        self.label = label
        self.time_delta = time_delta
        self.last_time = 0.0
        self.save_data = save_data
        if add_uid:
            save_dir = os.path.join(save_dir, uuid.uuid4().hex[:8])
        self.save_dir = save_dir
        self._fieldnames = []
        self._rows = []
        self._csv_file = None
        self._writer = None
        if save_data:
            os.makedirs(save_dir, exist_ok=True)
            self._csv_file = open(os.path.join(save_dir, f"{label}.csv"), "a", newline="", encoding="utf-8")

    def write(self, metrics: Dict[str, float], force: bool = False) -> None:
        now = time.time()
        if not force and now - self.last_time < self.time_delta:
            return
        self.last_time = now
        clean = {key: _to_float(value) for key, value in metrics.items()}
        message = " | ".join(f"{key}: {value:.4g}" if isinstance(value, float) else f"{key}: {value}" for key, value in clean.items())
        print(f"[{self.label}] {message}", flush=True)
        if self.save_data:
            self._write_csv(clean)

    def _write_csv(self, metrics: Dict[str, float]) -> None:
        self._rows.append(metrics)
        new_fields = False
        for key in metrics.keys():
            if key not in self._fieldnames:
                self._fieldnames.append(key)
                new_fields = True
        if self._writer is None or new_fields:
            self._csv_file.seek(0)
            self._csv_file.truncate()
            self._writer = csv.DictWriter(self._csv_file, fieldnames=self._fieldnames)
            self._writer.writeheader()
            for row in self._rows:
                self._writer.writerow({key: row.get(key, "") for key in self._fieldnames})
        else:
            self._writer.writerow({key: metrics.get(key, "") for key in self._fieldnames})
        self._csv_file.flush()

    def close(self) -> None:
        if self._csv_file is not None:
            self._csv_file.close()


def _to_float(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return value
