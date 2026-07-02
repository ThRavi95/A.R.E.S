"""Client for folding peptide sequences with the public Meta ESMFold API."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import requests


class ESMFoldClient:
    """Thin HTTP client for the public ESMFold sequence-to-structure endpoint."""

    def __init__(self, timeout: float = 120.0, session: Optional[requests.Session] = None) -> None:
        self.timeout = timeout
        self.session = session or requests.Session()
        self.endpoint = "https://api.esmatlas.com/foldSequence/v1/pdb/"

    def fold_peptide(self, sequence: str, output_path: str) -> str:
        """Fold a peptide sequence and write the returned PDB content to disk."""
        sequence = sequence.strip()
        if not sequence:
            raise ValueError("sequence must be a non-empty amino acid string")

        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)

        try:
            response = self.session.post(
                self.endpoint,
                data=sequence,
                timeout=self.timeout,
                headers={"Content-Type": "text/plain; charset=utf-8"},
            )
            response.raise_for_status()
        except requests.Timeout as exc:
            raise RuntimeError("ESMFold request timed out") from exc
        except requests.RequestException as exc:
            raise RuntimeError(f"ESMFold request failed: {exc}") from exc

        try:
            output_file.write_text(response.text)
        except OSError as exc:
            raise RuntimeError(f"Failed to write PDB output to {output_file}") from exc

        return str(output_file)
