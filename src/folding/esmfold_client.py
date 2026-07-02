"""Client for folding peptide sequences with the public Meta ESMFold API."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import requests
from requests.adapters import HTTPAdapter
from urllib3.util import Retry


logger = logging.getLogger(__name__)


class ESMFoldClient:
    """Thin HTTP client for the public ESMFold sequence-to-structure endpoint."""

    def __init__(self, timeout: float = 120.0, session: Optional[requests.Session] = None) -> None:
        self.timeout = timeout
        self._owns_session = session is None
        self.session = session or self._build_session()
        self._configure_session(self.session)
        self.endpoint = "https://api.esmatlas.com/foldSequence/v1/pdb/"

    def __enter__(self) -> "ESMFoldClient":
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        self.close()
        return False

    def close(self) -> None:
        if self._owns_session:
            self.session.close()

    @staticmethod
    def _build_session() -> requests.Session:
        return requests.Session()

    @staticmethod
    def _build_retry_adapter() -> HTTPAdapter:
        retry = Retry(
            total=5,
            connect=5,
            read=5,
            status=5,
            backoff_factor=1,
            status_forcelist=(429, 500, 502, 503, 504),
            allowed_methods=frozenset({"POST"}),
            respect_retry_after_header=True,
            raise_on_status=False,
        )
        return HTTPAdapter(max_retries=retry)

    @classmethod
    def _configure_session(cls, session: requests.Session) -> None:
        adapter = cls._build_retry_adapter()
        session.mount("https://", adapter)
        session.mount("http://", adapter)

    def fold_peptide(self, sequence: str, output_path: str) -> Optional[str]:
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
            logger.warning("ESMFold request timed out for sequence length %d", len(sequence))
            return None
        except requests.RequestException as exc:
            logger.warning("ESMFold request failed after retries: %s", exc)
            return None

        pdb_text = response.text.strip()
        if not pdb_text:
            logger.warning("ESMFold returned an empty structure payload for sequence length %d", len(sequence))
            return None

        try:
            output_file.write_text(pdb_text)
        except OSError as exc:
            raise RuntimeError(f"Failed to write PDB output to {output_file}") from exc

        return str(output_file)
