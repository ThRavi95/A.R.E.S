"""Docking wrapper around AutoDock Vina for peptide ligands."""

from __future__ import annotations

import re
import shutil
import subprocess
from pathlib import Path


class VinaDocker:
    """Run AutoDock Vina against a fixed receptor and binding pocket."""

    def __init__(
        self,
        vina_executable: str,
        receptor_pdbqt: str,
        center_x: float,
        center_y: float,
        center_z: float,
        size_x: float,
        size_y: float,
        size_z: float,
    ) -> None:
        self.vina_executable = self._resolve_vina_executable(vina_executable)
        self.receptor_pdbqt = self._resolve_existing_file(receptor_pdbqt, "receptor")
        self.center_x = float(center_x)
        self.center_y = float(center_y)
        self.center_z = float(center_z)
        self.size_x = float(size_x)
        self.size_y = float(size_y)
        self.size_z = float(size_z)

    def prepare_ligand(self, pdb_file: str, out_pdbqt: str) -> str:
        """Convert a peptide PDB file into a charged PDBQT ligand file."""
        pdb_path = self._resolve_existing_file(pdb_file, "ligand PDB")
        out_path = Path(out_pdbqt)
        out_path.parent.mkdir(parents=True, exist_ok=True)

        try:
            from rdkit import Chem
            from meeko import MoleculePreparation

            try:
                from meeko import PDBQTWriterLegacy as PDBQTWriter
            except ImportError:
                from meeko import PDBQTWriter
        except ImportError as exc:
            raise RuntimeError(
                "Ligand preparation requires the 'rdkit' and 'meeko' packages"
            ) from exc

        molecule = Chem.MolFromPDBFile(str(pdb_path), removeHs=False)
        if molecule is None:
            raise ValueError(f"Unable to read PDB ligand from {pdb_path}")

        preparation = MoleculePreparation()
        prepared = preparation.prepare(molecule)
        if isinstance(prepared, (list, tuple)):
            prepared = prepared[0]

        write_result = PDBQTWriter.write_string(prepared)
        if isinstance(write_result, tuple):
            pdbqt_text = write_result[0]
            success = write_result[1] if len(write_result) > 1 else True
            error_message = write_result[2] if len(write_result) > 2 else ""
            if not success:
                raise RuntimeError(error_message or f"Failed to prepare ligand {pdb_path}")
        else:
            pdbqt_text = write_result

        out_path.write_text(pdbqt_text)
        return str(out_path)

    def run_docking(self, ligand_pdbqt: str, output_log: str) -> str:
        """Execute Vina headlessly and write its log to disk."""
        ligand_path = self._resolve_existing_file(ligand_pdbqt, "ligand PDBQT")
        log_path = Path(output_log)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        output_pose_path = log_path.with_suffix(".pdbqt")

        command = [
            self.vina_executable,
            "--receptor",
            str(self.receptor_pdbqt),
            "--ligand",
            str(ligand_path),
            "--center_x",
            str(self.center_x),
            "--center_y",
            str(self.center_y),
            "--center_z",
            str(self.center_z),
            "--size_x",
            str(self.size_x),
            "--size_y",
            str(self.size_y),
            "--size_z",
            str(self.size_z),
            "--out",
            str(output_pose_path),
            "--log",
            str(log_path),
        ]

        try:
            subprocess.run(command, check=True, capture_output=True, text=True)
        except subprocess.CalledProcessError as exc:
            stderr = exc.stderr.strip() if exc.stderr else ""
            stdout = exc.stdout.strip() if exc.stdout else ""
            details = stderr or stdout or f"exit code {exc.returncode}"
            raise RuntimeError(f"Vina docking failed: {details}") from exc

        return str(log_path)

    def parse_affinity(self, log_file: str) -> float:
        """Extract the top-mode affinity score from a Vina log file."""
        log_path = self._resolve_existing_file(log_file, "Vina log")
        log_text = log_path.read_text()

        patterns = (
            r"REMARK VINA RESULT:\s+(-?\d+(?:\.\d+)?)",
            r"^\s*1\s+(-?\d+(?:\.\d+)?)\b",
        )
        for pattern in patterns:
            match = re.search(pattern, log_text, flags=re.MULTILINE)
            if match:
                return float(match.group(1))

        raise ValueError(f"Could not find a Mode 1 affinity score in {log_path}")

    @staticmethod
    def _resolve_existing_file(path: str, label: str) -> Path:
        file_path = Path(path)
        if not file_path.is_file():
            raise FileNotFoundError(f"{label} file not found: {file_path}")
        return file_path

    @staticmethod
    def _resolve_vina_executable(vina_executable: str) -> str:
        executable_path = Path(vina_executable)
        if executable_path.is_file():
            return str(executable_path)

        resolved = shutil.which(vina_executable)
        if resolved is None:
            raise FileNotFoundError(f"Vina executable not found: {vina_executable}")
        return resolved