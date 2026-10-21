import json
from pathlib import Path

import pytest

FIXTURES = Path(__file__).parent / "fixtures"


@pytest.fixture
def qna_structure_payload() -> dict:
    """Trimmed SDMX-JSON structure for the quarterly national accounts dataflow."""
    return json.loads((FIXTURES / "oecd_qna_structure.json").read_text())


@pytest.fixture
def qna_gdp_csv() -> str:
    """Real OECD CSV response: GDP for USA and DEU from 2025Q1."""
    return (FIXTURES / "oecd_qna_gdp.csv").read_text()
