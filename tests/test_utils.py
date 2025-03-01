import pytest
import numpy as np
from openparticle import ParticleOperator
from src.lobe._utils import pretty_print, predict_number_of_block_encoding_ancillae


def test_pretty_print_all_zeros():
    wavefunction = np.zeros(8)
    wavefunction[0] = 1
    pretty_print_string = pretty_print(wavefunction, [3]).replace(" ", "")
    expected_string = "1.0|000>\n"
    assert pretty_print_string == expected_string


def test_pretty_print_all_ones():
    wavefunction = np.zeros(8)
    wavefunction[-1] = 1
    pretty_print_string = pretty_print(wavefunction, [3]).replace(" ", "")
    expected_string = "1.0|111>\n"
    assert pretty_print_string == expected_string


@pytest.mark.parametrize("decimals", np.random.randint(1, 8, size=10))
def test_pretty_print_displays_amplitudes_correctly(decimals):
    amplitudes = (
        np.random.uniform(-1, 1, size=2) + np.random.uniform(-1, 1, size=2) * 1j
    )
    amplitudes /= np.linalg.norm(amplitudes)
    wavefunction = np.zeros(8, dtype=np.complex128)
    wavefunction[0] = amplitudes[0]
    wavefunction[-1] = amplitudes[1]
    pretty_print_string = pretty_print(
        wavefunction, [3], decimal_places=decimals
    ).replace(" ", "")
    expected_string = f"{str(amplitudes[0].round(decimals))}|000>\n{str(amplitudes[1].round(decimals))}|111>\n"
    assert pretty_print_string == expected_string


def test_pretty_print_does_not_reverse_register_printout():
    wavefunction = np.zeros(8)
    wavefunction[1] = 1  # 001
    pretty_print_string = pretty_print(wavefunction, [3]).replace(" ", "")
    expected_string = "1.0|001>\n"
    assert pretty_print_string == expected_string


def test_pretty_print_separates_registers_properly():
    wavefunction = np.zeros(8)
    wavefunction[1] = 1  # 001
    pretty_print_string = pretty_print(wavefunction, [1, 2]).replace(" ", "")
    expected_string = "1.0|0|01>\n"
    assert pretty_print_string == expected_string


@pytest.mark.parametrize(
    ["operator", "expected_be_ancillae"],
    [
        (ParticleOperator("b0 b1 b2 b3^"), 1),
        (ParticleOperator("b0 b1 b2 b3^") + ParticleOperator("b3^ b2 b1 b0"), 1),
        (ParticleOperator("a0 a1 b2 b3^") + ParticleOperator("a1^ a0 b1 b0"), 3),
        (ParticleOperator("a0 a1^ a1 a1 a3"), 3),
        (
            ParticleOperator("a0 a1^ a1 a1 a3")
            + ParticleOperator("a3^ a1^ a1^ a1 a0^"),
            4,
        ),
    ],
)
def test_predict_number_of_block_encoding_ancillae(operator, expected_be_ancillae):
    assert predict_number_of_block_encoding_ancillae(operator) == expected_be_ancillae
