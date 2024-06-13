import pytest
import numpy as np
from src.lobe._utils import get_index_of_reversed_bitstring, pretty_print


@pytest.mark.parametrize("trial", range(5))
@pytest.mark.parametrize("size_of_bitstring", range(1, 20))
def test_get_index_of_reversed_bitstring(trial, size_of_bitstring):
    bitstring = "".join(np.random.choice(["0", "1"], size=size_of_bitstring))
    original_integer = int(bitstring, 2)
    number_of_qubits = len(bitstring)
    index = get_index_of_reversed_bitstring(original_integer, number_of_qubits)

    expected_index = int(bitstring[::-1], 2)

    assert index == expected_index


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


def test_pretty_print_reverses_register_printout():
    wavefunction = np.zeros(8)
    wavefunction[1] = 1  # 001
    pretty_print_string = pretty_print(wavefunction, [3]).replace(" ", "")
    expected_string = "1.0|100>\n"
    assert pretty_print_string == expected_string


def test_pretty_print_separates_registers_properly():
    wavefunction = np.zeros(8)
    wavefunction[1] = 1  # 001
    pretty_print_string = pretty_print(wavefunction, [1, 2]).replace(" ", "")
    expected_string = "1.0|0|10>\n"
    assert pretty_print_string == expected_string
