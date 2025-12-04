import pytest
import numpy as np
from openparticle import ParticleOperator, generate_matrix
from openparticle.hamiltonians.yukawa_hamiltonians import yukawa_hamiltonian
from src.lobe._utils import (
    pretty_print,
    predict_number_of_block_encoding_ancillae,
    translate_antifermions_to_fermions,
    get_basis_of_full_system,
)


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


@pytest.mark.parametrize(
    ["operator", "expected_operator"],
    [
        (ParticleOperator("b0 b1 b2 b3^"), ParticleOperator("b0 b1 b2 b3^")),
        (ParticleOperator("b0 d1 b2 d0^"), ParticleOperator("b0 b4 b2 b3^")),
        (
            ParticleOperator("b0 d0") + ParticleOperator("d0^ b0^"),
            ParticleOperator("b0 b1") + ParticleOperator("b1^ b0^"),
        ),
        (ParticleOperator("a0 a2 b4 d1"), ParticleOperator("a0 a2 b4 b6")),
    ],
)
def test_translate_antifermions_to_fermions(operator, expected_operator):
    translated_operator = translate_antifermions_to_fermions(operator)
    assert translated_operator == expected_operator


@pytest.mark.parametrize("resolution", range(1, 4))
def test_translate_antifermions_to_fermions_yukawa(resolution):
    operator = yukawa_hamiltonian(resolution, 1, 1, 1)
    translated_hamiltonian = translate_antifermions_to_fermions(operator)
    assert len(operator.to_list()) == len(translated_hamiltonian.to_list())
    original_number_of_modes = (
        operator.max_fermionic_mode + operator.max_antifermionic_mode
    )
    if operator.has_fermions:
        original_number_of_modes += 1
    if operator.has_antifermions:
        original_number_of_modes += 1
    assert original_number_of_modes == translated_hamiltonian.max_fermionic_mode + 1
    assert len(operator.group()) == len(translated_hamiltonian.group())
    assert np.allclose(operator.coeffs, translated_hamiltonian.coeffs)


@pytest.mark.parametrize("number_of_terms", range(1, 10, 2))
@pytest.mark.parametrize("max_mode", range(1, 3))
@pytest.mark.parametrize("maximum_occupation", [1, 3])
def test_translate_antifermions_to_fermions_random(
    number_of_terms, max_mode, maximum_occupation
):
    operator = ParticleOperator.random(
        n_terms=number_of_terms + 1, max_mode=max_mode, normal_order=False
    )
    translated_operator = translate_antifermions_to_fermions(operator)
    num_fermionic_modes = operator.max_fermionic_mode
    if operator.has_fermions:
        num_fermionic_modes += 1
    num_antifermionic_modes = operator.max_antifermionic_mode
    if operator.has_antifermions:
        num_antifermionic_modes += 1
    num_bosonic_modes = operator.max_bosonic_mode
    if operator.has_bosons:
        num_bosonic_modes += 1
    if num_fermionic_modes is None:
        num_fermionic_modes = 0
    if num_antifermionic_modes is None:
        num_antifermionic_modes = 0
    if num_bosonic_modes is None:
        num_bosonic_modes = 0
    basis = get_basis_of_full_system(
        maximum_occupation,
        num_fermionic_modes,
        num_antifermionic_modes,
        num_bosonic_modes,
    )
    matrix = generate_matrix(operator, basis)

    num_translated_fermionic_modes = translated_operator.max_fermionic_mode
    if translated_operator.has_fermions:
        num_translated_fermionic_modes += 1
    if num_translated_fermionic_modes is None:
        num_translated_fermionic_modes = 0
    translated_basis = get_basis_of_full_system(
        maximum_occupation,
        num_translated_fermionic_modes,
        0,
        num_bosonic_modes,
    )
    translated_matrix = generate_matrix(translated_operator, translated_basis)
    assert np.allclose(matrix, translated_matrix)
