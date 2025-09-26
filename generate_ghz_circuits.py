"""generate_ghz_circuits.py
Generate GHZ state preparation circuits with and without verification, including possible X errors after each gate.
"""
import stim
from typing import List
from pymatching import Matching
import numpy as np
from scipy.sparse import csc_matrix
import itertools

def generate_ghz_circuit(n_qubits: int):
    """
    Generate a GHZ state preparation circuit (no verification).

    Args:
        n_qubits (int): Number of qubits in the GHZ state.

    Returns:
        stim.Circuit: The generated GHZ circuit.
    """
    qs = range(n_qubits)
    circuit = stim.Circuit()
    circuit.append_operation("R", qs)  # Reset the first qubit to |0>
    circuit.append_operation("H", [0])  # Apply Hadamard gate to the first qubit
    for i in qs[:-1]:
        circuit.append_operation("CNOT", [i, i + 1])  # Apply CNOT from the first qubit to each other qubit
    return circuit

def generate_ghz_circuit_verified(n_qubits: int, ver: List[List[int]]):
    """
    Generate a GHZ state preparation circuit with verification.

    Args:
        n_qubits (int): Number of qubits in the GHZ state.
        ver (List[List[int]]): List of lists, each specifying which qubits are verified by each ancilla.

    Returns:
        stim.Circuit: The generated verified GHZ circuit.
    """
    qs = range(n_qubits)
    circuit = stim.Circuit()
    circuit.append_operation("R", qs)  # Reset the first qubit to |0>
    circuit.append_operation("H", [0])  # Apply Hadamard gate to the first qubit
    for i in qs[:-1]:
        circuit.append_operation("CNOT", [i, i + 1])  # Apply CNOT from the first qubit to each other qubit

    meas = range(n_qubits, n_qubits + len(ver))
    circuit.append_operation("R", meas)  # Reset the ancilla qubit
    for i, m in enumerate(meas):
        for q in ver[i]:
            circuit.append_operation("CNOT", [q, m])  # Apply CNOT from the first qubit to each other qubit
    circuit.append_operation("M", meas)  # Measure the ancilla qubit
    
    
    return circuit

def generate_ghz_circuit_with_x_errors(n_qubits: int, x_error_mask: int, prepare: bool = True):
    """
    Generate a GHZ state preparation circuit with possible X errors after each gate.

    Args:
        n_qubits (int): Number of qubits in the GHZ state.
        x_error_mask (int): Bitmask indicating which gates are followed by X errors.
            Bits 0 to n_qubits-1: after R gates (one per qubit)
            Bit n_qubits: after H gate
            Bits n_qubits+1 and up: after each CNOT (2 bits per CNOT, encoding which qubits get X errors)
        prepare (bool): If True, include initial reset operations.

    Returns:
        stim.Circuit: The generated circuit with X errors.
    """
    qs = range(n_qubits)
    circuit = stim.Circuit()
    # Gate 0: R
    if prepare:
        circuit.append_operation("R", qs)
    for i, q in enumerate(qs):
        if (x_error_mask >> i) & 1:
            circuit.append_operation("X_ERROR", q, 1)
    # Gate 1: H
    circuit.append_operation("H", [0])
    if (x_error_mask >> n_qubits) & 1:
        circuit.append_operation("X_ERROR", [0], 1)
    # Gates 2+: CNOTs
    for i, ctrl in enumerate(qs[:-1]):
        circuit.append_operation("CNOT", [ctrl, ctrl + 1])
        flag = x_error_mask >> (2*i + 1 + n_qubits)
        if (flag & 3) == 3: # both qubits get X errors
            circuit.append_operation("X_ERROR", [ctrl, ctrl + 1], 1)
        elif (flag & 3) == 2: # only target qubit gets X error
            circuit.append_operation("X_ERROR", [ctrl + 1], 1)
        elif (flag & 3) == 1: # only control qubit gets X error
            circuit.append_operation("X_ERROR", [ctrl], 1)
        else: pass
        
    
    return circuit

def generate_ghz_verified_circuit_with_x_errors(n_qubits: int, x_error_mask: int, ver: List[List[int]], prepare: bool = True):
    """
    Generate a verified GHZ state preparation circuit with possible X errors after each gate.

    Args:
        n_qubits (int): Number of qubits in the GHZ state.
        x_error_mask (int): Bitmask indicating which gates are followed by X errors.
        ver (List[List[int]]): List of lists, each specifying which qubits are verified by each ancilla.
        prepare (bool): If True, include initial reset operations.

    Returns:
        stim.Circuit: The generated verified circuit with X errors.
    """
    circuit = generate_ghz_circuit_with_x_errors(n_qubits, x_error_mask, prepare)
    meas = range(n_qubits, n_qubits + len(ver))
    circuit.append_operation("TICK")
    if prepare:
        circuit.append_operation("R", meas)  # Reset the ancilla qubit
    for i, m in enumerate(meas):
        for q in ver[i]:
            circuit.append_operation("CNOT", [q, m])  # Apply CNOT from the first qubit to each other qubit
    circuit.append_operation("TICK")
    return circuit

def find_weight(error_mask, n_qubits):
    """
    Compute the total number of X errors in a given error mask.

    Args:
        error_mask (int): Bitmask encoding X errors after each gate.
        n_qubits (int): Number of qubits in the GHZ state.

    Returns:
        int: Total number of X errors (weight).
    """
    # Count X errors after one-qubit gates
    one_qubit_gates = (error_mask & ((1 << (n_qubits + 1)) - 1)).bit_count()

    # Count X errors after CNOT gates
    cnots = error_mask >> (1 + n_qubits)
    cnots = ((cnots >> 1) | cnots) & 0x555555555555555
    return one_qubit_gates + cnots.bit_count()


def generate_fixed_weight(N, k):
    """
    Yield all integers < 2^N with exactly k bits set.

    Args:
        N (int): Number of bits.
        k (int): Number of bits set (weight).

    Yields:
        int: Integer with k bits set.
    """
    if k == 0:
        yield 0
        return
    x = (1 << k) - 1  # smallest k-bit number
    limit = 1 << N
    while x < limit:
        yield x
        # Gosper's hack: next with same popcount
        c = x & -x
        r = x + c
        x = (((r ^ x) >> 2) // c) | r

def find_errors_fast(n_qubits):
    """
    Efficiently find all error masks with weight up to max_weight for n_qubits.

    Args:
        n_qubits (int): Number of qubits in the GHZ state.

    Returns:
        List[int]: List of error masks with valid weights.
    """
    max_weight = (n_qubits - 1) // 2 - 1
    n_bits = n_qubits + 1 + n_qubits - 1

    errors = []
    # For each possible error weight, generate all error masks of that weight
    for weight in range(1, max_weight + 1):
        for error_mask in generate_fixed_weight(n_bits, weight):
            mask_errors = []
            bottom = error_mask & ((1 << (n_qubits + 1)) - 1)
            mask_errors.append(bottom)
            top = error_mask >> (n_qubits + 1)
            # For each CNOT, expand error mask to account for possible error locations
            for i in range(n_qubits - 1):
                if (top >> i) & 1:
                    new_errors = []
                    for e in mask_errors:
                        for j in range(3):
                            err = e | ((j + 1) << (n_qubits + 1 + 2 * i))
                            new_errors.append(err)
                    mask_errors = new_errors
            errors += mask_errors
    return errors

def generate_noisy_ghz_circuits(n_qubits: int):
    """
    Generate a batch of noisy GHZ circuits with all possible X error patterns up to max_weight.

    Args:
        n_qubits (int): Number of qubits in the GHZ state.

    Returns:
        Tuple[stim.Circuit, List[int]]: The batch circuit and list of error masks used.
    """
    max_weight = (n_qubits - 1) // 2 - 1 
    errors = find_errors_fast(n_qubits)
    print(f"Generating {len(errors)} circuits...")
    print(f"Max weight of X errors: {max_weight}")
    circuit = stim.Circuit()
    circuit.append_operation("R", range(n_qubits))
    for e in errors:
        circuit += generate_ghz_circuit_with_x_errors(n_qubits, e, prepare=False)
        circuit.append_operation("TICK")
        circuit.append_operation("MR", range(n_qubits))
        circuit.append_operation("TICK")
        
    return circuit, errors


def generate_noisy_verified_ghz_circuits(n_qubits: int, ver: List[List[int]]):
    """
    Generate a batch of noisy verified GHZ circuits with all possible X error patterns up to max_weight.

    Args:
        n_qubits (int): Number of qubits in the GHZ state.
        ver (List[List[int]]): List of lists, each specifying which qubits are verified by each ancilla.

    Returns:
        Tuple[stim.Circuit, List[int]]: The batch circuit and list of error masks used.
    """
    errors = find_errors_fast(n_qubits)
    circuit = stim.Circuit()
    qs = range(n_qubits + len(ver))
    circuit.append_operation("R", qs)
    for e in errors:
        circuit += generate_ghz_verified_circuit_with_x_errors(n_qubits, e, ver, prepare=False)
        circuit.append_operation("TICK")
        circuit.append_operation("MR", qs)
        circuit.append_operation("TICK")
        
    return circuit, errors


def get_error_weights(n_qubits, measurements):
    def repetition_code(n):
        """
        Construct the parity check matrix of a repetition code of length n.

        Args:
            n (int): Length of the code.

        Returns:
            csc_matrix: Parity check matrix.
        """
        row_ind, col_ind = zip(*((i, j) for i in range(n - 1) for j in (i, (i+1)%n)))
        data = np.ones(2*(n - 1), dtype=np.uint8)
        return csc_matrix((data, (row_ind, col_ind)))
    
    H = repetition_code(n_qubits)
    m = Matching(H)
    syndromes = H@measurements.T % 2
    errors_predicted = m.decode_batch(syndromes.T)
    error_weights = np.sum(errors_predicted, axis=1)
    return error_weights


def is_circuit_ft(n_qubits, ver):
    """
    Test if the verified GHZ circuit is fault-tolerant for a given verification pattern.

    Args:
        n_qubits (int): Number of qubits in the GHZ state.
        ver (List[List[int]]): List of lists, each specifying which qubits are verified by each ancilla.

    Returns:
        Tuple[int, int]: Number of false negatives and false positives.
    """
    # Generatre circuits
    circuits, errors = generate_noisy_verified_ghz_circuits(n_qubits, ver)
    # Compile and sample
    sampler = circuits.compile_sampler()
    results = sampler.sample(1)
    
    results = results.reshape((len(errors), n_qubits + len(ver)))
    measurements = results[:,:-len(ver)]
    verifications = results[:,-len(ver):]
    error_weights = get_error_weights(n_qubits, measurements)
    n_faults = np.array([find_weight(e, n_qubits) for e in errors])
    # False negatives: error weight exceeds number of faults and verification fails
    false_negatives = error_weights - n_faults > np.sum(verifications, axis=1)
    # False positives: error weight does not exceed faults but verification triggers
    # false_positives = (n_faults >= error_weights) & np.any(verifications, axis=1)
    # print(f"False negatives found for verification {ver}: {false_negatives.sum()}")
    if false_negatives.sum() > 0:
        return False
    else:
        return True
    
def search_ft_verifications(n, n_verifications):
    ft_vers = []
    tuples = [(i, j) for i in range(n) for j in range(i)]
    for comb in itertools.combinations(tuples, n_verifications):
        if is_circuit_ft(n, list(comb)):
            ft_vers.append(comb)
    return ft_vers