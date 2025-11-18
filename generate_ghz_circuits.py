"""generate_ghz_circuits.py
Generate GHZ state preparation circuits with and without verification, including possible X errors after each gate.
"""
import stim
from typing import List
from pymatching import Matching
import numpy as np
from scipy.sparse import csc_matrix
import itertools
from mqt.qecc.circuit_synthesis.noise import CircuitLevelNoise


def generate_ghz_circuit(n_qubits: int, reset: bool = True):
    """
    Generate a GHZ state preparation circuit (no verification).

    Args:
        n_qubits (int): Number of qubits in the GHZ state.

    Returns:
        stim.Circuit: The generated GHZ circuit.
    """
    qs = range(n_qubits)
    circuit = stim.Circuit()
    if reset:
        circuit.append_operation("R", qs)  # Reset the first qubit to |0>
    circuit.append_operation("H", [0])  # Apply Hadamard gate to the first qubit
    for i in qs[:-1]:
        circuit.append_operation("CNOT", [i, i + 1])  # Apply CNOT from the first qubit to each other qubit
    return circuit

def generate_ghz_circuit_verified(n_qubits: int, ver: List[List[int]], prepare: bool = True):
    """
    Generate a GHZ state preparation circuit with verification.

    Args:
        n_qubits (int): Number of qubits in the GHZ state.
        ver (List[List[int]]): List of lists, each specifying which qubits are verified by each ancilla.

    Returns:
        stim.Circuit: The generated verified GHZ circuit.
    """
    qs = range(n_qubits)
    meas = range(n_qubits, n_qubits + len(ver))
    circuit = stim.Circuit()
    if prepare:
        circuit.append_operation("R", qs)  # Reset the first qubit to |0>
        circuit.append_operation("R", meas)  # Reset the ancilla qubit
    circuit.append_operation("H", [0])  # Apply Hadamard gate to the first qubit
    for i in qs[:-1]:
        circuit.append_operation("CNOT", [i, i + 1])  # Apply CNOT from the first qubit to each other qubit

    for i, m in enumerate(meas):
        for q in ver[i]:
            circuit.append_operation("CNOT", [q, m])  # Apply CNOT from the first qubit to each other qubit
    circuit.append_operation("MR", meas)  # Measure the ancilla qubit
    
    
    return circuit


def generate_noisy_ghz_verification_circuit(n_qubits: int, ver: List[List[int]], p: float):
    circuit = generate_ghz_circuit_verified(n_qubits, ver, prepare = False)
    noisy_circuit = CircuitLevelNoise(p,p,p,p).apply(circuit)
    noisy_circuit.append("TICK")
    noisy_circuit.append("MR", range(n_qubits))
    return noisy_circuit
    
def generate_ghz_circuit_from_error_mask(n_qubits: int, x_error_mask: int, prepare: bool = True):
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

def generate_ghz_circuit_from_error_dict(n_qubits: int, error_dict: dict, prepare: bool = True):
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
        if i in error_dict:
            circuit.append_operation("X_ERROR", q, 1)
    # Gate 1: H
    circuit.append_operation("H", [0])
    if n_qubits in error_dict:
        circuit.append_operation("X_ERROR", [0], 1)
    # Gates 2+: CNOTs
    gate_index = n_qubits + 1
    for i, ctrl in enumerate(qs[:-1]):
        circuit.append_operation("CNOT", [ctrl, ctrl + 1])
        if gate_index in error_dict and error_dict[gate_index] == "XX": # both qubits get X errors
            circuit.append_operation("X_ERROR", [ctrl, ctrl + 1], 1)
        elif gate_index in error_dict and error_dict[gate_index] == "IX": # only target qubit gets X error
            circuit.append_operation("X_ERROR", [ctrl + 1], 1)
        elif gate_index in error_dict and error_dict[gate_index] == "XI": # only control qubit gets X error
            circuit.append_operation("X_ERROR", [ctrl], 1)
        else: pass

        gate_index += 1
        
    
    return circuit, gate_index

def generate_ghz_verified_circuit_from_error_maks(n_qubits: int, x_error_mask: int, ver: List[List[int]], prepare: bool = True):
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
    circuit = generate_ghz_circuit_from_error_mask(n_qubits, x_error_mask, prepare)
    meas = range(n_qubits, n_qubits + len(ver))
    circuit.append_operation("TICK")
    if prepare:
        circuit.append_operation("R", meas)  # Reset the ancilla qubit
 
    for i, m in enumerate(meas):
        for q in ver[i]:
            circuit.append_operation("CNOT", [q, m])  # Apply CNOT from the first qubit to each other qubit
            
    circuit.append_operation("TICK")
    return circuit

def generate_ghz_verified_circuit_from_error_dict(n_qubits: int, error_dict: dict, ver: list[list[int]], prepare: bool = True):
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
    circuit, gate_index = generate_ghz_circuit_from_error_dict(n_qubits, error_dict, prepare)
    meas = range(n_qubits, n_qubits + len(ver))
    circuit.append_operation("TICK")
    if prepare:
        circuit.append_operation("R", meas)  # Reset the ancilla qubit
    for i, m in enumerate(meas):
        if gate_index in dict(error_dict):
            circuit.append_operation("X_ERROR", m, 1)
        gate_index += 1
    for i, m in enumerate(meas):
        for q in ver[i]:
            circuit.append_operation("CNOT", [q, m])  # Apply CNOT from the first qubit to each other qubit
            if (gate_index in error_dict):
                if error_dict[gate_index] == "XX": # both qubits get X errors
                    circuit.append_operation("X_ERROR", [q, m], 1)
                elif error_dict[gate_index] == "IX": # only target qubit gets X error
                    circuit.append_operation("X_ERROR", [m], 1)
                elif error_dict[gate_index] == "XI": # only control qubit gets X error
                    circuit.append_operation("X_ERROR", [q], 1)
            gate_index += 1
            
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
        
def get_indices_of_set_bits(x):
    """
    Get the indices of set bits in an integer.

    Args:
        x (int): The integer to analyze.

    Returns:
        List[int]: List of indices where bits are set.
    """
    indices = []
    index = 0
    while x:
        if x & 1:
            indices.append(index)
        x >>= 1
        index += 1
    return indices
        
        
def add_errors_after_gates_recursively(affected_gates: list, type_mask: int):
    one_qubit_errors = ["X"]
    #one_qubit_errors = ["X", "Y", "Z"]
    """two_qubit_errors = [
        "IX", "IY", "IZ", "XI", "YI", "ZI",
        "XX", "XY", "XZ", "YX", "YY", "YZ",
        "ZX", "ZY", "ZZ"
    ]"""
    two_qubit_errors = ["IX", "XI", "XX"]
    # Base case: only one gate
    if len(affected_gates) == 1:
        gate_type = (type_mask >> affected_gates[0]) & 1
        if gate_type:  # two-qubit
            for err in two_qubit_errors:
                yield {affected_gates[0]: err}
        else:  # one-qubit
            for err in one_qubit_errors:
                yield {affected_gates[0]: err}
        return

    # Recursive case
    for sub_errors in add_errors_after_gates_recursively(affected_gates[1:], type_mask):
        gate_type = (type_mask >> affected_gates[0]) & 1
        if gate_type:
            for err in two_qubit_errors:
                yield sub_errors | {affected_gates[0]: err}
        else:  
            for err in one_qubit_errors:
                yield sub_errors | {affected_gates[0]: err}


def find_errors(n_gates, n_faults, type_mask):
    for error_mask in generate_fixed_weight(n_gates, n_faults):
        affected_gates = get_indices_of_set_bits(error_mask)
        for error in add_errors_after_gates_recursively(affected_gates, type_mask):
            yield error
                
def find_circuit_errors(n_qubits, n_ancilla=0):
    """
    Find all error masks with weight up to max_weight for n_qubits.

    Args:
        n_qubits (int): Number of qubits in the GHZ state.

    Returns:
        List[int]: List of error masks with valid weights.
    """
    max_weight = (n_qubits - 1) // 2
    n_gates = 2 * n_qubits + 3 * n_ancilla 
    ghz_2q_gates = (1 << (n_qubits - 1)) - 1
    ver_2q_gates = (1 << (2 * n_ancilla)) - 1
    type_mask = ghz_2q_gates << (n_qubits + 1) | ver_2q_gates << (2 * n_qubits + n_ancilla)
    errors = []
    # For each possible error weight, generate all error masks of that weight
    for n_faults in range(1, max_weight):
        for error_dict in find_errors(n_gates, n_faults, type_mask):
            errors.append(error_dict)
    return errors

def find_errors_fast(n_qubits):
    """
    Efficiently find all error masks with weight up to max_weight for n_qubits.

    Args:
        n_qubits (int): Number of qubits in the GHZ state.

    Returns:
        List[int]: List of error masks with valid weights.
    """
    max_weight = (n_qubits - 1) // 2 - 1
    n_bits = 2 * n_qubits

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
        circuit += generate_ghz_circuit_from_error_mask(n_qubits, e, prepare=False)
        circuit.append_operation("TICK")
        circuit.append_operation("MR", range(n_qubits))
        circuit.append_operation("TICK")
        
    return circuit, errors



def generate_noisy_verified_ghz_circuits_fast(n_qubits: int, ver: List[List[int]]):
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
        circuit += generate_ghz_verified_circuit_from_error_maks(n_qubits, e, ver, prepare=False)
        circuit.append_operation("TICK")
        circuit.append_operation("MR", qs)
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
    errors = find_circuit_errors(n_qubits, len(ver))
    circuit = stim.Circuit()
    qs = range(n_qubits + len(ver))
    circuit.append_operation("R", qs)
    for e in errors:
        circuit += generate_ghz_verified_circuit_from_error_dict(n_qubits, e, ver, prepare=False)
        circuit.append_operation("TICK")
        circuit.append_operation("MR", qs)
        circuit.append_operation("TICK")
        
    return circuit, errors

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
    
def get_error_weights(n_qubits, measurements):
    
    H = repetition_code(n_qubits)
    m = Matching(H)
    syndromes = H@measurements.T % 2
    errors_predicted = m.decode_batch(syndromes.T)
    error_weights = np.sum(errors_predicted, axis=1)
    return error_weights

def run_verified_ghz_fast(n_qubits, ver):
    # Generatre circuits
    circuits, errors = generate_noisy_verified_ghz_circuits_fast(n_qubits, ver)
    # Compile and sample
    sampler = circuits.compile_sampler()
    results = sampler.sample(1)
    
    results = results.reshape((len(errors), n_qubits + len(ver)))
    measurements = results[:,:-len(ver)]
    verifications = results[:,-len(ver):]
    error_weights = get_error_weights(n_qubits, measurements)
    n_faults = np.array([find_weight(e, n_qubits) for e in errors])
    
    return n_faults, error_weights, verifications, errors

def run_verified_ghz(n_qubits: int, ver: list[list]):
    """
        Run a GHZ verication circuit exhaustively adding errros.
        Parameters
        ----------
        n_qubits : int
            Number of data qubits used to prepare the GHZ state.
        ver : list[list]
            List of qubits touched by each measurement.
        Returns
        -------
        n_faults : numpy.ndarray[int]
            1D integer array of length N giving the number of faults associated with
            each generated noisy circuit instance. N equals the number of error
            instances produced by generate_noisy_verified_ghz_circuits.
        error_weights : numpy.ndarray
            1D integer array of length N giving the weight of the final error on the data qubits.
        verifications : numpy.ndarray
            Binary matrix of shape (N, len(ver)) containing the sampled verification
            measurement outcomes (0/1) for each instance and each verification bit.
        errors : list
            List of raw error descriptions returned by generate_noisy_verified_ghz_circuits.
            The list length is N and each element describes the faults inserted into
            the corresponding circuit instance.
        Example
        -------
        >>> # Produce summaries for a 5-qubit GHZ with one verification
        >>> n_faults, error_weights, verifications, errors = run_verified_ghz(5, [[0,1]])
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
    n_faults = np.array([len(e) for e in errors])
    
    return n_faults, error_weights, verifications, errors

def run_verified_ghz_mc(n_qubits: int, ver: list[list], p: float, n_runs: int):
    """
        Run a GHZ verication circuit exhaustively adding errros.
        Parameters
        ----------
        n_qubits : int
            Number of data qubits used to prepare the GHZ state.
        ver : list[list]
            List of qubits touched by each measurement.
        Returns
        -------
        n_faults : numpy.ndarray[int]
            1D integer array of length N giving the number of faults associated with
            each generated noisy circuit instance. N equals the number of error
            instances produced by generate_noisy_verified_ghz_circuits.
        error_weights : numpy.ndarray
            1D integer array of length N giving the weight of the final error on the data qubits.
        verifications : numpy.ndarray
            Binary matrix of shape (N, len(ver)) containing the sampled verification
            measurement outcomes (0/1) for each instance and each verification bit.
        errors : list
            List of raw error descriptions returned by generate_noisy_verified_ghz_circuits.
            The list length is N and each element describes the faults inserted into
            the corresponding circuit instance.
        Example
        -------
        >>> # Produce summaries for a 5-qubit GHZ with one verification
        >>> n_faults, error_weights, verifications, errors = run_verified_ghz(5, [[0,1]])
    """
   
    # Generatre circuits
    circuit = generate_noisy_ghz_verification_circuit(n_qubits, ver, p=p)
    # Compile and sample
    sampler = circuit.compile_sampler()
    results = sampler.sample(n_runs)
    
    results = results.reshape((n_runs, n_qubits + len(ver)))
    measurements = results[:,:-len(ver)]
    verifications = results[:,-len(ver):]
    error_weights = get_error_weights(n_qubits, measurements)
    
    return error_weights, verifications

def is_circuit_ft_fast(n_qubits, ver):
    """
    Test if the verified GHZ circuit is fault-tolerant for a given verification pattern.

    Args:
        n_qubits (int): Number of qubits in the GHZ state.
        ver (List[List[int]]): List of lists, each specifying which qubits are verified by each ancilla.

    Returns:
        Tuple[int, int]: Number of false negatives and false positives.
    """
    # Generatre circuits
    
    circuits, errors = generate_noisy_verified_ghz_circuits_fast(n_qubits, ver)
    # Compile and sample
    sampler = circuits.compile_sampler()
    results = sampler.sample(1)
    
    results = results.reshape((len(errors), n_qubits + len(ver)))
    measurements = results[:,:-len(ver)]
    verifications = results[:,-len(ver):]
    n_faults = np.array([find_weight(e, n_qubits) for e in errors])
    # False negatives: error weight exceeds number of faults and verification fails
    error_weights = get_error_weights(n_qubits, measurements)
    false_negatives = error_weights - n_faults > np.sum(verifications, axis=1)
    # False positives: error weight does not exceed faults but verification triggers
    # false_positives = (n_faults >= error_weights) & np.any(verifications, axis=1)
    positives = np.any(verifications, axis=1)
    
    #print(f"False negatives found for verification {ver}: {false_negatives.sum()}")
    if false_negatives.sum() > 0:
        return False
    else:
        return True


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
    n_faults = np.array([len(e) for e in errors])
    error_weights = get_error_weights(n_qubits, measurements)
    false_negatives = error_weights - n_faults > np.sum(verifications, axis=1)
    
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

def search_ft_verifications_fast(n, n_verifications):
    ft_vers = []
    tuples = [(i, j) for i in range(n) for j in range(i)]
    for comb in itertools.combinations(tuples, n_verifications):
        if is_circuit_ft_fast(n, list(comb)):
            ft_vers.append(comb)
    return ft_vers

if __name__ == "__main__":
    errors = find_circuit_errors(5, 0)
    print(f"Number of errors found: {len(errors)}")
    errors = find_errors_fast(5)
    print(f"Number of errors found (fast): {len(errors)}")

       
    