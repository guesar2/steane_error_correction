import stim

def generate_ghz_circuit(n_qubits: int):
    """
    Generate a GHZ state preparation circuit.

    Args:
        n_qubits (int): number of qubits.
    
    """
    qs = range(n_qubits)
    circuit = stim.Circuit()
    circuit.append_operation("R", qs)  # Reset the first qubit to |0>
    circuit.append_operation("H", [0])  # Apply Hadamard gate to the first qubit
    for i in qs[:-1]:
        circuit.append_operation("CNOT", [i, i + 1])  # Apply CNOT from the first qubit to each other qubit
    return circuit

def generate_ghz_circuit_with_x_errors(n_qubits: int, x_error_mask: int, prepare: bool = True):
    """
    Generate a GHZ state preparation circuit with possible X errors after each gate.

    Args:
        n_qubits (int): Number of qubits.
        x_error_mask (int): Bitmask indicating which gates are followed by X errors.
                            Bits 0 - d-1: after R, Bit d: after H, Bits d+: after each CNOT.
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
        if (flag & 3) == 3:
            circuit.append_operation("X_ERROR", [ctrl, ctrl + 1], 1)
        elif (flag & 3) == 2:
            circuit.append_operation("X_ERROR", [ctrl + 1], 1)
        elif (flag & 3) == 1:
            circuit.append_operation("X_ERROR", [ctrl], 1)
        else: pass
    return circuit

def find_weight(error_mask, n_qubits):
    one_qubit_gates = (error_mask & ((1 << n_qubits + 1) - 1)).bit_count()

    cnots = error_mask >> 1 + n_qubits
 
    cnots = ((cnots >> 1) | cnots) & 0x555555555555555
    
    return one_qubit_gates + cnots.bit_count()


def generate_noisy_ghz_circuits(n_qubits: int):
    """
    Generate a GHZ state preparation circuit with possible X errors after each gate.

    Args:
        n_qubits (int): Number of qubits.
        x_error_mask (int): Bitmask indicating which gates are followed by X errors.
                            Bits 0 - d-1: after R, Bit d: after H, Bits d+: after each CNOT.
    """
    n_circuits = 2**(n_qubits + 1 + 2*(n_qubits - 1))
    max_weight = (n_qubits - 1) // 2 - 1 
    errors = [e for e in range(n_circuits) if find_weight(e, n_qubits) < max_weight + 1 and find_weight(e, n_qubits) > 0]
    print(f"Generating {len(errors)} circuits...")
    print(f"Max weight of X errors: {max_weight}")
    circuit = stim.Circuit()
    circuit.append_operation("R", range(n_qubits))
    for e in errors:
        circuit += generate_ghz_circuit_with_x_errors(n_qubits, e, prepare=False)
        circuit.append_operation("TICK")
        circuit.append_operation("MR", range(n_qubits))
        circuit.append_operation("TICK")
        
    return circuit
    