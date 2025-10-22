# generate_ghz_circuits.py

This module provides tools for generating and analyzing quantum GHZ state preparation circuits.

Check the `ghz_circuits.ipynb` notebook for example usage.

## Requirements
- stim
- numpy
- scipy
- pymatching

Install dependencies with:
```bash
pip install -r requirements.txt
```

# Calculation of the probability of discarding GHZ states for a given error probability

In a GHZ preparation circuit the prepared state is discarded if any of the ancilla verification qubits flips on. Thus, the discard probability of GHZ states is the probability of at least one verification qubit flipping on. Because an error $e$ of weight $w(e)$ occurs with probability $P(e) = p^{w(e)}(1-p)^{n-w(e)}$, the discard probability can be calculated as
```math
\begin{align}
\text{P}(\text{discard}) &= \sum_e \text{P}(\text{discard} | e) \cdot P(e)\\
&=  \sum_w \text{P}(\text{discard} | w(e) = w) \cdot
\binom{n}{w} p^{w}(1-p)^{n-w}
\end{align}
```

where $n$ is the total number of gates. In the case of GHZ preparation circuits $n = 2\,n_{qubits} + 3\,n_{ancilla}$.

Here, $\binom{n}{w}$ is the binomial coefficient, i.e., the number of ways to choose $w$ errors among $n$ gates.

Therefore, we need to find $\text{P}(\text{discard} | w(e) = w)$: the discard probability given that the weight of the error was $w$.

With this module, you can use the output of `run_verified_ghz(n_qubits, ver)` (check its documentation for details) to find, for a given weight, how many errors cause the state to be discarded. 
