import pennylane as qml
from pennylane import numpy as np

H = 6 * qml.Identity(1) - \
    0.5 * qml.PauliZ(1) @ qml.PauliZ(4) - \
    0.5 * qml.PauliZ(2) @ qml.PauliZ(3) - \
    0.5 * qml.PauliZ(4) @ qml.PauliZ(5) - \
    0.5 * qml.PauliZ(3) @ qml.PauliZ(4)


print(f"Definicja problemu jako QUBO dała nam Hamiltonian H={H}")

# poniewaz chcemy minimalizować energię a nie max odwrócimy wszystkie znaki

H1 = - 6 * qml.Identity(1) + \
    0.5 * qml.PauliZ(1) @ qml.PauliZ(4) + \
    0.5 * qml.PauliZ(2) @ qml.PauliZ(3) + \
    0.5 * qml.PauliZ(4) @ qml.PauliZ(5) + \
    0.5 * qml.PauliZ(3) @ qml.PauliZ(4)

print(f" Minimalizacja zamiast maksymalizacji daje nam odwrócenie wszystkich znakow dlatego nowy H1 = {H1}")

# teraz zbudujemy obwód (circuit) 

dev = qml.device("default.qubit", wires=H1.wires)

@qml.qnode(dev)
def circuit(params):
    for param, wire in zip(params, H1.wires):
        qml.RY(param, wires = wire)
    return qml.expval(H1)

# weryfikacja - sprawdzam jedno rozwiązanie np 00000

print(f"Przykladowe rozwiązanie dla 0,0,0,0,0 daje energie H1 o wartości: {circuit([0,0,0,0,0])}")

# wyszukanie rozwiązania

params = np.random.rand(len(H1.wires))
opt = qml.AdagradOptimizer(stepsize = 0.5)
epochs = 200
for epoch in range(epochs):
    params = opt.step(circuit, params)

result = circuit(params)

print(f"Najnizsza konfiguracja z wykoryzstaniem 200 epok wyszukiwania daje energie H1_min= {result}")

dev = qml.device("default.qubit", wires=H1.wires, shots=1)

@qml.qnode(dev)
def circuit(params):
    for param, wire in zip(params, H1.wires):
        qml.RY(param, wires = wire)
    return qml.sample()

correct_result = circuit(params)

print(f"Prawidlowe rozwiązanie otrzymujemy dla konfiguracji {correct_result}")
