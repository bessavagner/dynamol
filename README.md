# dynamol

 Dinâmica molecular de partículas confinadas em caixa adiabática

## Instalação

Baixe ou clone, e dentro da pasta abra um terminal/PowerShell e rode

 ~~~terminal
pip install .
 ~~~

ou

 ~~~terminal
pip install -e .
 ~~~

para instalar em modo de edição

## Uso básico

A simulação tem como base um conjunto de <img src="https://render.githubusercontent.com/render/math?math=\large N"> partículas em uma caixa cúbica (3D) ou em um retângulo (2D), cujas dimensões são construídas com base no volume molar de um gás ideal a <img src="https://render.githubusercontent.com/render/math?math=\large 273.15">k e pressão de <img src="https://render.githubusercontent.com/render/math?math=\large 101325"> Pa ou <img src="https://render.githubusercontent.com/render/math?math=\large 1"> atm, e massa molar escolhida pelo usuário. Você pode ver uma lista de moléculas disponíveis na variável ```dynamol.data.atomic_mass```, que é um dicionário cujas chaves são diferentes elementos e moléculas. A configuração inicial é de uma rede retangular, cuja separação depende do valor de densitade imputado. Condira e implementação de ```dynamol.construct.Lattice```.

Primeiramente, construa um objeto de simulação. Por enquanto, apenas ```IdealGas``` está implementado:

~~~python
import dynamol as dm
gas = dm.systems.IdealGas(N=216, temperature=4, atom='argon', compress=1.0, dim=2)
~~~

A temperatura poderá ser ajustada para garantir a estaticidade do sistema (<img src="https://render.githubusercontent.com/render/math?math=\large \vec{V}_{cm} = 0">), assim como o fator de compressão, a fim de garantir que a distância inicial entre as partículas não sejam menor do que <img src="https://render.githubusercontent.com/render/math?math=\large 1"> unidade de comprimento, e o número de partículas deve ser uma potência da dimenão. Caso forneca um número de partículas sem raiz inteira, o programa ajusta automaticamente para o próximo número com raiz inteira. Confira o 'Espaçamento inicial' no outuput do construtor, e prefira usar valores de ```compress``` que fornecam 'Espaçamento inicial' abaixo de 1.5 (ou da escolha para o raio de corte do potencial). Para executar a simulação:

~~~python
n_iteracoes = 10000
n_arquivos = 5000
gas.execute_simulation(n_iteracoes, n_files=n_arquivos)
~~~

Acima devem ser fornecidos o número de iterações (passos) da simulação e o número de arquivos de posição que devem ser gerados, que também corresponde ao número de linhas de dados das variáveis extraídas: Energias, temperatura e pressão. Todos os valores estão em dimensões reduzidas, cujo sistema de unidades pode ser conferido com

~~~python
print(gas.units)
~~~

### Dados de saída:

Todos os arquivos estão no formato ```.h5```. Os arquivos são as posições para cada certo número de passos, o qual é definido pelo arugmento ```n_files```. Na pasta 'outputs\positions\' você pode encontrar todas as confiugrações salvas. Para ler um arquivo em numpy.array:

~~~python
import h5py
import numpy as np
with h5py.File(f'outputs/positions/positions_0000.h5') as f:
    r = f['positions'][:]
~~~

Já na pasta 'outputs\variables\' contém um único arquivo: 'variables.h5,' contendo os valores das variáveis calculadas. Para plotar o gráfico da temperatura, por exemplo:

~~~python
import h5py
import matplotlib.pyplot as plt
with h5py.File('outputs/variables/variables.h5', 'r') as f:
    E = f['Temperature'][:]
fig, ax = plt.subplots(figsize=(12, 8))
ax.plot(*E.T)
plt.show()
~~~

Para ver as outras chaves do arquivo 'variables.h5':

~~~python
import h5py
with h5py.File('outputs/variables/variables.h5', 'r') as f:
    print(f.keys())
~~~

## Observações

O sistema simulado consiste de um processo <img src="https://render.githubusercontent.com/render/math?math=\large NVE">, ou seja, um *ensemble* a volume e energia constantes. O o número de simulações para o equilíbrio com a caixa pode ser muito alto, dependendo do número de partículas e da temperatura.

A integração de movimento utiliza o método de velocity-Verlet, e as interações são mediadas pelo potencial de Lennard-Jonnes de dimensões reduzidas: os parâmetros <img src="https://render.githubusercontent.com/render/math?math=\large \epsilon"> e <img src="https://render.githubusercontent.com/render/math?math=\large \sigma"> são, respectivamente, as unidades de energia e espaço.

## Autor

- [@bessavagner](https://github.com/bessavagner)
