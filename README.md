## Projeto do componente curricular Teoria da Decisão da UFMG - Escola de Engenharia - Engenharia de Sistemas
## Problema de otimização de localização de pontos de acesso de internet sem fio para atender clientes

### Solução do problema mono-objetivo
- Implementação com BVNS
### Solução do problema bi-objetivo
- Implementação do PW e do PE a partir do BVNS
### Decisão da melhor solução
 - [Adicionar]

 #### Detalhes da saída

- A saídas dos métodos são direcionados para pasta output/[nome_metodo]/file_save
- Estrutura da saída (dict.pickle):
- Key : significado

'cc': consumo do cliente i; 
'ap': vetor binario para indicar se a PA é usada;
'acp': matrix binaria (clientesxPA) que indica se a PA atende ao cliente;
'd': vetor de distacias;
'grid': espaço entre as PAs;
'sizex': domínio de busca X;
'sizey': domínio de busca Y;
