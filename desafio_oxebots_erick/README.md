# desafio_oxebots_erick

Este pacote ROS 2 fornece um nó baseado em Python, `robot_controller`, que utiliza um servidor de ações (e cliente) para navegar em um ambiente de labirinto com base no algoritmo "Hand on Wall". O pacote utiliza [interface de ação MoveBase](https://github.com/ros2/ros2_documentation/blob/humble/source/Tutorials/Intermediate/Writing-an-Action-Server-Client.rst) para enviar objetivos (avançar, virar à esquerda, virar à direita, etc.) e publica comandos de velocidade de acordo para mover o robô.

## Índice

- [desafio\_oxebots\_erick](#desafio_oxebots_erick)
  - [Índice](#índice)
  - [Visão Geral](#visão-geral)
  - [Uso](#uso)
  - [Nós e Tópicos](#nós-e-tópicos)
    - [Nó: `robot_controller`](#nó-robot_controller)
      - [Tópicos Inscritos](#tópicos-inscritos)
      - [Tópicos Publicados](#tópicos-publicados)
      - [Ações](#ações)
  - [Configuração](#configuração)

---

## Visão Geral

O **Controlador do Robô** implementa uma Máquina de Estados Finitos (FSM) simples para navegar em um labirinto. O nó se inscreve tanto em odometria quanto em dados de laser, e então decide qual ação tomar a seguir:

1. **Verificar a posição atual** em relação a um limite (neste exemplo, a posição `y` com um alvo de 10 metros).  
2. **Observar as leituras do laser** para decidir se precisa virar ou avançar.  
3. **Enviar a meta de ação correspondente** (por exemplo, `MOVE_FORWARD`, `TURN_LEFT`, `TURN_RIGHT`) para o servidor de ações incluído.  

O nó usa um servidor de ações (`/move_base_action`) para processar essas metas. Ele também atua como cliente de ações para enviar as metas para si mesmo (modelo de servidor-cliente auto-contido).

---

## Uso

Após configurar seu workspace, você pode executar o nó de solução usando o seguinte comando:

```bash
ros2 launch desafio_oxebots_erick maze_solver.launch.yaml
```

---

## Nós e Tópicos

### Nó: `robot_controller`

O nó principal que implementa a FSM, se inscreve em dados de odometria e laser, e hospeda o servidor `/move_base_action`.

#### Tópicos Inscritos

- **`/odom`** ([`nav_msgs/Odometry`](http://docs.ros.org/en/api/nav_msgs/html/msg/Odometry.html))  
  Utilizado para obter a posição e orientação atuais do robô.

- **`/scan`** ([`sensor_msgs/LaserScan`](http://docs.ros.org/en/api/sensor_msgs/html/msg/LaserScan.html))  
  Utilizado para detectar obstáculos à frente e ao lado esquerdo.

#### Tópicos Publicados

- **`/cmd_vel`** ([`geometry_msgs/Twist`](http://docs.ros.org/en/api/geometry_msgs/html/msg/Twist.html))  
  Utilizado para controlar a velocidade linear e angular do robô.

#### Ações

- **Servidor**: `MoveBase` (`/move_base_action`)  
  Permite que clientes externos ou internos solicitem metas de movimento:
  - `STOP`
  - `MOVE_FORWARD`
  - `TURN_LEFT`
  - `TURN_RIGHT`

- **Cliente**: envia metas para o mesmo servidor `/move_base_action` (auto-contido):
  - É assim que os estados da FSM comandam o robô a mover-se ou virar.

---

## Configuração

Alguns parâmetros de configuração podem ser ajustados para ajustar o comportamento do robô direto no arquivo de lançamento `maze_solver.launch.yaml`:

- `distance_threshold`: distância mínima para considerar um obstáculo ou iniciar uma curva.
- `end_position_y`: coordenada `y` alvo para considerar a navegação no labirinto completa.
- `tb3_max_lin_vel` e `tb3_max_ang_vel`: velocidades lineares e angulares máximas do robô.
- `update_frequency`: frequência com a qual a máquina de estados é atualizada.
