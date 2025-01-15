# desafio_oxebots_erick_interfaces

Este é um pacote ROS 2 que define interfaces personalizadas (ações) usadas para resolver o **Desafio Oxebots**. Especificamente, ele contém a definição da ação `MoveBase`, que é empregada para tarefas básicas de navegação, como avançar, virar à esquerda/direita ou parar o robô.

Este pacote contém **apenas** definições de interfaces (sem nós executáveis). Outros pacotes (por exemplo, o controlador do robô) irão **importar** e **usar** essas interfaces para que os seus processos se comuniquem por meio de ações.
