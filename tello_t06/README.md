T0.6 — teste fisico supervisionado (experimental, NAO validado em drone)

Instalar: python3 -m pip install djitellopy==2.5.0
Executar no terminal, com Wi-Fi do Tello: python3 hover_test.py

Um takeoff, hover zero por 5 s apenas se ACK ok, um land; depois monitoramento passivo por 20 s.
Nao envia novamente takeoff ou land. Log CSV com telemetria.

ATENCAO: nao garante pouso, nem consegue interromper chamada SDK bloqueada.
A tecla K corta motores, pode causar queda. Nao testar deliberadamente.
Area ampla, sem pessoas, protetores, supervisao visual. Se ainda estiver no ar ao fim do programa, nao reiniciar o voo.
A telemetria do djitellopy e um cache; pode ficar antiga sem aviso. h=0 nao prova touchdown.
Um timeout pode ser resposta perdida ou comando nao executado; SDK nao tem ID de correlacao.
