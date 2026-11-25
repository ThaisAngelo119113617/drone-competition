T0.7 — revisão experimental do T0.6. NAO VALIDADO EM VOO.

Arquivos: tello_io.py, safety.py, hover_test.py.

Instale djitellopy==2.5.0; conecte ao Wi-Fi do Tello.
Execute: python3 hover_test.py

Alterações:
- terminal restaurado na thread principal por safety.stop(), inclusive em Ctrl+C normal;
- interrupção do monitoramento pós-pouso tratada;
- ACK de land invalidado se emergency solicitado, inclusive durante espera;
- Safety.should_stop() e consume_hover_request() para missões futuras;
- preflight rejeita h ausente/fora de -10..20 cm e tof ausente/fora de 0..300 cm;
- mantém um único takeoff e um único land, sem retries automáticos;
- observa telemetria durante 20 segundos depois de land (cache sem timestamp confiável).

LIMITAÇÕES IMPORTANTES:
- Um ACK 'ok' não confirma pouso; cache de telemetria não comprova frescor.
- O watchdog não interrompe comandos bloqueados; a tecla K é corte de motores e pode causar queda.
- O protocolo UDP não correlaciona ACKs a comandos. ACKs podem ser atrasados.
- Se o processo for encerrado à força, nenhuma limpeza é garantida.
- Se a decolagem for incerta, o código ainda tenta um land único, não assume que está no solo.
- Durante monitoramento pós-pouso não há comando de voo normal; somente K é caminho excepcional.
- Não testar K deliberadamente. Somente voo curto supervisionado em ambiente livre e com protetores.
- Se houver perda de controle ou estado incerto, NÃO reiniciar o voo; seguir procedimento de recuperação do fabricante.

Marcadores: PDF 12 x 12 cm do quadrado preto (inclui borda preta do ArUco), DICT_4X4_50, IDs 0-19. Imprimir A4 em tamanho real 100%; manter margem branca ao redor.
