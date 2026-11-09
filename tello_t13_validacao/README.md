# T1.3 — Validacao do mapa do circulo e do ID 03

**Somente solo, motores desligados.** Este teste NAO implementa EKF nem comandos de voo.

## Executar

Conecte ao Wi-Fi do Tello, ative o ambiente Python que ja usa para os testes e execute:

```bash
python3 t13_mapping.py
```

Digite `SOLO`. `S` salva imagem, `Q` encerra. Gera `t13_map.csv`.

## Configuracao

- ArUco `DICT_4X4_50`, lado preto a preto de 0,115 m.
- ID 00: marcador inferior do circulo; ID 01: superior. Centros separados por 0,55 m.
- ID 03: marcador do obstaculo seguinte, aproximadamente 2,2 m da posicao inicial.
- Circulo: diametro interno aproximado 0,45 m.
- Meta futura de parada: 0,75 m **em profundidade optica** ate o ID 03, nao em distancia euclidiana. A diferenca importa se a camera estiver inclinada.
- Calibracao: `tello_calibration.npz` para 960x720.

## Protocolo

1. Posicione o Tello desligado para voo, em frente ao circulo, e deixe os tres IDs visiveis. Verifique se `ID03 Z` fica perto de 2,2 m e `ID0-ID1` perto de 0,55 m. A medida de 2,2 m e aproximada.
2. Salve uma captura. Mova o drone **manualmente**, sem ligar motores, 20-30 cm para esquerda e direita, e para frente/tras. Salve capturas em cada posicao.
3. Verifique `id3_circle_depth_m`: deve ser aproximadamente constante quando o drone muda de posicao, se os obstaculos forem fixos e o referencial estiver bem estimado.
4. Envie `t13_map.csv` e 2-3 capturas para analise.

## Interpretacao e limites

`MAP_READY` significa apenas que os tres marcadores foram detectados e a transformacao foi calculada. **Nao** significa que a travessia e segura.

O ponto medio dos IDs 00 e 01 so e o centro livre se os marcadores forem simetricos. A folga teorica de 13,5 cm por lado usa uma largura de drone aproximada de 18 cm e ignora protetores, oscilacao e erro de estimativa. Confirme fisicamente a menor abertura livre, a largura real do drone e a distancia entre a barra azul e o trajeto. O marcador 03 esta em outro plano; nao deve ser usado como alvo de passagem do circulo.

Este teste nao resolve as incertezas anteriores de comando de pouso. Nao realize travessia automatica ate validar o controle e a parada em ambiente livre.
