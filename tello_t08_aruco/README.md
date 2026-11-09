# T0.8 — Leitura de ArUcos com camera do Tello

**Teste apenas no chao.** Nao envia takeoff, land ou RC.

## Instalar

```bash
python3 -m pip install djitellopy opencv-contrib-python
```

Se voce ja tem OpenCV com `cv2.aruco`, nao precisa reinstalar.

## Executar

Conecte o notebook ao Wi-Fi do Tello, entre na pasta e rode:

```bash
python3 detect_aruco.py
```

O padrao e `DICT_4X4_50` (marcadores gerados neste projeto). Para um dicionario diferente:

```bash
python3 detect_aruco.py --dict DICT_5X5_100
```

Q ou ESC fecha a janela. S salva uma imagem com os IDs marcados.

Se os marcadores da competicao nao forem detectados, confirme qual **dicionario ArUco** foi usado para produzi-los. Marcadores de dicionarios diferentes podem nao ser reconhecidos. A camera nao estima distancia real neste teste; centros sao coordenadas em pixels.
