# T1.4 — EKF de centro do círculo (somente SOLO)

## Executar

Instale `djitellopy`, `opencv-contrib-python`, `numpy`. Conecte ao Wi-Fi do Tello, mantenha motores desligados e execute:

```bash
python3 t14_ekf.py
```

Digite `SOLO`. `Q` encerra, `S` salva screenshot. O arquivo `t14_ekf.csv` registra estimativa, desvio-padrão, observações aceitas/rejeitadas e sugestão de correção **não executada**.

## Teste sugerido

1. Aponte o drone parado para o círculo e observe IDs 0+1.
2. Oculte apenas ID 1: ID 0 deve continuar estimando centro ~27,5 cm acima do marcador inferior.
3. Oculte apenas ID 0: ID 1 deve continuar estimando centro ~27,5 cm abaixo do marcador superior.
4. Desloque manualmente o drone para cima/baixo e esquerda/direita, com motores desligados; verifique se a indicação acompanha o deslocamento.
5. Mostre apenas ID 3 e observe `3:VISIBLE_NOT_FUSED`: o mapa ainda não está calibrado em referencial rígido para fusão segura durante rotações.

## Limitações importantes

- Estado EKF `[x,y,z,vx,vy,vz]` em coordenadas **da câmera**, com modelo de velocidade constante. Isso é uma aproximação local para movimentos pequenos, não um EKF de navegação 6-DoF.
- A inferência de centro por um marcador assume câmera aproximadamente frontal, marcadores verticais, centros simétricos e separação 55 cm. Para inclinação/yaw significativos é preciso uma pose rígida do conjunto.
- ID 03 é observado para checagem, mas **não é fundido** no EKF: sua posição relativa em coordenadas da câmera muda com a rotação e não há extrínsecos calibrados. O programa não afirma manter localização só com ID 03.
- `sigma` do filtro é incerteza **modelada**, não erro absoluto medido. Não use para certificar folga de 45 cm.
- O programa **não** chama takeoff, land, RC nem move o drone. Não o utilize para atravessar o círculo.
- O alvo de parada 75 cm do ID 03 será implementado na máquina de estados depois de validar o mapa 3D e segurança da trajetória.
