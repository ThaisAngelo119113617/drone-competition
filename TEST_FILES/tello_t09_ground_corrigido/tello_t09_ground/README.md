# T0.9 — Diagnóstico UDP no solo

Este teste **não decola e não envia** `takeoff`, `land`, `rc` ou `emergency`. Envia apenas `command`, `battery?`, `speed?` e `time?`. Não use o aplicativo Tello ou outro script simultaneamente. Deixe o drone no solo, com espaço livre, e mantenha supervisão.

## 1. Verificar Wi-Fi no Ubuntu

Conecte-se ao Wi-Fi do Tello. Em outro terminal:

```bash
iw dev
nmcli -f IN-USE,SSID,DEVICE dev wifi
```

Identifique a interface conectada ao Tello (ex.: `wlp2s0`), então execute, substituindo `wlp2s0`:

```bash
iw dev wlp2s0 get power_save
```

Se estiver `on`, você pode desativar **temporariamente** (opcional):

```bash
sudo iw dev wlp2s0 set power_save off
```

Não mude configurações se não souber qual interface é a do Tello. O teste funciona sem essa alteração.

## 2. Rodar o teste

```bash
cd tello_t09_ground
python3 ground_udp_diagnostic.py
```

Digite `SOLO` para autorizar consultas **sem voo**. O programa envia 30 consultas por fase com pausas de 500, 200 e 50 ms. Registra `SEND`, `RESPONSE`, `TIMEOUT` e respostas fora de hora no CSV. É intencionalmente sequencial: não envia comandos em paralelo.

O script usa um socket UDP próprio ligado à porta 8889 e não precisa de `djitellopy` ou OpenCV. Se aparecer `Address already in use`, encerre os outros scripts de Tello antes de tentar novamente.

## 3. Captura passiva opcional

Em um **segundo terminal**, antes de executar o Python, substitua `wlp2s0` pela interface real:

```bash
sudo tcpdump -i wlp2s0 -n -tt -w t09_ground.pcap 'host 192.168.10.1 and udp'
```

Encerre a captura com `Ctrl+C` **depois** que o teste terminar. O `.pcap` pode conter tráfego de rede e identificadores; revise antes de compartilhar.

## 4. O que enviar

- `t09_ground_events.csv`
- Saída do terminal do teste
- Resultado de `iw dev <interface> get power_save`
- Opcionalmente, o `.pcap`

## Limitações

- Este teste mede confiabilidade/latência de consultas SDK em solo, **não** mede a entrega de `land` durante o voo.
- O UDP não inclui identificadores de comando: uma resposta antiga pode ser associada à consulta seguinte, mesmo com drenagem prévia. Por isso o CSV registra o payload e respostas inesperadas; a captura de pacotes ajuda a interpretar.
- Consultas `speed?` e `time?` podem responder de forma diferente dependendo do firmware. Isso não deve ser confundido automaticamente com perda de pacote.
- A fase de 50 ms é uma condição de carga de consultas sequenciais, **não** um teste de `rc` a 20 Hz. Não faça teste com `rc` enquanto o drone estiver no chão.
