# El Hareketi ile Araç Kontrol (Demo)

Bu proje kameradan el hareketlerini algılar:

- **Yumruk**: `STOP`
- **Açık el**: `FORWARD`
- **İşaret parmağı (diğerleri kapalı)**: Parmağın baktığı yöne göre `LEFT` / `RIGHT`
- **Algılanmıyor / belirsiz**: `IDLE`

Ekranda anlık **komut** ve **FPS** gösterilir. İsterseniz aynı komutu UDP/TCP ile bir araca/robota da gönderebilirsiniz.

## Kurulum (uv / Fedora 43)

```bash
uv python pin 3.12
uv sync
```

## Çalıştırma

Sadece ekranda gösterim:

```bash
uv run python main.py
```

UDP ile komut gönder:

```bash
uv run python main.py --send udp --host 192.168.1.50 --port 5005
```

TCP ile komut gönder:

```bash
uv run python main.py --send tcp --host 192.168.1.50 --port 5005
```

## Tuşlar

- `q`: çıkış
- `f`: görüntüyü yatay çevir (mirror)

## Notlar

- Komut metni tek satır ASCII olarak gönderilir: `STOP`, `FORWARD`, `LEFT`, `RIGHT`, `IDLE`
- Gönderim, varsayılan olarak **komut değişince** yapılır (trafik azaltmak için).

