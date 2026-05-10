# El Hareketi ile Araç Kontrol (Demo)

Bu proje kameradan el hareketlerini algılar:

- **Yumruk**: `STOP`
- **Açık el**: `FORWARD`
- **İşaret parmağı (diğerleri kapalı)**: Parmağın baktığı yöne göre `LEFT` / `RIGHT`
- **Algılanmıyor / belirsiz**: `IDLE`

Ekranda anlık **komut** ve **FPS** gösterilir. İsterseniz aynı komutu UDP/TCP ile bir araca/robota da gönderebilirsiniz.

## Kurulum (Windows)

```bash
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
```

## Çalıştırma

Sadece ekranda gösterim:

```bash
python main.py
```

UDP ile komut gönder:

```bash
python main.py --send udp --host 192.168.1.50 --port 5005
```

TCP ile komut gönder:

```bash
python main.py --send tcp --host 192.168.1.50 --port 5005
```

## Tuşlar

- `q`: çıkış
- `f`: görüntüyü yatay çevir (mirror)

## Notlar

- Komut metni tek satır ASCII olarak gönderilir: `STOP`, `FORWARD`, `LEFT`, `RIGHT`, `IDLE`
- Gönderim, varsayılan olarak **komut değişince** yapılır (trafik azaltmak için).

