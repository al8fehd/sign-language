# El Hareketi ile Araç Kontrol (Demo)

Bu proje kameradan el hareketlerini algılar:

- **Yumruk**: `STOP`
- **Açık el**: `FORWARD`
- **İşaret parmağı (diğerleri kapalı)**: Parmağın baktığı yöne göre `LEFT` / `RIGHT`
- **Algılanmıyor / belirsiz**: `IDLE`

Ekranda anlık **komut** ve **FPS** gösterilir. İsterseniz aynı komutu UDP/TCP ile bir araca/robota da gönderebilirsiniz.

## Kurulum

Linux ve Windows için geçerli bir Python sanal ortamı önerilir.

```bash
python -m venv .venv
source .venv/bin/activate   # Linux/macOS
# veya
.\.venv\Scripts\activate  # Windows
pip install -r requirements.txt
```

Eğer `uv` kullanmak isterseniz, proje kök dizininde aşağıdaki adımları uygulayın:

```bash
cd /home/al8fehd/Belgeler/el
uv sync
uv run python main.py
```

Bu komutlar şunları yapar:

- `uv sync`: proje bağımlılıklarını ve ortamı `uv` ile senkronize eder.
- `uv run python main.py`: `uv` ortamı içinde `main.py` dosyasını çalıştırır.

`uv` yüklü değilse veya doğrudan Python ile çalışmak isterseniz:

```bash
python main.py
```

Yerel `uv` kullanımı için daha fazla detay `RUN_WITH_UV.md` dosyasında yer alır.

UDP ile komut göndermek için:

```bash
uv run python main.py --send udp --host 192.168.1.50 --port 5005
```

TCP ile komut göndermek için:

```bash
uv run python main.py --send tcp --host 192.168.1.50 --port 5005
```

## Tuşlar

- `q`: çıkış
- `f`: görüntüyü yatay çevir (mirror)

## Notlar

- Komut metni tek satır ASCII olarak gönderilir: `STOP`, `FORWARD`, `LEFT`, `RIGHT`, `IDLE`
- Gönderim, varsayılan olarak **komut değişince** yapılır (trafik azaltmak için).

