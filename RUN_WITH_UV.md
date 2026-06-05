# UV ile Çalıştırma

Bu proje kök dizininde `uv` komutu yüklüyse şu adımları kullanın:

```bash
cd /home/al8fehd/Belgeler/el
uv sync
uv run python main.py
```

Açıklama:

- `uv sync`: `uv` ortamını ve proje bağımlılıklarını senkronize eder.
- `uv run python main.py`: senkronize edilmiş `uv` ortamı içinde `main.py` çalıştırılır.

Eğer `uv` yoksa veya sadece Python ile çalışmak isterseniz:

```bash
python main.py
```

Not:

- Bu dosya yerel kullanım içindir.
- `RUN_WITH_UV.md` dosyası `.gitignore` içine eklenmiştir, bu yüzden git tarafından izlenmez.
