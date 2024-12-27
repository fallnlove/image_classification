from pathlib import Path

import gdown

URLS = {
    "https://drive.google.com/uc?id=1e5Lx_Qnqls1e8nTsxZNFmPq-PqE8AvR3": "models/model_195.pth",
}


def main():
    path_gzip = Path("models/").absolute().resolve()
    path_gzip.mkdir(exist_ok=True, parents=True)

    for url, path in URLS.items():
        gdown.download(url, path)


if __name__ == "__main__":
    main()
