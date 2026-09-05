{ pkgs ? import <nixpkgs> {} }:
pkgs.mkShell {
  buildInputs = with pkgs; [
    uv            # Gestor de pacotes Python ultra-rápido
    espeak-ng     # Backend de fonemas para o Kokoro
    libsndfile    # Manipulação de áudio
    ffmpeg        # Processamento de mídia
  ];
  shellHook = ''
    export LD_LIBRARY_PATH="${pkgs.espeak-ng}/lib:${pkgs.libsndfile}/lib:$LD_LIBRARY_PATH"
    echo "--- 🚀 Ryzen 7 + AVX-512 + UVX ready ---"
  '';
}

