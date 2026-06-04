{
  description = "Autoresearch ROCm dev environment";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/50ab793786d9de88ee30ec4e4c24fb4236fc2674";
    flake-utils.url = "github:numtide/flake-utils";
  };

  outputs = { self, nixpkgs, flake-utils }:
    flake-utils.lib.eachDefaultSystem (system:
      let
        pkgs = import nixpkgs {
          inherit system;
          config.allowUnfree = true;
        };

        rocmPkgs = if pkgs ? pkgsRocm then pkgs.pkgsRocm else pkgs;

        rocmLibs = with rocmPkgs.rocmPackages; [
          rocm-runtime
          clr
          rocblas
        ] ++ (with pkgs; [
          stdenv.cc.cc
          zstd
          libxml2
          ncurses
        ]);

      in
      {
        devShells.default =
          pkgs.mkShell {
            buildInputs = with pkgs; [
              uv
              python312
            ] ++ rocmLibs;

            shellHook = ''
              export ROCM_PATH="${rocmPkgs.rocmPackages.rocm-runtime}"
              export HIP_PATH="${rocmPkgs.rocmPackages.clr}"
              export LD_LIBRARY_PATH="${pkgs.lib.makeLibraryPath rocmLibs}:$LD_LIBRARY_PATH"
              export TORCH_USE_HIP_DSA=1
              export AMD_SERIALIZE_KERNEL=1
              export ROCM_VERSION=6.2.3
              export PYTORCH_ROCM_ARCH="gfx1100"
              export GFX_ARCH=gfx1100
              export HSA_OVERRIDE_GFX_VERSION=11.0.0
              export TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=0
              export TORCH_BLAS_PREFER_HIPBLASLT=0
              export HIP_VISIBLE_DEVICES=0
              export HIP_MEMORY_POOL_LIMIT=16000000000
              export PYTORCH_HIP_ALLOC_CONF=garbage_collection_threshold:0.9,max_split_size_mb:512

              # Create a venv with ROCm torch if not ready
              _venv=".venv-rocm"
              _torch_index="https://download.pytorch.org/whl/rocm6.2"
              if [ ! -f "$_venv/.ready" ]; then
                echo "Setting up ROCm venv (one-time)..."
                rm -rf "$_venv"
                python3 -m venv "$_venv"
                "$_venv/bin/pip" install --quiet --upgrade pip
                "$_venv/bin/pip" install --quiet \
                  matplotlib numpy pandas pyarrow requests rustbpe tiktoken 2>&1 | tail -3
                "$_venv/bin/pip" install --quiet "torch==2.5.1" \
                  --index-url "$_torch_index" 2>&1 | tail -3
                touch "$_venv/.ready"
                echo "ROCm venv ready."
              fi
              . "$_venv/bin/activate"

              echo "ROCm autoresearch shell ready."
            '';
          };
      }
    );
}
