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
              gfortran
              fortran-fpm
              openblas
            ] ++ rocmLibs;

            shellHook = ''
              export ROCM_PATH="${rocmPkgs.rocmPackages.rocm-runtime}"
              export HIP_PATH="${rocmPkgs.rocmPackages.clr}"
              # não incluir stdenv.cc.cc aqui: sombreava libstdc++ do sistema e quebrava node/pi (CXXABI_1.3.15)
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

              # torch-free shell: training/inference moved to pure Fortran (src/).
              # No venv bootstrap — python3 here is for numpy-only tooling (parity
              # harness, future .pt export reader). See hyp_34ea7c.

              echo "ROCm autoresearch shell ready (torch-free)."
            '';
          };
      }
    );
}
