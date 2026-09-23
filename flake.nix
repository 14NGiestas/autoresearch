{
  description = "Autoresearch: Fortran CPU lab (gfortran + fpm + OpenBLAS) with a CPU-only default";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
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

        # ---- bibliotecas por stack (nome diz o que tem dentro) --------------
        cpuLibs = with pkgs; [
          gfortran
          fortran-fpm
          openblas
          python312
          uv
        ];

        rocmLibs = (with rocmPkgs.rocmPackages; [
          rocm-runtime
          clr
          rocblas
          # hipfort: expoe o HIP e as bibliotecas aceleradas em Fortran, com
          # iso_c_binding. O KERNEL continua a ser HIP C++ (e' o desenho dele),
          # mas o lado do HOST (malloc, memcpy, chamadas de biblioteca) passa a
          # ser Fortran. Verificado: hipfort-7.2.3, a mesma versao do clr e do
          # rocblas. O ROCM_PATH que ele exige ja' e' exportado abaixo.
          hipfort
        ]) ++ (with pkgs; [
          zstd
          libxml2
          # NOTE: ncurses deliberately absent — its libtinfo shadows the
          # system one via LD_LIBRARY_PATH and kills system bash (needs
          # GLIBC_2.42 from nixos-unstable). Nothing here needs it.
        ]);

        # `fpm` aponta para fortran-fpm (no nixpkgs o binario se chama
        # fortran-fpm; todo mundo digita `fpm`). Mesmo espirito do mfi.
        fpmAlias = pkgs.writeShellScriptBin "fpm" ''
          exec ${pkgs.fortran-fpm}/bin/fortran-fpm "$@"
        '';

        commonBuildInputs = [ pkgs.gfortran pkgs.fortran-fpm fpmAlias ];

        # ---- fabricas de shell ---------------------------------------------
        mkCpuShell = pkgs.mkShell {
          nativeBuildInputs = commonBuildInputs ++ [ pkgs.python312 pkgs.uv ];
          buildInputs = cpuLibs;
          shellHook = ''
            export OMP_NUM_THREADS="''${OMP_NUM_THREADS:-4}"
            export OPENBLAS_NUM_THREADS="''${OPENBLAS_NUM_THREADS:-4}"
            export EVAL_OMP="$OMP_NUM_THREADS"
            export LD_LIBRARY_PATH="${pkgs.lib.makeLibraryPath [ pkgs.openblas ]}:$LD_LIBRARY_PATH"
            echo "cpu-only shell ready (no GPU runtime; OMP=$OMP_NUM_THREADS)."
          '';
        };

        mkRocmShell = pkgs.mkShell {
          nativeBuildInputs = commonBuildInputs ++ [ pkgs.python312 pkgs.uv ];
          buildInputs = cpuLibs ++ rocmLibs;
          shellHook = ''
            export ROCM_PATH="${rocmPkgs.rocmPackages.rocm-runtime}"
            export HIP_PATH="${rocmPkgs.rocmPackages.clr}"
            # não incluir stdenv.cc.cc aqui: sombreava libstdc++ do sistema e quebrava node/pi (CXXABI_1.3.15)
            export LD_LIBRARY_PATH="${pkgs.lib.makeLibraryPath rocmLibs}:$LD_LIBRARY_PATH"
            export TORCH_USE_HIP_DSA=1
            # AMD_SERIALIZE_KERNEL ficava aqui com o valor 1. Essa variavel
            # serializa cada lancamento de kernel e obriga a uma sincronizacao
            # por kernel. E' de depuracao, para cacar races, e custa ordens de
            # grandeza. Com ela ligada o host espera a GPU em cada chamada, a
            # placa da' um spike curto e nao rampa, e toda a medida de tempo da
            # GPU mede a serializacao em vez da placa.
            export ROCM_VERSION=6.2.3
            export PYTORCH_ROCM_ARCH="gfx1100"
            export GFX_ARCH=gfx1100
            export HSA_OVERRIDE_GFX_VERSION=11.0.0
            export TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=0
            export TORCH_BLAS_PREFER_HIPBLASLT=0
            export HIP_VISIBLE_DEVICES=0
            export HIP_MEMORY_POOL_LIMIT=16000000000
            export PYTORCH_HIP_ALLOC_CONF=garbage_collection_threshold:0.9,max_split_size_mb:512
            export OMP_NUM_THREADS="''${OMP_NUM_THREADS:-4}"
            echo "rocm shell ready (GPU AMD gfx1100; torch-free)."
          '';
        };

      in
      {
        # Nomes por STACK; alias por MAQUINA aponta para o stack certo dela.
        # Os dois usam o MESMO nixpkgs rev (flake.lock): gfortran e OpenBLAS sao
        # as mesmas derivacoes em qualquer maquina -- e' isso que torna bpb e
        # tok/s comparaveis entre fermi e halfbeast.
        devShells = {
          cpu-only = mkCpuShell;
          rocm = mkRocmShell;

          # default = CPU-only (espirito do mfi): entrar no shell da maquina nao
          # pode custar o download de ~GB de runtime de GPU que o trabalho
          # Fortran nao usa. Quem quer GPU pede: `nix develop .#rocm`.
          default = mkCpuShell;

          # por maquina
          fermi = mkRocmShell;      # GPU AMD gfx1100
          halfbeast = mkCpuShell;   # Intel i9-7900X, sem GPU util

          # legado: .#inference era o nome antigo do cpu-only
          inference = mkCpuShell;
        };
      }
    );
}
