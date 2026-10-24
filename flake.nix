{
  description = "zmxtools: a toolkit to read Zemax OpticStudio files";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    flake-utils.url = "github:numtide/flake-utils";
  };

  outputs = { self, nixpkgs, flake-utils }@inputs:
    flake-utils.lib.eachDefaultSystem (system:
      let
        pkgs = nixpkgs.legacyPackages.${system};

        # Python with all runtime dependencies of the project available.
        pythonWithDeps = pkgs.python313.withPackages (ps: [
          ps.numpy
          ps.coloredlogs
          ps.importlib-metadata
          ps.typing-extensions
        ]);
      in
      {
        packages.default = pythonWithDeps;

        apps.python = flake-utils.lib.mkApp {
          drv = pythonWithDeps;
          name = "python";
        };

        devShells.default = pkgs.mkShell {
          packages = [
            pythonWithDeps
            pkgs.uv
            pkgs.git
          ] ++ pkgs.lib.optionals pkgs.stdenv.hostPlatform.isLinux [
            # Runtime library required by manylinux wheels (numpy et al.):
            # stdenv.cc.cc.lib provides libstdc++.so.6. Never add glibc to
            # LD_LIBRARY_PATH on NixOS: it must match the system's glibc.
            pkgs.stdenv.cc.cc.lib
          ];

          # Use the Nix-provided interpreter instead of downloading one.
          UV_PYTHON_PREFERENCE = "only-system";

          shellHook = ''
            # NixOS has no global libstdc++; manylinux wheels (numpy) load it at
            # runtime, so expose it explicitly on the library search path.
            # glibc is deliberately NOT added: it must stay the system's glibc.
            export LD_LIBRARY_PATH="${pkgs.stdenv.cc.cc.lib}/lib:''${LD_LIBRARY_PATH:-}"
            if [ ! -d .venv ]; then
              echo "zmxtools: creating .venv with uv..."
            else
              echo "zmxtools: updating .venv with uv..."
            fi
            uv sync --group docs
            uv-update() { uv lock --upgrade && uv sync --upgrade; }
            uv-lock() { uv lock --upgrade; }
            echo "uv-update: upgrade all locked packages, then resync the venv"
            echo "uv-lock: refresh uv.lock against the pyproject requirements"
          '';
        };
      });
}
