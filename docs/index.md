# Welcome to MkDocs

For full documentation visit [mkdocs.org](https://www.mkdocs.org).

## MkDocs Commands

* `mkdocs new [dir-name]` - Create a new project.
* `mkdocs serve` - Start the live-reloading docs server.
* `mkdocs build` - Build the documentation site.
* `mkdocs -h` - Print help message and exit.

## Project layout

    docs/
        │── index.md  # The documentation homepage.
        └── ...       # Other markdown pages, images and other files.
    fastgres/           # Source root of FASTgres.
    │──  archives/               #
    │──  async_components/       #
    │──  baseline/               #
    │──  models/                 #
    │──  query_encoding/         #
    └──  workloads/              #
    unittests/          # Set of unittests to ensure FASTgres robustness.
    mkdocs.yml          # Documentation configuration file.
    pyproject.toml      # Package information file.
    README.md           # Setup and usage information regarding FASTgres.
    requirements.txt    # Package requirements for FASTgres.
