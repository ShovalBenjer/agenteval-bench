# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Enterprise foundation files: conftest, __main__, py.typed, Makefile, pre-commit config
- SECURITY.md, CHANGELOG.md, and CONTRIBUTING.md

## [0.1.0] - 2026-08-24

### Added
- Initial release with EvalSuite, EvalRunner, EvalCase, EvalResult, RunResult models
- DeterministicScorer with exact, contains, regex, and json_schema matchers
- CLI with `run` command and CI gate support
