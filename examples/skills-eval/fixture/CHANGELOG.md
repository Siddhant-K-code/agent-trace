# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [Unreleased]

### Added
- `export --csv` writes one row per event.
- New `retention.days` setting deletes local data older than N days.

### Changed
- `list` shows the newest items first.

### Fixed
- `list` no longer fails on an empty data directory.

### Internal
- CI runs on a test matrix.

## [1.2.0] - 2025-01-10

### Added
- `watch` command.

### Fixed
- Timestamps in `inspect` use the local time zone.
- `export` keeps Unicode characters.
