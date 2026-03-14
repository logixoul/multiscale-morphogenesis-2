# Copilot Instructions

## Project Guidelines
- User prefers no defensive checks/safety rails for config parsing in this project and wants direct access assuming valid config.toml. 
- User wants fallback values only when a parameter table is missing in TOML; if the table exists, assume its fields are valid and access directly.