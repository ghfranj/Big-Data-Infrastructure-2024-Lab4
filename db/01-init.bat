@echo off
setlocal enabledelayedexpansion

REM Fetch credentials from HashiCorp Vault
REM set VAULT_URL=http://vault:8200
REM set VAULT_TOKEN=myroot
REM set VAULT_PATH=secret/data/database

REM Parse response to extract credentials
for /f "tokens=*" %%A in ('curl -s -H "X-Vault-Token: %VAULT_TOKEN%" %VAULT_URL%/v1/%VAULT_PATH% ^| jq -r ".data.data.DB_PASSWORD"') do set "APP_DB_PASS=%%A"
for /f "tokens=*" %%B in ('curl -s -H "X-Vault-Token: %VAULT_TOKEN%" %VAULT_URL%/v1/%VAULT_PATH% ^| jq -r ".data.data.DB_USER"') do set "APP_DB_USER=%%B"
for /f "tokens=*" %%C in ('curl -s -H "X-Vault-Token: %VAULT_TOKEN%" %VAULT_URL%/v1/%VAULT_PATH% ^| jq -r ".data.data.DB_NAME"') do set "APP_DB_NAME=%%C"

REM Export fetched credentials as environment variables
echo got from vault !APP_DB_NAME!
echo got from vault !APP_DB_USER!
echo got from vault !APP_DB_PASS!

psql -v ON_ERROR_STOP=1 --username "%POSTGRES_USER%" --dbname "%POSTGRES_DB%" <<-EOSQL
   CREATE USER !APP_DB_USER! WITH PASSWORD '!APP_DB_PASS!';
  CREATE DATABASE !APP_DB_NAME!;
  GRANT ALL PRIVILEGES ON DATABASE !APP_DB_NAME! TO !APP_DB_USER!;
  ALTER DATABASE !APP_DB_NAME! OWNER TO !APP_DB_USER!;
  \connect !APP_DB_NAME! !APP_DB_USER!;

  COMMIT;
EOSQL
