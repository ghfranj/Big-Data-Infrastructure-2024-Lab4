#!/bin/bash
set -e

# Fetch credentials from HashiCorp Vault
VAULT_URL="http://vault:8200"
VAULT_TOKEN="myroot"

# Vault path for database credentials
VAULT_PATH="secret/data/database"
# Parse response to extract credentials
APP_DB_PASS=$(curl -s -H "X-Vault-Token: $VAULT_TOKEN" $VAULT_URL/v1/$VAULT_PATH | jq -r '.data.data.DB_PASSWORD')
APP_DB_USER=$(curl -s -H "X-Vault-Token: $VAULT_TOKEN" $VAULT_URL/v1/$VAULT_PATH | jq -r '.data.data.DB_USER')
APP_DB_NAME=$(curl -s -H "X-Vault-Token: $VAULT_TOKEN" $VAULT_URL/v1/$VAULT_PATH | jq -r '.data.data.DB_NAME')

# Export fetched credentials as environment variables
export APP_DB_PASS="$APP_DB_PASS"
export APP_DB_USER="$APP_DB_USER"
export APP_DB_NAME="$APP_DB_NAME"

export PGPASSWORD=$POSTGRES_PASSWORD;
psql -v ON_ERROR_STOP=1 --username "$POSTGRES_USER" --dbname "$POSTGRES_DB" <<-EOSQL
   CREATE USER $APP_DB_USER WITH PASSWORD '$APP_DB_PASS';
  CREATE DATABASE $APP_DB_NAME;
  GRANT ALL PRIVILEGES ON DATABASE $APP_DB_NAME TO $APP_DB_USER;
  ALTER DATABASE $APP_DB_NAME OWNER TO $APP_DB_USER;

  COMMIT;
EOSQL

